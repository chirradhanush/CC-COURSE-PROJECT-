#StringIndexer encodes categorical columns like street_marker and hour_bucket.

#VectorAssembler combines numeric features into one vector.

#StandardScaler normalizes the feature space.

#LogisticRegression trains a binary classifier on label = target_occupied.

#Added class weights to balance occupied / not-occupied samples.”


from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def main():
    spark = (
        SparkSession.builder
        .appName("parking_ml_training_better")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Load data
    df_raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv("data/ml_ready/parking_ml_ready.csv")
    )

    print("[ml] Loaded training data schema:")
    df_raw.printSchema()

    # Rename target to label
    df = df_raw.withColumnRenamed("target_occupied", "label")

    # Basic cleanup
    df = df.dropna(subset=["label", "hour", "dow", "duration_min", "street_marker"])

    # 2. Feature engineering (new columns)

    # (a) log_duration_min to stabilize long stays
    df = df.withColumn(
        "log_duration_min",
        F.log1p(F.col("duration_min"))  # log(1 + x)
    )

    # (b) hour_bucket as categorical string (e.g. "morning","midday","evening","overnight")
    df = df.withColumn(
        "hour_bucket",
        F.when((F.col("hour") >= 7) & (F.col("hour") < 11), F.lit("morning"))
         .when((F.col("hour") >= 11) & (F.col("hour") < 16), F.lit("midday"))
         .when((F.col("hour") >= 16) & (F.col("hour") < 21), F.lit("evening"))
         .otherwise(F.lit("overnight"))
    )

    # (c) is_peak flag: busy business-ish hours Mon-Fri 8am-6pm
    df = df.withColumn(
        "is_peak",
        (
            (F.col("dow").isin(2.0,3.0,4.0,5.0,6.0)) &  # Spark dow might be 1=Sun ... 7=Sat OR 0-6; your data shows dow as doubles like 0.0,2.0,5.0 so we'll just trust pattern you saw.
            (F.col("hour") >= 8.0) &
            (F.col("hour") <= 18.0) &
            (F.col("is_weekend") == 0)
        ).cast("int")
    )

    # (d) interactions
    df = df.withColumn("hour_x_weekend", F.col("hour") * F.col("is_weekend").cast("double"))
    df = df.withColumn("dow_x_weekend",  F.col("dow")  * F.col("is_weekend").cast("double"))

    # 3. Class weights for imbalance
    # We'll calculate weight = (total_rows / (2 * class_count(label)))
    label_counts = df.groupBy("label").count().collect()
    total = sum(row["count"] for row in label_counts)
    counts = {row["label"]: row["count"] for row in label_counts}
    w0 = total / (2.0 * counts.get(0, 1))
    w1 = total / (2.0 * counts.get(1, 1))

    df = df.withColumn(
        "class_weight",
        F.when(F.col("label") == 1, F.lit(w1)).otherwise(F.lit(w0))
    )

    print("[ml] Class weights:")
    print("  label=0 ->", w0)
    print("  label=1 ->", w1)

    # 4. Stratified-ish split:
    # We'll tag each row with a random number per class,
    # then split each class, then union back.
    df = df.withColumn("rand", F.rand(seed=42))

    pos = df.filter(F.col("label") == 1)
    neg = df.filter(F.col("label") == 0)

    pos_train = pos.filter(F.col("rand") < 0.7)
    pos_test  = pos.filter(F.col("rand") >= 0.7)

    neg_train = neg.filter(F.col("rand") < 0.7)
    neg_test  = neg.filter(F.col("rand") >= 0.7)

    train_df = pos_train.unionByName(neg_train).drop("rand")
    test_df  = pos_test.unionByName(neg_test).drop("rand")

    print(f"[ml] train rows: {train_df.count()}  test rows: {test_df.count()}")

    # 5. Build ML pipeline
    # Categorical indexers:
    street_indexer = StringIndexer(
        inputCol="street_marker",
        outputCol="street_marker_idx",
        handleInvalid="keep"
    )

    hour_bucket_indexer = StringIndexer(
        inputCol="hour_bucket",
        outputCol="hour_bucket_idx",
        handleInvalid="keep"
    )

    # Final feature columns (numeric + indexed categorical)
    feature_cols = [
        "hour",
        "dow",
        "is_weekend",
        "duration_min",
        "log_duration_min",
        "is_peak",
        "hour_x_weekend",
        "dow_x_weekend",
        "street_marker_idx",
        "hour_bucket_idx"
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
        handleInvalid="keep"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    # Logistic Regression with class weighting
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        predictionCol="prediction",
        probabilityCol="probability",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0  # pure L2 for stability
    )

    pipeline = Pipeline(stages=[
        street_indexer,
        hour_bucket_indexer,
        assembler,
        scaler,
        lr
    ])

    # 6. Train
    model = pipeline.fit(train_df)

    # 7. Evaluate on test set
    scored_test = model.transform(test_df)

    print("[ml] Sample predictions:")
    scored_test.select(
        "device_id",
        "street_marker",
        "hour",
        "dow",
        "is_weekend",
        "duration_min",
        "label",
        "probability",
        "prediction"
    ).show(10, truncate=False)

    # AUC
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="probability",
        metricName="areaUnderROC"
    )
    auc = evaluator_auc.evaluate(scored_test)

    # Confusion matrix counts
    cm_counts = (
        scored_test
        .select(
            F.col("prediction").cast("int").alias("pred"),
            F.col("label").cast("int").alias("label")
        )
        .groupBy("pred", "label")
        .count()
    )
    print("[ml] Confusion matrix counts:")
    cm_counts.show()

    cm_local = { (row["pred"], row["label"]): row["count"] for row in cm_counts.collect() }
    TP = cm_local.get((1,1), 0)
    FP = cm_local.get((1,0), 0)
    FN = cm_local.get((0,1), 0)
    TN = cm_local.get((0,0), 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    print("[ml] Metrics:")
    print(f"  AUC        = {auc:.4f}")
    print(f"  Accuracy   = {accuracy:.4f}")
    print(f"  Precision  = {precision:.4f}")
    print(f"  Recall     = {recall:.4f}")

    print("[ml] Training finished successfully (no model.save() on Windows).")

    spark.stop()

if __name__ == "__main__":
    main()
