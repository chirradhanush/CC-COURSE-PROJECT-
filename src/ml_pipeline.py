#!/usr/bin/env python

from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder


def get_paths():
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "cleaned_csv": root / "data" / "cleaned" / "smart_parking_clean.csv",
        "model_dir": root / "ML" / "models" / "parking_rf_model_rich",
    }


def create_spark():
    spark = (
        SparkSession.builder
        .appName("smart_parking_ml_rich_features")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    paths = get_paths()
    cleaned_csv = paths["cleaned_csv"]

    print(f"[ml_pipeline] Reading rich cleaned data from: {cleaned_csv}")

    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"{cleaned_csv} not found. "
            "Please generate smart_parking_clean.csv with your ingestion/EDA notebook."
        )

    spark = create_spark()

    # Load rich cleaned CSV exactly like in your notebook
    df_full = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(cleaned_csv))
    )

    print(f"[ml_pipeline] Loaded rows={df_full.count()}, cols={len(df_full.columns)}")
    print("[ml_pipeline] Columns:", df_full.columns)

    # 1) Rename target_occupied -> label
    if "target_occupied" not in df_full.columns:
        raise ValueError("Expected 'target_occupied' column not found in smart_parking_clean.csv")

    df_full = df_full.withColumnRenamed("target_occupied", "label")

    # 2) Drop leakage columns (same as ml.py)
    leaky_cols = ["Status", "_occupied", "_occupied_imp"]
    for c in leaky_cols:
        if c in df_full.columns:
            df_full = df_full.drop(c)

    # 3) Feature lists (from your notebook / ml.py)
    numeric_features = [
        "Hour", "DayOfWeek", "IsWeekend", "Month", "DayOfMonth",
        "DwellMinutes", "InViolation",
        "_duration_min", "_hour", "_dow", "_is_weekend",
        "_duration_min_w", "_duration_robust_z",
        "rolling_occ_N10", "arrivals_N10",
    ]

    cat_features = [
        "PartOfDay", "AreaName", "SideName", "SideOfStreetCode",
        "street_marker_lumped", "device_id_lumped",
    ]

    # Only keep the ones that actually exist (in case some were not present)
    existing_numeric = [c for c in numeric_features if c in df_full.columns]
    existing_cat = [c for c in cat_features if c in df_full.columns]

    missing_numeric = [c for c in numeric_features if c not in df_full.columns]
    missing_cat = [c for c in cat_features if c not in df_full.columns]

    if missing_numeric or missing_cat:
        print("[ml_pipeline] WARNING: Some expected feature columns are missing:")
        if missing_numeric:
            print("  Missing numeric:", missing_numeric)
        if missing_cat:
            print("  Missing categorical:", missing_cat)

    required_cols = ["label"] + existing_numeric + existing_cat

    df_full = df_full.dropna(subset=required_cols)
    df_full = df_full.withColumn("label", F.col("label").cast("double"))

    print(f"[ml_pipeline] After dropping NA in required features: rows={df_full.count()}")

    # 4) Index categorical features
    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep",
        )
        for c in existing_cat
    ]

    # 5) Assemble numeric + indexed categorical into features_raw, then scale
    feature_cols = existing_numeric + [f"{c}_idx" for c in existing_cat]

    print("[ml_pipeline] Using numeric features:", existing_numeric)
    print("[ml_pipeline] Using categorical features:", existing_cat)
    print("[ml_pipeline] Final feature columns:", feature_cols)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=False,
        withStd=True,
    )

    preprocess_pipeline = Pipeline(stages=indexers + [assembler, scaler])
    preprocess_model = preprocess_pipeline.fit(df_full)
    df_prepared = preprocess_model.transform(df_full)

    print("[ml_pipeline] Schema after preprocessing:")
    df_prepared.printSchema()

    # 6) Train / test split (same as your notebook: 70/30)
    train_df, test_df = df_prepared.randomSplit([0.7, 0.3], seed=42)
    print(f"[ml_pipeline] Train rows={train_df.count()}, Test rows={test_df.count()}")

    # 7) Tuned Random Forest (your exact code logic)
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        seed=42,
        featureSubsetStrategy="sqrt",
        probabilityCol="rf_prob",
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [100, 200])
        .addGrid(rf.maxDepth, [8, 12])
        .build()
    )

    tvs = TrainValidationSplit(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=1,
    )

    print("[ml_pipeline] Training tuned RandomForest...")
    tvs_model = tvs.fit(train_df)
    best_rf_model = tvs_model.bestModel

    print("Best RandomForest params:")
    print("  numTrees:", best_rf_model.getNumTrees)
    print("  maxDepth:", best_rf_model.getOrDefault("maxDepth"))

    # 8) Evaluate on held-out test set
    rf_preds = best_rf_model.transform(test_df).cache()

    print("[ml_pipeline] Sample predictions:")
    show_cols = ["label", "prediction", "rf_prob"]
    for c in ["street_marker_lumped", "device_id_lumped", "PartOfDay"]:
        if c in rf_preds.columns:
            show_cols.insert(0, c)
    rf_preds.select(*show_cols).show(10, truncate=False)

    auc_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )

    auc = auc_eval.evaluate(rf_preds)
    acc = acc_eval.evaluate(rf_preds)
    f1 = f1_eval.evaluate(rf_preds)

    print("\n==== Tuned Random Forest (Rich Features) ====")
    print(f"AUC:      {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print("============================================\n")

    print("Confusion matrix (prediction vs label):")
    rf_preds.groupBy("prediction", "label").count().orderBy("prediction", "label").show()

    # 9) Save the best model
    model_dir = paths["model_dir"]
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ml_pipeline] Saving best model to: {model_dir}")
    best_rf_model.write().overwrite().save(str(model_dir))

    print("[ml_pipeline] DONE.")
    spark.stop()


if __name__ == "__main__":
    main()
