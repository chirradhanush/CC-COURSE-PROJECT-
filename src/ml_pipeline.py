#!/usr/bin/env python

from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
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
        "train_ready": root / "data" / "train_ready",
        "model_dir": root / "ML" / "models" / "parking_rf_model",
    }


def create_spark():
    spark = (
        SparkSession.builder
        .appName("smart_parking_ml_pipeline")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    paths = get_paths()
    train_ready_path = paths["train_ready"]

    print(f"[ml_pipeline] Reading train_ready from: {train_ready_path}")

    spark = create_spark()

    # Load data
    df = spark.read.parquet(str(train_ready_path))
    df = df.filter(F.col("target_occupied").isNotNull())
    df = df.withColumn("label", F.col("target_occupied").cast("double"))

    # Numeric features
    numeric_cols = [
        "duration_min",
        "hour",
        "dow",
        "is_weekend",
        "InViolation",
        "DwellMinutes",
        "Month",
        "DayOfMonth",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for c in numeric_cols:
        df = df.filter(F.col(c).isNotNull())

    # Categorical features
    cat_cols = []
    if "device_id" in df.columns:
        cat_cols.append("device_id")
    if "street_marker" in df.columns:
        cat_cols.append("street_marker")
    if "PartOfDay" in df.columns:
        cat_cols.append("PartOfDay")

    print("Numeric:", numeric_cols)
    print("Categorical:", cat_cols)

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]
    indexed_cat = [f"{c}_idx" for c in cat_cols]

    feature_cols = numeric_cols + indexed_cat
    print("ALL FEATURES:", feature_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False,
    )

    # ------------------------------------------------------------
    # YOUR HIGH-ACCURACY RF MODEL EXACTLY AS YOU PROVIDED
    # ------------------------------------------------------------
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        seed=42,
        featureSubsetStrategy="sqrt",
        probabilityCol="rf_prob"
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [100, 200])
        .addGrid(rf.maxDepth, [8, 12])
        .build()
    )

    pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=1
    )

    print("[ml_pipeline] Training tuned RF model...")
    tvs_model = tvs.fit(train_df)
    best_rf_model = tvs_model.bestModel

    print("Best params:")
    print("  numTrees:", best_rf_model.stages[-1].getNumTrees)
    print("  maxDepth:", best_rf_model.stages[-1].getOrDefault("maxDepth"))

    # Predictions
    preds = best_rf_model.transform(test_df)

    auc_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    auc = auc_eval.evaluate(preds)
    acc = acc_eval.evaluate(preds)
    f1 = f1_eval.evaluate(preds)

    print("\n====== FINAL METRICS (TUNED RF) ======")
    print(f"AUC:      {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print("=====================================\n")

    print("Confusion Matrix:")
    preds.groupBy("prediction", "label").count().orderBy("prediction", "label").show()

    # Save model
    model_dir = paths["model_dir"]
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    best_rf_model.write().overwrite().save(str(model_dir))

    print("[ml_pipeline] DONE.")
    spark.stop()


if __name__ == "__main__":
    main()
