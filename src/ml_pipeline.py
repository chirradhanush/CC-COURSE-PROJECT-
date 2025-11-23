#!/usr/bin/env python

"""
ml_pipeline.py
--------------
Train and evaluate a classification model for Smart Parking occupancy.

INPUT:
    data/train_ready/   (parquet files written by prepare_data.py)

STEPS:
    - Load train_ready dataset
    - Clean / cast columns
    - Build Spark ML pipeline:
        * StringIndexer for categorical features
        * VectorAssembler for all features
        * (optional) StandardScaler
        * RandomForestClassifier (final model)
    - Train / test split
    - Evaluate (Accuracy, F1, AUC)
    - Print confusion matrix
    - Save trained model under ML/models/parking_rf_model
"""

from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


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

    print(f"[ml_pipeline] Using train_ready from: {train_ready_path}")

    spark = create_spark()

    # ===========================
    # 1. LOAD DATA
    # ===========================
    df = spark.read.parquet(str(train_ready_path))

    print(f"[ml_pipeline] Loaded rows = {df.count()}, columns = {len(df.columns)}")
    print("[ml_pipeline] Columns:", df.columns)

    # Ensure label column exists
    if "target_occupied" not in df.columns:
        raise ValueError("Expected column 'target_occupied' not found in train_ready data.")

    # ===========================
    # 2. BASIC CLEANING
    # ===========================
    # Drop rows with null label
    df = df.filter(F.col("target_occupied").isNotNull())

    # Cast label to double and rename to "label" (Spark ML convention)
    df = df.withColumn("label", F.col("target_occupied").cast("double"))

    # Drop rows with missing critical numerical features if needed
    # (You can adjust this list if you see lots of nulls)
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

    # ===========================
    # 3. DEFINE FEATURES
    # ===========================
    # Categorical columns to index
    cat_cols = []
    if "device_id" in df.columns:
        cat_cols.append("device_id")
    if "street_marker" in df.columns:
        cat_cols.append("street_marker")
    if "PartOfDay" in df.columns:
        cat_cols.append("PartOfDay")

    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep"
        )
        for c in cat_cols
    ]

    indexed_cat_cols = [f"{c}_idx" for c in cat_cols]

    feature_cols = numeric_cols + indexed_cat_cols

    print("[ml_pipeline] Numeric feature columns:", numeric_cols)
    print("[ml_pipeline] Categorical feature columns:", cat_cols)
    print("[ml_pipeline] All feature columns:", feature_cols)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )

    # StandardScaler is mostly useful for linear models, but it's fine to use here
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False,
    )

    # ===========================
    # 4. MODEL
    # ===========================
    # You can tweak hyperparameters if you like
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42,
    )

    # Full pipeline
    stages = indexers + [assembler, scaler, rf]
    pipeline = Pipeline(stages=stages)

    # ===========================
    # 5. TRAIN / TEST SPLIT
    # ===========================
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print(f"[ml_pipeline] Train rows = {train_df.count()}, Test rows = {test_df.count()}")

    # ===========================
    # 6. FIT MODEL
    # ===========================
    print("[ml_pipeline] Training RandomForest model...")
    model = pipeline.fit(train_df)

    # ===========================
    # 7. EVALUATE
    # ===========================
    pred_df = model.transform(test_df).cache()

    print("[ml_pipeline] Example predictions:")
    pred_df.select("device_id", "street_marker", "label", "prediction", "probability").show(10, truncate=False)

    # AUC (Binary)
    bin_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = bin_eval.evaluate(pred_df)

    # Accuracy & F1
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

    accuracy = acc_eval.evaluate(pred_df)
    f1 = f1_eval.evaluate(pred_df)

    print("\n================ ML METRICS ================")
    print(f"AUC        : {auc:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print("============================================\n")

    # Confusion matrix
    print("[ml_pipeline] Confusion matrix (label vs prediction):")
    cm = (
        pred_df.groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
    )
    cm.show()

    # ===========================
    # 8. SAVE MODEL
    # ===========================
    model_dir = paths["model_dir"]
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ml_pipeline] Saving model to: {model_dir}")
    # Overwrite if exists
    model.write().overwrite().save(str(model_dir))

    print("[ml_pipeline] ✔ DONE")
    spark.stop()


if __name__ == "__main__":
    main()
