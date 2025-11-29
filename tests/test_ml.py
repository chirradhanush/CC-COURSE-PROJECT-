
import os
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def _create_spark():
    return (
        SparkSession.builder
        .appName("smart_parking_tests_ml")
        .master("local[*]")
        .getOrCreate()
    )


def test_predictions_artifact_and_metrics():
    """
    Read data/model/predictions.csv (created by ml_pipeline.py),
    check that it exists, has expected columns, and that the accuracy
    is above a reasonable sanity threshold (e.g., 0.60).
    """
    path = "data/model/predictions.csv"
    assert os.path.exists(path), f"Expected predictions file at {path}, but not found."

    spark = _create_spark()
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        required_cols = ["prediction", "label"]
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"Missing required columns in predictions.csv: {missing}"

        df_clean = df.select("prediction", "label").dropna()
        assert df_clean.count() > 0, "No valid rows in predictions.csv after dropna."

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy",
        )
        accuracy = evaluator.evaluate(df_clean)
        assert accuracy >= 0.60, f"Model accuracy too low in predictions.csv: {accuracy:.4f}"
    finally:
        spark.stop()
