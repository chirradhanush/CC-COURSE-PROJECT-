
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def _create_spark():
    return (
        SparkSession.builder
        .appName("smart_parking_tests_ingestion")
        .master("local[*]")
        .getOrCreate()
    )


def test_cleaned_parquet_exists():
    path = "data/cleaned/smart_parking_clean.parquet"
    assert os.path.exists(path), f"Expected cleaned parquet at {path}, but file/folder not found."


def test_cleaned_schema_and_basic_quality():
    spark = _create_spark()
    try:
        df = spark.read.parquet("data/cleaned/smart_parking_clean.parquet")
        # Basic sanity: non-empty
        assert df.count() > 0, "Cleaned dataset is empty."

        # Expected core columns (adjust if needed)
        required_cols = [
            "BayId",
            "DeviceId",
            "StreetMarker",
            "ArrivalTime",
            "DepartureTime",
            "DurationSeconds",
            "DwellMinutes",
            "Hour",
            "DayOfWeek",
            "IsWeekend",
            "Month",
            "DayOfMonth",
            "PartOfDay",
            "rolling_occ_N10",
            "arrivals_N10",
            "InViolation",
            "target_occupied",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"Missing expected columns in cleaned dataset: {missing}"

        # Basic range checks (sample)
        sample = df.select("Hour", "DayOfWeek", "InViolation", "target_occupied").dropna().limit(10000)
        hours_bad = sample.filter((col("Hour") < 0) | (col("Hour") > 23)).count()
        days_bad = sample.filter((col("DayOfWeek") < 1) | (col("DayOfWeek") > 7)).count()
        viol_bad = sample.filter(~col("InViolation").isin(0, 1)).count()
        target_bad = sample.filter(~col("target_occupied").isin(0, 1)).count()

        assert hours_bad == 0, "Found Hour values outside 0–23."
        assert days_bad == 0, "Found DayOfWeek values outside 1–7."
        assert viol_bad == 0, "Found InViolation values not in {0,1}."
        assert target_bad == 0, "Found target_occupied values not in {0,1}."
    finally:
        spark.stop()
