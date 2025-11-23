#!/usr/bin/env python

"""
prepare_data.py
---------------
Ingestion + cleaning + feature engineering for the Smart Parking dataset.

INPUT:
    data/raw/smart_parking_15k.csv

OUTPUTS:
    data/cleaned/smart_parking_clean.parquet
    data/ml_ready/parking_ml_ready.parquet
    data/train_ready/ (directory of parquet files)
"""

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_paths():
    root = Path(__file__).resolve().parents[1]

    return {
        "root": root,
        "raw_csv": root / "data" / "raw" / "smart_parking_15k.csv",
        "cleaned_out": root / "data" / "cleaned" / "smart_parking_clean.parquet",
        "ml_ready_out": root / "data" / "ml_ready" / "parking_ml_ready.parquet",
        "train_ready_out": root / "data" / "train_ready",
    }


def create_spark():
    spark = (
        SparkSession.builder
        .appName("prepare_data_smart_parking")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    paths = get_paths()
    raw_csv = paths["raw_csv"]

    print(f"[prepare_data] Raw CSV: {raw_csv}")

    spark = create_spark()

    # ===========================
    # 1. READ RAW DATA
    # ===========================
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(raw_csv))
    )

    print(f"[prepare_data] Loaded rows = {df.count()}, columns = {len(df.columns)}")

    # ==========================================================
    # 2. CLEAN & NORMALIZE STRING COLUMNS
    # ==========================================================

    df = (
        df.withColumn("device_id", F.upper(F.trim(F.col("DeviceId").cast("string"))))
          .withColumn("street_marker", F.upper(F.trim(F.col("StreetMarker").cast("string"))))
          .withColumn("status_raw", F.upper(F.trim(F.col("Status").cast("string"))))
          .withColumn("sign_plate_id", F.upper(F.trim(F.col("SignPlateID").cast("string"))))
          .withColumn("street_name", F.upper(F.trim(F.col("StreetName").cast("string"))))
    )

    # ==========================================================
    # 3. PARSE TIMESTAMPS
    # ==========================================================
    # Convert ArrivalTime / DepartureTime to timestamp
    def to_ts(col):
        return F.coalesce(
            F.to_timestamp(col),
            F.to_timestamp(col, "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp(col, "dd/MM/yyyy HH:mm"),
            F.to_timestamp(col, "MM/dd/yyyy HH:mm"),
        )

    df = df.withColumn("arrival_ts", to_ts(F.col("ArrivalTime").cast("string")))
    df = df.withColumn("departure_ts", to_ts(F.col("DepartureTime").cast("string")))

    # Most consistent event_time is ArrivalTime
    df = df.withColumn("event_time", F.col("arrival_ts"))

    # ==========================================================
    # 4. DURATION IN MINUTES
    # ==========================================================
    df = df.withColumn("duration_min", F.col("DurationSeconds") / 60.0)

    # Fix negative or absurd durations
    df = df.withColumn("duration_min",
                       F.when(F.col("duration_min") < 0, None)
                       .when(F.col("duration_min") > 1440, None)  # > 24 hours
                       .otherwise(F.col("duration_min"))
                       )

    # ==========================================================
    # 5. BINARY LABEL: target_occupied
    # ==========================================================
    # Your Status shows variants like:
    #   "PRESENT", "UNOCCUPIED", "FREE", "OCCUPIED", etc.
    df = df.withColumn(
        "target_occupied",
        F.when(F.col("status_raw").isin("OCCUPIED", "PRESENT", "1"), 1).otherwise(0)
    )

    # ==========================================================
    # 6. TIME FEATURES
    # ==========================================================
    df = (
        df.withColumn("hour", F.hour("event_time"))
          .withColumn("dow", F.dayofweek("event_time"))  # 1=Sun, 7=Sat
          .withColumn("is_weekend", F.when(F.col("dow").isin(1, 7), 1).otherwise(0))
    )

    # ==========================================================
    # 7. DROP ROWS WITHOUT EVENT_TIME OR LABEL
    # ==========================================================
    df_clean = df.filter(
        F.col("event_time").isNotNull() &
        F.col("target_occupied").isNotNull()
    )

    print(f"[prepare_data] Cleaned rows = {df_clean.count()}")

    # ==========================================================
    # 8. WRITE CLEANED DATASET
    # ==========================================================
    paths["cleaned_out"].parent.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_data] Writing cleaned → {paths['cleaned_out']}")
    (
        df_clean.write
        .mode("overwrite")
        .parquet(str(paths["cleaned_out"]))
    )

    # ==========================================================
    # 9. ML-READY SUBSET
    # ==========================================================
    ml_cols = [
        "device_id",
        "street_marker",
        "event_time",
        "duration_min",
        "hour",
        "dow",
        "is_weekend",
        "target_occupied",
        "InViolation",
        "DwellMinutes",
        "PartOfDay",
        "Month",
        "DayOfMonth"
    ]

    ml_cols = [c for c in ml_cols if c in df_clean.columns]

    ml_df = df_clean.select(*ml_cols)

    paths["ml_ready_out"].parent.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_data] Writing ML-ready → {paths['ml_ready_out']}")
    (
        ml_df.write
        .mode("overwrite")
        .parquet(str(paths["ml_ready_out"]))
    )

    # ==========================================================
    # 10. TRAIN_READY (same as ML-ready)
    # ==========================================================
    print(f"[prepare_data] Writing train_ready → {paths['train_ready_out']}")
    paths["train_ready_out"].mkdir(parents=True, exist_ok=True)
    (
        ml_df.write
        .mode("overwrite")
        .parquet(str(paths["train_ready_out"]))
    )

    print("[prepare_data] ✔ DONE")
    spark.stop()


if __name__ == "__main__":
    main()
