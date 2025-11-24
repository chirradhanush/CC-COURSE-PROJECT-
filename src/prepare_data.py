#!/usr/bin/env python

from pathlib import Path
from pyspark.sql import SparkSession


def get_paths():
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "cleaned_csv": root / "data" / "cleaned" / "smart_parking_clean.csv",
        "cleaned_parquet": root / "data" / "cleaned" / "smart_parking_clean.parquet",
        "train_ready": root / "data" / "train_ready",
    }


def create_spark():
    spark = (
        SparkSession.builder
        .appName("smart_parking_prepare_data_passthrough")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    paths = get_paths()
    cleaned_csv = paths["cleaned_csv"]

    print(f"[prepare_data] Expecting cleaned CSV at: {cleaned_csv}")

    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"{cleaned_csv} not found. "
            "Please generate smart_parking_clean.csv with your ingestion/EDA notebook."
        )

    spark = create_spark()

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(cleaned_csv))
    )

    print(f"[prepare_data] Loaded cleaned CSV rows={df.count()}, cols={len(df.columns)}")

    # Write as parquet for potential convenience (not required by ML)
    paths["cleaned_parquet"].parent.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_data] Writing cleaned parquet to: {paths['cleaned_parquet']}")
    (
        df.write
        .mode("overwrite")
        .parquet(str(paths["cleaned_parquet"]))
    )

    # Also write to train_ready as-is (ML pipeline will handle feature selection)
    paths["train_ready"].mkdir(parents=True, exist_ok=True)
    print(f"[prepare_data] Writing train_ready parquet (passthrough) to: {paths['train_ready']}")
    (
        df.write
        .mode("overwrite")
        .parquet(str(paths["train_ready"]))
    )

    print("[prepare_data] Done (no extra cleaning applied; using precomputed smart_parking_clean.csv).")
    spark.stop()


if __name__ == "__main__":
    main()
