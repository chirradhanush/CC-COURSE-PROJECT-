
import os
from pyspark.sql import SparkSession


def _create_spark():
    return (
        SparkSession.builder
        .appName("smart_parking_tests_sql")
        .master("local[*]")
        .getOrCreate()
    )


def test_non_trivial_sql_query_runs():
    """
    Validate that we can register the cleaned parquet as a temp view
    and run a non-trivial aggregation + HAVING filter using Spark SQL.
    """
    assert os.path.exists("data/cleaned/smart_parking_clean.parquet"), (
        "Cleaned parquet not found at data/cleaned/smart_parking_clean.parquet"
    )

    spark = _create_spark()
    try:
        df = spark.read.parquet("data/cleaned/smart_parking_clean.parquet")
        df.createOrReplaceTempView("parking")

        sql_query = """
        SELECT
            AreaName,
            COUNT(*) AS events,
            AVG(DwellMinutes) AS avg_dwell
        FROM parking
        GROUP BY AreaName
        HAVING COUNT(*) >= 100
        ORDER BY avg_dwell DESC
        """

        result = spark.sql(sql_query)

        assert result.count() > 0, "SQL query returned no rows; check data or query."
        cols = result.columns
        for c in ["AreaName", "events", "avg_dwell"]:
            assert c in cols, f"Expected column '{c}' missing from SQL result."
    finally:
        spark.stop()
