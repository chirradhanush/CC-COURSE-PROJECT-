
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def _create_spark():
    return (
        SparkSession.builder
        .appName("smart_parking_tests_streaming")
        .master("local[*]")
        .getOrCreate()
    )


def test_rate_stream_microbatch():
    """
    Use Spark Structured Streaming with the built-in 'rate' source
    to validate that we can start a streaming query, process at least one
    micro-batch, and materialize results into an in-memory table.
    """
    spark = _create_spark()
    try:
        rate_df = (
            spark.readStream
            .format("rate")
            .option("rowsPerSecond", 5)
            .load()
        )

        transformed = rate_df.withColumn("double_value", col("value") * 2)

        query = (
            transformed.writeStream
            .format("memory")
            .queryName("rate_test_table")
            .outputMode("append")
            .start()
        )

        query.processAllAvailable()

        result = spark.sql("SELECT COUNT(*) AS cnt FROM rate_test_table")
        row = result.collect()[0]
        assert row["cnt"] > 0, "Streaming query did not produce any rows."

        query.stop()
    finally:
        spark.stop()
