import os
import glob
import json
import random
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    TimestampType,
)

def load_seed_events(local_folder):
    """
    Read all event_*.json files from local_folder using plain Python,
    and return a list of dicts:
      {
        "device_id": "...",
        "street_marker": "...",
        "event_time_ts": datetime(...),
        "occupied": 1
      }
    We handle timestamps like '2019-09-01 08:06:25+00:00'
    by stripping the timezone part for local demo.
    """
    events = []
    for path in glob.glob(os.path.join(local_folder, "event_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw_time = raw["event_time"]  # e.g. '2019-09-01 08:06:25+00:00'

        # Drop the timezone part like '+00:00' if present
        # Split on '+' and take the first chunk
        # ('2019-09-01 08:06:25', '00:00')
        if "+" in raw_time:
            raw_time = raw_time.split("+")[0]

        # Now parse 'YYYY-MM-DD HH:MM:SS'
        try:
            ts = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # If it still doesn't match, skip this row
            continue

        events.append(
            {
                "device_id": raw["device_id"],
                "street_marker": raw["street_marker"],
                "event_time_ts": ts,
                "occupied": int(raw["occupied"]),
            }
        )

    return events


def main():
    spark = (
        SparkSession.builder
        .appName("smart_parking_stream_demo")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Load seed events (Python-side safe read, no Hadoop)
    seed_folder = "streaming/incoming"
    seed_events = load_seed_events(seed_folder)

    if not seed_events:
        print("[stream] ERROR: Still found 0 usable seed events in streaming/incoming.")
        print("[stream] Check that event_time has correct format or that there are JSON files.")
        spark.stop()
        return

    # 2. Create static Spark DF from seed
    seed_schema = StructType([
        StructField("device_id", StringType(), True),
        StructField("street_marker", StringType(), True),
        StructField("event_time_ts", TimestampType(), True),
        StructField("occupied", IntegerType(), True),
    ])

    hist_df = spark.createDataFrame(seed_events, schema=seed_schema).cache()

    print("[stream] Loaded historical seed events:")
    hist_df.show(truncate=False)

    # Create broadcast list for UDF sampling
    events_list = [
        row.asDict()
        for row in hist_df.collect()
    ]
    bc_events = spark.sparkContext.broadcast(events_list)

    # 3. Structured Streaming heartbeat
    rate_stream = (
        spark.readStream
        .format("rate")
        .option("rowsPerSecond", 1)
        .load()
    )

    # 4. UDF to "emit" one random historical event per tick
    def choose_event(_val):
        ev = random.choice(bc_events.value)
        return (
            ev["device_id"],
            ev["street_marker"],
            ev["event_time_ts"].strftime("%Y-%m-%d %H:%M:%S"),
            int(ev["occupied"]),
        )

    choose_schema = StructType([
        StructField("device_id", StringType(), True),
        StructField("street_marker", StringType(), True),
        StructField("event_time_str", StringType(), True),
        StructField("occupied", IntegerType(), True),
    ])

    choose_event_udf = F.udf(choose_event, choose_schema)

    simulated_stream = (
        rate_stream
        .withColumn("picked", choose_event_udf(F.col("value")))
        .select(
            F.col("picked.device_id").alias("device_id"),
            F.col("picked.street_marker").alias("street_marker"),
            F.to_timestamp("picked.event_time_str").alias("event_time_ts"),
            F.col("picked.occupied").alias("occupied"),
        )
    )

    # 5. Rolling 5-minute occupancy by street_marker
    occupancy_windowed = (
        simulated_stream
        .withWatermark("event_time_ts", "10 minutes")
        .groupBy(
            F.window("event_time_ts", "5 minutes", "1 minute"),
            F.col("street_marker"),
        )
        .agg(
            F.avg("occupied").alias("occupancy_rate"),
            F.count("*").alias("event_count"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "street_marker",
            F.round("occupancy_rate", 3).alias("occupancy_rate"),
            "event_count",
        )
        .orderBy("window_start", "street_marker")
    )

    query = (
        occupancy_windowed
        .writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", "false")
        .start()
    )

    print("[stream] Streaming query started (Windows-safe).")
    print("[stream] Every second we emit a pseudo-live parking event sampled from your real seed JSONs in streaming/incoming.")
    print("[stream] Ctrl+C to stop.")

    query.awaitTermination()


if __name__ == "__main__":
    main()
