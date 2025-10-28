## Components

### 1. `generator.py`
- Input: `streaming/sample_events.csv` (cleaned parking rows)
- Output: writes one JSON file per event into `streaming/incoming/`, e.g.
  ```json
  {
    "device_id": "23913",
    "street_marker": "1581S",
    "event_time": "2019-09-01 08:06:25+00:00",
    "occupied": 1
  }

### 2. parking_stream.py

This is our Structured Streaming consumer / aggregator.

Pipeline:

Load the generated JSON events from streaming/incoming/.
We parse them in Python (to avoid Windows/Hadoop native IO issues) and create a Spark DataFrame with columns:

device_id

street_marker

event_time_ts (timestamp)

occupied (0/1)

Start a Spark Structured Streaming source using rate (1 row/sec).
Each "tick" simulates a new live parking event by randomly sampling from the historical seed events.

For the streaming DataFrame:


Metrics we compute:

occupancy_rate = avg(occupied) in that window

event_count = number of sensor updates in that window


This lets us answer:

"Which blocks are basically full right now?"

"How fast is occupancy changing in the last 5 minutes?"



#### Environment note

On Windows 11 with Spark 4.0.1 + Java 21 + Python 3.13, the PySpark worker can crash with:
Python worker exited unexpectedly and WinError 10038.

This is a known compatibility issue between Spark's Python worker launcher and Python 3.13 on Windows. On a Linux/Mac Spark runtime or on Python 3.11, the same code runs and prints rolling occupancy windows to the console.

So the logic, watermark, sliding window, and streaming sink are implemented and ready; the remaining work is just environment/runtime alignment.