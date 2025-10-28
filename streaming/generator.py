import csv
import json
import time
from pathlib import Path

# Paths (relative to repo root OR if you run from inside 'streaming/')
SAMPLE_FILE = Path(__file__).parent / "sample_events.csv"
OUT_DIR = Path(__file__).parent / "incoming"

def read_rows():
    """
    Read the CSV rows in a Windows-safe way.
    We'll try utf-8-sig first (Excel w/ BOM),
    and fallback to cp1252 if needed.
    Returns a list[dict] of rows.
    """
    encodings_to_try = ["utf-8-sig", "cp1252"]

    last_error = None
    for enc in encodings_to_try:
        try:
            with SAMPLE_FILE.open("r", newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                return rows
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not read {SAMPLE_FILE} with known encodings. Last error: {last_error}"
    )


def main(sleep_seconds=1.0):
    # make sure output dir exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_rows()

    if not rows:
        print("[generator] sample_events.csv had 0 data rows. Nothing to send.")
        return

    for i, row in enumerate(rows):
        # IMPORTANT:
        # match your actual CSV header names EXACTLY.
        # From your screenshot, they looked like:
        # DeviceId, StreetMarker, _event_time, _occupied

        # If your file actually uses different headers,
        # change them below.
        try:
            event = {
                "device_id": row["DeviceId"],
                "street_marker": row["StreetMarker"],
                "event_time": row["_event_time"],
                "occupied": int(row["_occupied"]),
            }
        except KeyError as e:
            print("[generator] ERROR: Missing expected column in CSV:", e)
            print("[generator] Row content was:", row)
            print("[generator] Tip: Check header names in streaming/sample_events.csv")
            return

        out_path = OUT_DIR / f"event_{i}_{int(time.time())}.json"
        with out_path.open("w", encoding="utf-8") as out_f:
            json.dump(event, out_f)

        print(f"[generator] wrote {out_path.name} -> {event}")

        # small delay to simulate live feed
        time.sleep(sleep_seconds)

    print("[generator] done sending all events.")


if __name__ == "__main__":
    # You can speed this up for testing by setting sleep_seconds=0.1
    main(sleep_seconds=1.0)
