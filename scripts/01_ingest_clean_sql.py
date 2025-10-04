# scripts/01_ingest_clean_sql.py
import argparse, os, csv, sys
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from py4j.protocol import Py4JJavaError

# ---------- helpers (all SQL expressions, no Python UDFs) ----------
def money_to_double(col):
    s = F.coalesce(F.col(col).cast("string"), F.lit(""))
    s = F.regexp_replace(
        s,
        r"(?i)(₹|\u20B9|rs\.?|inr|â‚¹|Γé╣|├óΓÇÜ┬╣|╬ô├⌐Γòú|Γö£├│╬ô├ç├£Γö¼Γòú|Γò¼├┤Γö£ΓîÉ╬ô├▓├║)",
        ""
    )
    s = F.regexp_replace(s, r"[^0-9,.\s]+", " ")
    token = F.regexp_extract(s, r"(\d[\d,]*(?:\.\d{1,2})?)", 1)
    token = F.regexp_replace(token, ",", "")
    return F.when(F.length(token) > 0, token.cast("double")).otherwise(F.lit(None).cast("double"))

def first_word(col):
    s = F.coalesce(F.col(col).cast("string"), F.lit(""))
    return F.lower(F.regexp_extract(s, r"^\s*([^\s\-\_/|]+)", 1))

def only_digits_to_long(col, word_to_strip=None):
    s = F.coalesce(F.col(col).cast("string"), F.lit(""))
    if word_to_strip:
        s = F.regexp_replace(s, rf"(?i)\b{word_to_strip}\b", "")
    s = F.regexp_replace(s, r"[^\d]", "")
    return F.when(F.length(s) > 0, s.cast("long")).otherwise(F.lit(None).cast("long"))

def parse_cameras_from_text(col):
    s = F.coalesce(F.col(col).cast("string"), F.lit(""))
    first  = F.regexp_extract(s, r"(?i)(\d+)\s*mp", 1)
    second = F.regexp_extract(s, r"(?i)\d+\s*mp\D+(\d+)\s*mp", 1)
    first  = F.when(F.length(first)  > 0, first ).otherwise(F.lit("nil"))
    second = F.when(F.length(second) > 0, second).otherwise(F.lit("nil"))
    return first, second

def extract_battery_mAh(col):
    s = F.coalesce(F.col(col).cast("string"), F.lit(""))
    token = F.regexp_extract(s, r"(?i)\b(\d{3,5})\s*mAh\b", 1)
    return F.when(F.length(token) > 0, token).otherwise(F.lit("nil"))

def to_int_from_text(col):
    s = F.coalesce(F.col(col).cast("string"), F.lit(""))
    s = F.when(F.lower(F.trim(s)) == "nil", F.lit("0")).otherwise(s)
    tok = F.regexp_extract(s, r"(\d+)", 1)
    tok = F.when(F.length(tok) > 0, tok).otherwise(F.lit("0"))
    return tok.cast("int")

def to_double_from_text(col):
    s = F.coalesce(F.col(col).cast("string"), F.lit(""))
    s = F.when(F.lower(F.trim(s)) == "nil", F.lit("0")).otherwise(s)
    tok = F.regexp_extract(s, r"(\d+(?:\.\d+)?)", 1)
    tok = F.when(F.length(tok) > 0, tok).otherwise(F.lit("0"))
    return tok.cast("double")

def write_csv_fallback(df, out_file):
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = df.columns
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in df.toLocalIterator():
            w.writerow([row[c] if row[c] is not None else "" for c in cols])
    print(f"[fallback] Wrote CSV -> {out_file}")

# ---------- main ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--inr_to_usd", type=float, default=0.012)
    p.add_argument("--brands", default="ALL", help="Comma list. Matches first word of Product Name (lowercase).")
    args = p.parse_args()

    spark = (
        SparkSession.builder.appName("mobiles-cleaning-sql")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.hadoop.io.native.lib.available", "false")
        .getOrCreate()
    )

    # Read CSV as all-strings, then trim
    df = spark.read.csv(args.input, header=True, inferSchema=False, multiLine=False, escape='"')
    for c in df.columns:
        df = df.withColumn(c, F.trim(F.col(c).cast("string")))

    cols = set(df.columns)

    # Flexible column detection
    disc_col    = next((c for c in ["Discount price","Discount Price","discount_price","discountprice","Price"] if c in cols), None)
    act_col     = next((c for c in ["Actual price","Actual Price","actual_price","actualprice","MRP"] if c in cols), None)
    product_col = next((c for c in ["Product name","Product Name","product_name","Title"] if c in cols), None)
    rating_col  = next((c for c in ["Rating","Ratings","rating","ratings"] if c in cols), None)
    reviews_col = next((c for c in ["Reviews","reviews"] if c in cols), None)
    camera_col  = next((c for c in ["Camera","camera"] if c in cols), None)
    descr_col   = next((c for c in ["Description","description"] if c in cols), None)
    link_col    = next((c for c in ["Link","link","URL","Url"] if c in cols), None)

    # ---- Price cleanup ----
    price_disc_inr = money_to_double(disc_col) if disc_col else F.lit(None).cast("double")
    price_act_inr  = money_to_double(act_col)  if act_col  else F.lit(None).cast("double")
    price_inr = F.coalesce(price_disc_inr, price_act_inr)
    rate = F.lit(float(args.inr_to_usd))

    clean = df.withColumn("price_inr", price_inr) \
              .withColumn("price_usd", F.round(price_inr * rate, 2))

    # ---- brand from first word of Product Name ----
    if product_col:
        clean = clean.withColumn("brand", first_word(product_col))
    else:
        clean = clean.withColumn("brand", F.lit("unknown"))

    # ---- Ratings / Reviews ----
    # 1) Create ONE numeric `ratings` column (double) from any rating-like column.
    if rating_col:
        s = F.coalesce(F.col(rating_col).cast("string"), F.lit(""))
        s = F.regexp_extract(s, r"(\d[\d,\.]*)", 1)     # "4,278 Ratings" -> "4,278" ; "4.5 out of 5" -> "4.5"
        s = F.regexp_replace(s, ",", "")
        clean = clean.withColumn("ratings", F.when(F.length(s) > 0, s.cast("double")).otherwise(F.lit(None).cast("double")))
        # 2) Drop any original stringy variant if its name differs from "ratings"
        if rating_col != "ratings":
            clean = clean.drop(rating_col)

    # Reviews (count), keep as integer-like if present
    if reviews_col:
        clean = clean.withColumn("reviews", only_digits_to_long(reviews_col, "reviews"))

    # ---- Camera split ----
    if camera_col:
        p_cam, s_cam = parse_cameras_from_text(camera_col)
        clean = clean.withColumn("primary camera (MP)", p_cam) \
                     .withColumn("secondary camera (MP)", s_cam)

    # ---- Battery from description ----
    if descr_col:
        clean = clean.withColumn("battery", extract_battery_mAh(descr_col))

    # ---- Filter invalid/zero prices ----
    clean = clean.filter(F.col("price_inr").isNotNull() & (F.col("price_inr") > 0))

    # ---- Optional brand filter ----
    if args.brands.strip().upper() != "ALL":
        wanted = [b.strip().lower() for b in args.brands.split(",") if b.strip()]
        clean = clean.filter(F.col("brand").isin(wanted))

    # ---- Cast selected columns to numeric; "nil" -> 0 ----
    if "RAM (GB)" in clean.columns:
        clean = clean.withColumn("RAM (GB)", to_int_from_text("RAM (GB)"))
    if "Storage (GB)" in clean.columns:
        clean = clean.withColumn("Storage (GB)", to_int_from_text("Storage (GB)"))
    if "Display Size (inch)" in clean.columns:
        clean = clean.withColumn("Display Size (inch)", to_double_from_text("Display Size (inch)"))
    if "primary camera (MP)" in clean.columns:
        clean = clean.withColumn("primary camera (MP)", to_int_from_text("primary camera (MP)"))
    if "secondary camera (MP)" in clean.columns:
        clean = clean.withColumn("secondary camera (MP)", to_int_from_text("secondary camera (MP)"))
    if "battery" in clean.columns:
        clean = clean.withColumn("battery", to_int_from_text("battery"))

    # ---- Drop obvious duplicates ----
    if link_col:
        clean = clean.dropDuplicates([link_col])
    else:
        keys = []
        if product_col:
            keys.append(product_col)
        keys.append("price_inr")
        clean = clean.dropDuplicates(keys)

    # ---- Write CSV (Spark first; Windows native error -> Python fallback) ----
    csv_dir = str(Path(args.output) / "csv")
    try:
        (clean.coalesce(1)
              .write.mode("overwrite")
              .option("header", True)
              .csv(csv_dir))
        print(f"[spark] Wrote CSV -> {csv_dir}")
    except Py4JJavaError as e:
        emsg = (str(getattr(e, "java_exception", "")) or "") + " " + str(e)
        if ("UnsatisfiedLinkError" in emsg) or ("NativeIO$Windows.access0" in emsg):
            out_file = Path(args.output) / "clean.csv"
            print("[warn] Spark CSV commit hit Windows native IO bug; falling back to single CSV:", out_file)
            write_csv_fallback(clean, str(out_file))
        else:
            raise
    except Exception as e:
        emsg = str(e)
        if ("UnsatisfiedLinkError" in emsg) or ("NativeIO$Windows.access0" in emsg):
            out_file = Path(args.output) / "clean.csv"
            print("[warn] Spark CSV commit hit Windows native IO bug; falling back to single CSV:", out_file)
            write_csv_fallback(clean, str(out_file))
        else:
            raise

    print("=== CLEANING DONE ===")
    print(f"Rows: {clean.count()}")
    show_cols = [c for c in ["brand","price_inr","price_usd",
                             "RAM (GB)","Storage (GB)","Display Size (inch)",
                             "primary camera (MP)","secondary camera (MP)","battery",
                             "ratings","reviews"]
                 if c in clean.columns]
    if show_cols:
        clean.select(*show_cols).show(10, truncate=False)

    spark.stop()
