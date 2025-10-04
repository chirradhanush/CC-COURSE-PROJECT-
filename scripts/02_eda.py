# scripts/02_eda.py
import argparse, os
from pathlib import Path
from pyspark.sql import functions as F

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------
# small helpers
# ---------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def to_pd(df, limit=None):
    if limit:
        return df.limit(limit).toPandas()
    return df.toPandas()

def col(df, name, *aliases):
    """Find a column by case-insensitive name or any alias."""
    wanted = {name.lower(), *[a.lower() for a in aliases]}
    for c in df.columns:
        if c.lower() in wanted:
            return F.col(c)
    return None

def get_name(df, name, *aliases):
    wanted = {name.lower(), *[a.lower() for a in aliases]}
    for c in df.columns:
        if c.lower() in wanted:
            return c
    return None


# ---------------------------
# EDA
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to cleaned CSV (file) or directory with CSV part files")
    ap.add_argument("--outdir", default="reports", help="Where to write CSVs & PNGs")
    ap.add_argument("--sample_frac", type=float, default=0.25, help="Fraction for scatter samples")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    img_dir = out_dir / "img"
    csv_dir = out_dir / "tables"
    ensure_dir(img_dir)
    ensure_dir(csv_dir)

    spark = (
        SparkSession.builder.appName("mobiles-eda")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    # Load either a single CSV file or a CSV folder with part files
    read_path = args.input
    if Path(read_path).is_dir():
        df = spark.read.option("header", True).option("inferSchema", True).csv(str(read_path))
    else:
        df = spark.read.option("header", True).option("inferSchema", True).csv(str(read_path))

    def clean_num(colname: str):
        s = F.coalesce(F.col(colname).cast("string"), F.lit(""))
        s = F.regexp_extract(s, r"(\d[\d,\.]*)", 1)
        s = F.regexp_replace(s, ",", "")
        return F.when(F.length(s) > 0, s.cast("double")).otherwise(F.lit(None).cast("double"))
    
    if "ratings" in df.columns and not isinstance(df.schema["ratings"].dataType, DoubleType):
        df = df.withColumn("ratings", clean_num("ratings"))

    if "Rating" in df.columns and "ratings" not in df.columns:
        df = df.withColumn("ratings", clean_num("Rating")).drop("Rating")
    elif "Rating" in df.columns:
        df = df.drop("Rating")

    expected = {"ratings": "double"}
    dtypes = dict(df.dtypes)

    for c, t in expected.items():
        if c not in dtypes:
            raise ValueError(f"Missing column: {c}")
        if dtypes[c] != t:
            raise TypeError(f"Column {c} should be {t}, got {dtypes[c]}")

    def clean_num(col):
        s = F.coalesce(F.col(col).cast("string"), F.lit(""))
        # grab the first number token like 4,278 or 4.5
        s = F.regexp_extract(s, r"(\d[\d,\.]*)", 1)
        s = F.regexp_replace(s, ",", "")
        return F.when(F.length(s) > 0, s.cast("double")).otherwise(F.lit(None).cast("double"))

    # Apply to all possibly-messy numeric fields (only if they exist)
    maybe_numeric = [
        "ratings", "reviews",                # these often contain words
        "price_inr", "price_usd",
        "ram_gb", "storage_gb", "display_inches",
        "primary_camera_mp", "secondary_camera_mp",
        "battery_mah"
    ]
    for c in maybe_numeric:
        if c in df.columns:
            df = df.withColumn(c, clean_num(c))   

    # Map commonly used column names (case-insensitive)
    c_brand   = get_name(df, "brand", "brand_norm")
    c_pname   = get_name(df, "product name", "product", "name")
    c_pricei  = get_name(df, "price_inr", "priceinr")
    c_priceu  = get_name(df, "price_usd", "priceusd")
    c_rating  = get_name(df, "rating", "ratings")
    c_reviews = get_name(df, "reviews", "review_count")
    c_ram     = get_name(df, "ram_gb", "ram (gb)", "ram")
    c_storage = get_name(df, "storage_gb", "storage (gb)", "storage")
    c_display = get_name(df, "display_size_inch", "display size (inch)", "display")
    c_cam1    = get_name(df, "primary_camera_mp", "primary camera (mp)", "primary_camera")
    c_cam2    = get_name(df, "secondary_camera_mp", "secondary camera(mp)", "secondary_camera")
    c_batt    = get_name(df, "battery_mah", "battery")

    # Cast numeric columns defensively (if present)
    numeric_cols = [c_pricei, c_priceu, c_rating, c_reviews, c_ram, c_storage, c_display, c_cam1, c_cam2, c_batt]
    for c in [c for c in numeric_cols if c]:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # ---------- high-level counts ----------
    n_rows = df.count()
    n_cols = len(df.columns)

    # ---------- basic numeric describe ----------
    numerics = [c for c in [c_pricei, c_priceu, c_rating, c_reviews, c_ram, c_storage, c_display, c_cam1, c_cam2, c_batt] if c]
    desc_pd = to_pd(df.select([F.col(c) for c in numerics]).summary("count","mean","stddev","min","25%","50%","75%","max"))
    desc_pd.to_csv(csv_dir / "numeric_summary.csv", index=False)

    # ---------- top brands ----------
    if c_brand:
        top_brand_count = df.groupBy(c_brand).count().orderBy(F.desc("count"))
        to_pd(top_brand_count).to_csv(csv_dir / "top_brands_by_count.csv", index=False)

        if c_pricei:
            top_brand_price = (
                df.groupBy(c_brand)
                  .agg(
                      F.count(F.lit(1)).alias("n"),
                      F.expr(f"percentile_approx({c_pricei}, 0.5)").alias("median_price_inr"),
                      F.avg(F.col(c_pricei)).alias("avg_price_inr")
                  )
                  .orderBy(F.desc("n"))
            )
            to_pd(top_brand_price).to_csv(csv_dir / "top_brands_price_stats.csv", index=False)

    # ---------- price bands ----------
    if c_pricei:
        df = df.withColumn("price_band_inr",
                           F.when(F.col(c_pricei)<10000, "<10k")
                            .when(F.col(c_pricei)<20000, "10k–20k")
                            .when(F.col(c_pricei)<30000, "20k–30k")
                            .when(F.col(c_pricei)<50000, "30k–50k")
                            .otherwise(">=50k"))
        band_counts = df.groupBy("price_band_inr").count().orderBy("price_band_inr")
        to_pd(band_counts).to_csv(csv_dir / "price_band_counts.csv", index=False)

    # ---------- correlation matrix (numerics only) ----------
    corr_pd = None
    if numerics:
        pdf = to_pd(df.select([F.col(c) for c in numerics]))
        # safe: only keep columns that ended up numeric
        for colname in list(pdf.columns):
            pdf[colname] = pd.to_numeric(pdf[colname], errors="coerce")
        corr_pd = pdf.corr(numeric_only=True)
        corr_pd.to_csv(csv_dir / "correlation_matrix.csv")

    # ---------- figures ----------
    # 1) histogram of prices
    if c_pricei:
        pdf = to_pd(df.select(F.col(c_pricei)).where(F.col(c_pricei).isNotNull()))
        if not pdf.empty:
            plt.figure()
            pdf[c_pricei].plot(kind="hist", bins=30, title="Price (INR) distribution")
            plt.xlabel("price_inr")
            plt.tight_layout()
            plt.savefig(img_dir / "hist_price_inr.png", dpi=150)
            plt.close()

    # 2) top 10 brands by count (bar)
    if c_brand:
        pdf = to_pd(
            df.groupBy(c_brand).count().orderBy(F.desc("count")).limit(10)
        )
        if not pdf.empty:
            plt.figure()
            pdf.set_index(c_brand)["count"].plot(kind="bar", title="Top 10 brands by count")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(img_dir / "bar_top_brands.png", dpi=150)
            plt.close()

    # 3) scatter: price vs RAM (sample)
    if c_pricei and c_ram:
        sdf = df.select(F.col(c_pricei).alias("price_inr"), F.col(c_ram).alias("ram_gb")) \
                .where(F.col(c_pricei).isNotNull() & F.col(c_ram).isNotNull()) \
                .sample(False, max(min(args.sample_frac, 1.0), 0.0), seed=42)
        pdf = to_pd(sdf)
        if not pdf.empty:
            plt.figure()
            plt.scatter(pdf["ram_gb"], pdf["price_inr"], s=10, alpha=0.6)
            plt.xlabel("RAM (GB)")
            plt.ylabel("Price (INR)")
            plt.title("Price vs RAM (sample)")
            plt.tight_layout()
            plt.savefig(img_dir / "scatter_price_vs_ram.png", dpi=150)
            plt.close()

    # 4) scatter: price vs display size (sample)
    if c_pricei and c_display:
        sdf = df.select(F.col(c_pricei).alias("price_inr"), F.col(c_display).alias("display_in")) \
                .where(F.col(c_pricei).isNotNull() & F.col(c_display).isNotNull()) \
                .sample(False, max(min(args.sample_frac, 1.0), 0.0), seed=42)
        pdf = to_pd(sdf)
        if not pdf.empty:
            plt.figure()
            plt.scatter(pdf["display_in"], pdf["price_inr"], s=10, alpha=0.6)
            plt.xlabel("Display size (inch)")
            plt.ylabel("Price (INR)")
            plt.title("Price vs Display Size (sample)")
            plt.tight_layout()
            plt.savefig(img_dir / "scatter_price_vs_display.png", dpi=150)
            plt.close()

    # 5) box: price by brand (top 6)
    if c_pricei and c_brand:
        top6 = [r[c_brand] for r in df.groupBy(c_brand).count().orderBy(F.desc("count")).limit(6).collect()]
        pdf = to_pd(
            df.where(F.col(c_brand).isin(top6) & F.col(c_pricei).isNotNull())
              .select(F.col(c_brand).alias("brand"), F.col(c_pricei).alias("price_inr"))
        )
        if not pdf.empty:
            # draw a simple boxplot grouped by brand
            plt.figure()
            groups = [g["price_inr"].dropna().values for _, g in pdf.groupby("brand")]
            labels = list(pdf.groupby("brand").groups.keys())
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.ylabel("Price (INR)")
            plt.title("Price by Brand (top 6)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(img_dir / "box_price_by_brand.png", dpi=150)
            plt.close()

    # ---------- tiny markdown summary ----------
    md = [
        "# EDA Summary",
        f"- Rows: **{n_rows}**  |  Columns: **{n_cols}**",
        f"- Output tables: `{csv_dir}`",
        f"- Output images: `{img_dir}`",
        "",
        "## Notable files",
        "- `tables/numeric_summary.csv`: count/mean/std/min/quantiles for numeric columns",
        "- `tables/top_brands_by_count.csv`: brand frequency",
        "- `tables/top_brands_price_stats.csv`: brand n/median/avg price (INR)",
        "- `tables/price_band_counts.csv`: distribution across price bands",
        "- `tables/correlation_matrix.csv`: correlations among numeric fields",
        "- `img/*.png`: price histogram, top brands bar chart, scatter plots, box plot",
    ]
    (out_dir / "EDA_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    print("=== EDA COMPLETE ===")
    print(f"Rows: {n_rows} | Cols: {n_cols}")
    print(f"Tables -> {csv_dir}")
    print(f"Images -> {img_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
