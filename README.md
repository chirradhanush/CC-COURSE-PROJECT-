# Spark Mobiles – Ingestion & EDA

This project ingests a raw mobiles CSV, cleans and normalizes key fields (prices, ratings/reviews, cameras, battery, RAM/Storage), and writes a single canonical dataset. A separate EDA step reads that cleaned file and produces quick tables/plots for exploration.

**Ingestion** (`01_ingest_clean_sql.py`): robust text→numeric cleanup using pure Spark SQL functions (no UDFs), tolerant to messy labels (e.g., ₹, INR, commas), and to column name variations. It creates a single file `output/silver/clean.csv` and removes the redundant string `Rating` column; it retains a numeric `ratings` column.

**EDA** (`02_eda.py`): reads `output/silver/clean.csv`, validates numeric types, summarizes distributions, builds price bands, computes correlations, and emits CSV tables and PNG plots into a timestamped folder under `reports/`.

---

## Project Layout

```
.
├── data/
│   └── raw/
│       └── Mobiles_Dataset.csv        # your raw input file (example)
├── output/
│   └── silver/
│       └── clean.csv                  # written by ingestion
├── reports/
│   └── YYYYMMDD_HHMMSS/               # created by EDA (tables/ & img/)
├── scripts/
│   ├── 01_ingest_clean_sql.py
│   └── 02_eda.py
└── README.md
```

---

## Requirements

- **Python 3.8+**
- **Apache Spark** (tested with Spark 4.x, PySpark)
- **Java 11+** (you're on 21, that's fine)
- **Python libs for EDA**: `matplotlib`, `pandas`

**Windows note**: Spark on Windows may hit the Hadoop native-IO bug when writing CSV. The ingestion script detects this and falls back to a pure-Python writer to `output/silver/clean.csv`. No extra action needed.

---

## Quick Start

### 1. Ingest & clean raw data → write `clean.csv`

```powershell
# PowerShell
spark-submit scripts/01_ingest_clean_sql.py `
  --input data/raw/Mobiles_Dataset.csv `
  --output output/silver `
  --inr_to_usd 0.012 `
  --brands ALL
```

### 2. Run EDA (creates report folder)

```powershell
spark-submit scripts/02_eda.py `
  --input output/silver/clean.csv `
  --outdir reports
```

**You'll get:**

- `reports/tables/*.csv`
- `reports/img/*.png`
- `reports/EDA_SUMMARY.md`

---

## Ingestion Script Details

**Path**: `scripts/01_ingest_clean_sql.py`

### What it does

1. Reads the raw CSV (all columns as strings), trims whitespace.
2. **Detects common column name variants**:
   - **Prices**: `Discount price` / `Actual price` / `Price` / `MRP`
   - `Product name`, `Rating`/`Ratings`, `Reviews`, `Camera`, `Description`, `Link`
3. **Money cleanup**: strips currency tokens (₹, INR, Rs, odd encodings), commas, etc., and converts to `double` (`price_inr`). Also computes `price_usd = price_inr * --inr_to_usd`.
4. **Brand**: first word of product name, lowercased.
5. **Ratings/Reviews**: converts to integers by stripping words; drops the original string `Rating` column and keeps only numeric `ratings`.
6. **Camera/Battery/RAM/Storage/Display**: parses common patterns into numeric columns.
7. **Duplicates**: drops exact dupes (prefers URL `Link` if present, else `Product name` + `price_inr`).
8. **Brand filter**: `--brands ALL` keeps all brands; else pass a comma list (`"Samsung,Apple"`).
9. **Output**: writes one file `output/silver/clean.csv`.
   - If Spark CSV write fails (Windows native-IO), it automatically falls back to a Python writer.

### CLI

```
--input         Path to raw CSV (file or folder)
--output        Folder where clean.csv will be written (e.g., output/silver)
--inr_to_usd    FX rate to compute price_usd (default 0.012)
--brands        Comma list or "ALL" (default "ALL")
```

### Examples

```powershell
# All brands
spark-submit scripts/01_ingest_clean_sql.py --input data/raw/Mobiles_Dataset.csv --output output/silver --brands ALL

# Filter to a few brands
spark-submit scripts/01_ingest_clean_sql.py --input data/raw/Mobiles_Dataset.csv --output output/silver --brands "Samsung,Apple,Xiaomi"
```

---

## EDA Script Details

**Path**: `scripts/02_eda.py`

### What it does

1. Reads only `output/silver/clean.csv` (no directory check).
2. Ensures `ratings` is numeric (`double`) and removes any stray `Rating` string column if present.
3. Validates key numeric columns and casts them defensively.
4. **Emits CSV tables**:
   - `numeric_summary.csv` (count, mean, std, min/quantiles/max)
   - `top_brands_by_count.csv`
   - `top_brands_price_stats.csv` (n, median, average price)
   - `price_band_counts.csv` (<10k, 10k–20k, …)
   - `correlation_matrix.csv`
5. **Emits PNG figures**:
   - Price histogram, Top-10 brand counts (bar),
   - Scatter: price vs RAM, price vs display (sampled),
   - Box: price by top-6 brands
6. Writes a short `EDA_SUMMARY.md` in the timestamped run folder.

### CLI

```
--input        Path to the cleaned file (use output/silver/clean.csv)
--outdir       Reports root (default: reports)
--sample_frac  Fraction for scatter samples (default: 0.25)
```

### Example

```powershell
spark-submit scripts/02_eda.py --input output/silver/clean.csv --outdir reports
```


---

## Why the `ratings` Fix Matters

Many raw dumps contain both a text column (`Rating` like "4.2 out of 5") and counts (`Ratings` like "2,341 ratings").

- The **ingestion** step drops the text `Rating` and normalizes a single numeric `ratings` column—so downstream code is simpler and consistent.
- The **EDA** step assumes only `ratings` (numeric). If a rogue `Rating` appears, EDA drops it again to stay resilient.

---

## Re-running & Housekeeping

- It's safe to **re-run ingestion**; it overwrites `output/silver/clean.csv`.
- **EDA** overwrites based on selected brands and  past runs; it overwrites `reports/`.


---

## Troubleshooting

### Windows `UnsatisfiedLinkError` (Hadoop native-IO) when writing CSV

- The script detects this and falls back to a Python writer automatically.
- Expect a log line like: `[warn] Spark CSV write had an issue; falling back to Python writer.`

### Column name mismatches

- Ingestion detects multiple common variants; if your raw headers are unusual, adjust the lists near the top of `01_ingest_clean_sql.py` where columns are discovered.

### Brand filtering confusion

- Use `--brands ALL` to keep everything, or pass a quoted comma list to restrict.

---

