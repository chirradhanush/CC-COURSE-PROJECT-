# spark-mobiles

Week 8: Ingestion + EDA
- Run ingestion:
  spark-submit scripts/01_ingest_clean_sql.py --input data/raw --output output/silver --inr_to_usd 0.012 --brands "Samsung,Apple,Xiaomi,OnePlus"
- Run EDA:
  spark-submit scripts/02_eda.py --input output/silver/clean.csv --outdir reports
