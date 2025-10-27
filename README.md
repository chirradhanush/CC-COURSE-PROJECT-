## Smart Parking Occupancy - Ingestion & Initial EDA (New)

### What this sub-project is
We are building a Smart Parking Occupancy Prediction pipeline. The goal is to analyze and predict when parking locations are occupied by using historical parking activity data.

### What was added in this update
- `notebooks/ingestion_code.ipynb`: Notebook that loads the raw parking dataset, cleans columns (timestamps, occupancy status), and engineers features such as:
  - hour_of_day
  - day_of_week
  - weekend_flag
  - occupancy (0/1)
- `data/raw/smart_parking_15k.csv`: Prototype dataset (~15k rows).
- `data/cleaned/smart_parking_clean.csv`: Cleaned/normalized data after ingestion.
- `data/ml_ready/parking_ml_ready.csv`: Feature-ready table that will be used later for ML (Week 12 milestone).

### Why this matters
This completes the ingestion stage. We can now:
1. Run basic EDA (row counts, busiest hours, etc.).
2. Write Spark SQL queries for usage patterns.
3. Feed this data into streaming and ML components in future milestones.

### Next steps
- Add EDA summary (hourly occupancy trends, busiest blocks/zones).
- Add Spark SQL queries + tests (Milestone: SQL / Week 10).
- Add streaming job for live occupancy monitoring (Week 11).
- Train ML model for occupancy prediction (Week 12).
