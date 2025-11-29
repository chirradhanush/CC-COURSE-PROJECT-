
# Limitations

This document outlines the key limitations of the **Smart Parking Occupancy Prediction & Analytics Platform** in terms of data quality, modeling constraints, system assumptions, and real-world deployment considerations. These limitations are important for evaluation and highlight future areas of improvement.

---

# 1. Dataset Limitations

## 1.1 Imbalanced Violation Labels  
The dataset contains **only ~3–4% violation events**, creating a significant class imbalance.  
Effects:
- Harder for models to learn violation behavior.
- May overpredict the majority class (compliant).

Mitigations attempted:
- Feature engineering on dwell times.
- Using metrics such as F1 and AUC, not just accuracy.

---

## 1.2 Sensor Errors and Anomalies  
IoT parking sensors occasionally generate:
- Missing timestamps  
- Duplicate records  
- Sudden long-duration readings (e.g., >8–12 hours)  
- Incorrect occupancy flips  

These were corrected via:
- Winsorization  
- Duration thresholds  
- Dropping corrupted rows  

But some noise remains.

---

## 1.3 Limited Contextual Data  
The dataset lacks:
- Weather conditions  
- Traffic flow  
- Nearby events (concerts, sports)  
- Public holiday indicators  
- Real-time pricing or enforcement data  

These could significantly improve prediction accuracy.

---

# 2. Feature Engineering Limitations

## 2.1 Rolling Window Requires Correct Sort Order  
Rolling features (e.g., `rolling_occ_N10`) require strict chronological ordering per bay.  
Any disorder leads to:
- Incorrect rolling averages  
- Distorted behavioral patterns  

Although carefully handled, it is still sensitive to scrambled timestamps.

---

## 2.2 Lumping Reduces Specificity  
We lumped rare categories into `"Other"` for:
- `street_marker_lumped`
- `device_id_lumped`

While it prevents overfitting, it:
- Loses granular geographic detail  
- May remove signals for very small but important regions  

---

## 2.3 PartOfDay Is Coarse  
The `PartOfDay` feature is manually bucketed into:
- Morning
- Afternoon
- Evening
- Night

However:
- It may not capture nuanced patterns such as peak-hour bursts.
- “Morning” spans a wide range (e.g., 6 AM–11 AM).

---

# 3. Model Limitations

## 3.1 Limited Hyperparameter Search  
Spark ML has known constraints with deep grid searches.  
Our tuning used:
- Only 4 combinations  
- Single estimator (Random Forest)

This produces a strong baseline but not the best possible model.

---

## 3.2 No Advanced Models (due to Spark constraints)  
We did not use: 
- CatBoost  
- LightGBM  
- Neural Networks  

Reason:
- Spark MLlib integration limitations  
- Memory constraints in Colab/WSL   

These models may outperform Random Forest on this dataset.

---

## 3.3 Batch-Only Training  
The model is trained on historical batch data.  
It cannot:
- Learn continuously  
- Adapt to new events  
- Incorporate real-time corrections  

Online learning is not implemented.

---

# 4. System Limitations

## 4.1 No Full Streaming Integration  
While the system is designed to be streaming-ready, the implementation is:
- Not connected to Kafka / real IoT feeds  
- Only validated through structural streaming tests  
- Not deployed in a real-time environment  

The dashboard remains batch-driven.

---

## 4.2 Single-Machine Execution  
The project runs on:
- Local machine  
- Google Colab  
- Single Spark session  

It has not been tested on a real Spark cluster:
- No YARN  
- No Kubernetes  
- No multi-node scenario  

Thus large scalability remains unverified.

---

## 4.3 Dashboard Reads Static CSVs  
The Streamlit dashboard:
- Uses CSV outputs from the ML pipeline  
- Does not auto-refresh for new events  
- Cannot visualize real-time trends (yet)

---

# 5. SQL Limitations

## 5.1 Limited Query Complexity  
SQL queries are focused on:
- Aggregations  
- Grouping  
- Summary statistics  

They do not include:
- Window functions  
- Joins across multiple datasets  
- Geo-based clustering  
- Time-series analytics  

---

## 5.2 No ETL Integration with External Databases  
Queries run only on Spark DataFrames.  
There is no integration with:
- Snowflake  
- Databricks  
- AWS Athena  
- PostgreSQL/MySQL  

Thus, it is not production-grade ETL.


---

# 7. Spark Optimization Limitations
## 7.1 Limited Use of Spark Optimization Techniques

While the pipeline uses efficient formats like Parquet and DataFrame API, the project does not heavily incorporate:

-cache() or persist()

-repartition() and coalesce()

-Broadcast joins

-Adaptive Query Execution (AQE)

-Shuffle reduction strategies

Reason:
The dataset size (approx. hundreds of thousands of rows) does not strictly require aggressive optimizations.
However, in real-world deployments:

-Caching frequently reused DataFrames

-Broadcasting small dimension tables

-Reducing shuffle partitions

would improve latency and resource usage.

## 7.2 No Cluster-Level Optimization Testing

The system runs exclusively on:

-Google Colab

-Single-machine Spark (local mode)

Thus:

-No benchmarking across multi-node clusters

-No tests with larger executors or partition tuning

-No demonstration of cluster scalability

This limits insights into production performance.

# 7. Summary

The project achieves a highly functional and well-structured pipeline but has limitations related to:
- Dataset imbalance  
- Real-time integration  
- Model generalization  
- Scalability   

These limitations form the basis of potential enhancements in future iterations, particularly the shift toward **real-time smart city parking analytics**.

