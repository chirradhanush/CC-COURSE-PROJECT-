
# Smart Parking Occupancy Prediction 

## 🚗 Project Overview
This project implements a complete **end-to-end smart parking analytics and prediction pipeline** using **PySpark**, **Spark SQL**, **MLlib**, and a reproducible batch workflow. It processes IoT parking sensor data, performs feature engineering, executes complex queries, trains an ML model, and prepares the system for dashboarding and streaming extensions.

> **Goal:** Predict whether a parking bay will be *occupied* using rich temporal + contextual features.

---

## 🧱 Project Architecture

```
CC-COURSE-PROJECT-
│
├── data/
│   ├── raw/                     # Original CSV
├── data/cleaned/                # Cleaned dataset (from ingestion notebook)
│   ├── train_ready/             # ML-ready parquet files
│   └── model/                   # Predictions + feature importances from ML
│
├── src/
│   ├── prepare_data.py          # Produces cleaned & train_ready datasets
│   └── ml_pipeline.py           # Trains tuned RF model + saves artifacts
│
├── ML/models/                   # Saved RandomForest model
│
├── run.sh                       # One-command batch pipeline execution
└── dashboard.py (coming soon)   # Streamlit app for interactive insights
```

---

## 🛠 Technologies Used
- **Apache Spark / PySpark**
- **Spark SQL**
- **MLlib (RandomForestClassifier)**
- **TrainValidationSplit**
- **Plotly & Streamlit** (for upcoming dashboard)
- **Google Colab + GitHub** for reproducible execution

---

## 📥 1. Ingestion & Cleaning (Notebook-Based)

The enriched cleaned dataset `smart_parking_clean.csv` is produced in a dedicated notebook and includes:

- Derived time fields (`Hour`, `DayOfWeek`, `IsWeekend`, etc.)
- Duration features (`_duration_min`, `_duration_min_w`, `_duration_robust_z`)
- Rolling statistics (`rolling_occ_N10`, `arrivals_N10`)
- Lumped categorical encodings (`street_marker_lumped`, `device_id_lumped`)
- Final label: **target_occupied**

This file becomes the single source of truth for the ML pipeline.

---

## 🧼 2. prepare_data.py — Passthrough Transformer

Because the notebook already generates fully engineered features,  
`src/prepare_data.py` simply:

- Reads: `data/cleaned/smart_parking_clean.csv`
- Writes:  
  - `data/cleaned/smart_parking_clean.parquet`  
  - `data/train_ready/` (same dataset as parquet)

No additional cleaning is done to preserve accuracy consistency.

---

## 🤖 3. ml_pipeline.py — Model Training Pipeline

This script:

### ✔ Loads the enriched parquet dataset  
### ✔ Selects advanced engineered features  
### ✔ Indexes categorical columns  
### ✔ Builds a vectorized + scaled feature matrix  
### ✔ Performs **TrainValidationSplit** tuning  
- `numTrees` = *[100, 200]*
- `maxDepth` = *[8, 12]*

### ✔ Saves:
- `ML/models/parking_rf_model/`
- `data/model/predictions.csv`
- `data/model/feature_importances.csv`

These are used by the Streamlit dashboard.

---

## 📊 4. ML Performance Summary (From Pipeline)

Output from tuned Random Forest:

| Metric | Value |
|--------|--------|
| **Accuracy** | ~0.71 |
| **F1 Score** | ~0.71 |
| **AUC** | ~0.79 |


---

## 🏃 5. One-Command Execution

From project root:

```
./run.sh
```

This runs:

```
python src/prepare_data.py
python src/ml_pipeline.py
```

✔ Fully reproducible  
✔ Colab-friendly  
✔ Linux/WSL/macOS ready  

---

## 📈 6. Dashboard 

A full **Streamlit dashboard** (`dashboard.py`) visualizes:

- Peak occupancy hours  
- Heatmaps of demand (Hour × Day)  
- Violation hotspots  
- Dwell-time patterns  
- Confusion matrix  
- ROC curve  
- Feature importance plots  

To run (after ML pipeline is executed):

```
streamlit run dashboard.py
```

---

## 📦 7. How to Run in Google Colab

!apt-get install -y openjdk-11-jdk-headless -qq
!pip install pyspark==3.5.1
!git clone https://github.com/chirradhanush/CC-COURSE-PROJECT-.git
%cd CC-COURSE-PROJECT-
!chmod +x run.sh
!./run.sh




---


