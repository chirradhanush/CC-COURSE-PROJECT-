
# 🚗 Smart Parking Occupancy Prediction & Analytics Platform  
### *An End-to-End Big Data Pipeline using Apache Spark, MLlib & Streamlit*

---

## 📌 Overview  
This project implements a full **Smart Parking Analytics Platform** capable of:  
- Cleaning and processing raw IoT parking sensor data  
- Engineering rich temporal & behavioral features  
- Training a tuned **Random Forest classification model**  
- Generating predictions + feature importances  
- Powering a **Streamlit dashboard** for interactive analytics  
- Running the entire batch pipeline with **one command**

This is built as part of the **ITCS 6190 – Cloud Computing for Data Analysis** course.

---

# 🧱 Project Architecture

```
CC-COURSE-PROJECT-
│
├── data/
│   ├── raw/                         # Raw CSV files (optional / original)
│   ├── cleaned/                     # Cleaned CSV + parquet from ingestion
│   ├── train_ready/                 # ML-ready parquet files
│   └── model/                       # predictions.csv + feature importance
│
├── src/
│   ├── prepare_data.py              # Converts cleaned CSV → parquet
│   └── ml_pipeline.py               # ML model training + artifact generation
│
├── ML/models/                       # Saved trained RandomForest model
│
├── dashboard.py                     # Streamlit dashboard for analytics
│
├── run.sh                           # One-command batch pipeline executor
└── README.md
```

---

# 🛠️ Technologies Used

### **Big Data / Compute**
- Apache Spark  
- PySpark (SQL, DataFrame API)  
- Spark MLlib  

### **Machine Learning**
- RandomForestClassifier  
- TrainValidationSplit (model tuning)  
- Feature vectorization & scaling  

### **Dashboard & Analytics**
- Streamlit  
- Plotly  
- Pandas  

### **Development & Deployment**
- Google Colab  
- GitHub (public repo)  
- Local execution with `run.sh`  

---

# 📊 Data & Feature Engineering

The raw Melbourne Smart Parking dataset contains:
- BayId, DeviceId, StreetMarker, StreetName  
- ArrivalTime, DepartureTime  
- DurationSeconds  
- AreaName, SideName, SignPlateID  
- Status (Occupied/Free)  

### From this, we engineered:
#### **Temporal Features**
- `Hour`  
- `DayOfWeek`  
- `IsWeekend`  
- `Month`  
- `DayOfMonth`  
- `PartOfDay`  

#### **Duration / Behavioral Features**
- `DwellMinutes`  
- `_duration_min`  
- `_duration_min_w` (Winsorized)  
- `_duration_robust_z` (Robust Z-score standardized)  

#### **Rolling Window Features**
- `rolling_occ_N10` → Average occupancy in recent 10 events  
- `arrivals_N10` → Number of arrivals in recent 10 events  

These capture **short-term parking dynamics**, which significantly improved ML accuracy.

#### **Categorical Encoding / Lumping**
- `street_marker_lumped`  
- `device_id_lumped`  

Lumping reduces high-cardinality categories and prevents overfitting.

#### **Target Column**
- `target_occupied` (0 = free, 1 = occupied)

---

# 📥 Data Preparation Pipeline (`prepare_data.py`)

Since the ingestion & feature engineering were done inside a notebook,  
the script:

1. Loads the cleaned CSV:
   ```
   data/cleaned/smart_parking_clean.csv
   ```
2. Writes it to Parquet:
   - `data/cleaned/smart_parking_clean.parquet`
   - `data/train_ready/` (same dataset split into parquet for ML)

No additional transformations are performed to maintain consistency  
with the notebook results.

---

# 🤖 Machine Learning Pipeline (`src/ml_pipeline.py`)

The ML pipeline includes:

### **1. Load Parquet + Select Features**
- Numeric engineered features  
- Indexing for categorical features  
- VectorAssembler → StandardScaler  

### **2. Tuning Random Forest**
Using **TrainValidationSplit**:

- `numTrees`: [100, 200]  
- `maxDepth`: [8, 12]  

Optimized for **AUC (areaUnderROC)**.

### **3. Model Metrics**
Final performance:

| Metric | Value |
|--------|--------|
| **Accuracy** | ~71% |
| **F1-score** | ~0.72 |
| **AUC** | ~0.797 |

### **4. Save Artifacts**
- Best model → `ML/models/parking_rf_model_rich/`  
- `predictions.csv` → test-set predictions + probabilities  
- `feature_importances.csv` → RF importances for dashboard  

These outputs power the Streamlit dashboard.

---

# 🎛️ Streamlit Dashboard (`dashboard.py`)

The interactive dashboard presents **four analytical views**:

---

## **1️⃣ Overview**
- Average Occupancy  
- Average Violation Rate  
- Model Accuracy  
- Model F1-score  
- Pie Chart: *Occupied vs Free*  
- Pie Chart: *Violation vs Compliant*  

Shows dataset balance & compliance behavior.

---

## **2️⃣ Demand Patterns**
- **Occupancy by Hour** → reveals peak demand times  
- **DayOfWeek × Hour Heatmap** → weekday vs weekend patterns  
- **Top 10 Busiest Streets** → highest average occupancy  

Great for city planners to allocate parking or adjust pricing.

---

## **3️⃣ Violations & Dwell Time**
- **Violation rate by area** → compliance hotspots  
- **Dwell time distribution** (0–4 hours) → behavior skew  
- **Dwell time vs violation boxplot**  
  - Violators generally stay *much longer*  
  - Clean behavioral insight  

---

## **4️⃣ ML Model Performance**
- **Confusion matrix (normalized)**  
- **ROC curve** (AUC ~0.75)  
- **Feature importance plot**  
  - Rolling occupancy  
  - Recent arrivals  
  - Hour of day  
  - Dwell time metrics  

Explains *why* the model works.

---

# 🏃 One Command Execution

From the project root:

```
./run.sh
```

This runs:

```
python src/prepare_data.py  
python src/ml_pipeline.py  
```

Generates:

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

Dashboard appears at:  
`http://localhost:8501`

---

# 📈 Example Insights

- Parking occupancy ~50% overall  
- Violations only ~3.7% (rare but important)  
- Violators have significantly longer dwell times  
- Busiest streets: 13085S, C2918, 3166N  
- Peak demand during business hours  
- Rolling-window features strongly drive model accuracy  

---

# 🔮 Future Work

- Integration with IoT APIs / Kafka  
- Live map dashboard (real-time)  
- Automated violation alerts  
- Dynamic pricing engine  
- Advanced ML models (XGBoost, CatBoost, GNNs)  


---

# 🎯 Final Notes

This repository demonstrates a complete **production-style big data workflow**, including:  
- Data ingestion & feature engineering  
- ML model training & tuning  
- Automated batch pipeline execution  
- Analytics dashboard  
- Clear interpretability & insights  

## 📽️ Video Presentation
🔗 [Click here to watch the video](https://drive.google.com/file/d/1j0-Pg-sE6UaMBUS9wlyaugZ1rfjm2uvA/view?usp=drive_link)


