
# Reproduction Guide

This document provides a complete, step-by-step guide to **reproduce the entire Smart Parking Occupancy Prediction & Analytics Platform**, including:

- Environment setup  
- Data preparation  
- Running the batch pipeline  
- Viewing ML outputs  
- Launching the dashboard  
- Optional: Running on Google Colab  

---

# 1. Repository Structure

The project repository should look like this:

```
CC-COURSE-PROJECT-
│
├── data/
│   ├── cleaned/
│   │   └── smart_parking_clean.csv
│   ├── train_ready/
│   └── model/
│       ├── predictions.csv
│       └── feature_importances.csv
│
├── src/
│   ├── prepare_data.py
│   └── ml_pipeline.py
│
├── ML/models/
│   └── parking_rf_model_rich/   # Saved Random Forest model
│
├── dashboard.py
├── run.sh
└── README.md
```

---

# 2. Prerequisites

You can run the project either **locally** or on **Google Colab**.

---

# 2.1 Local Setup (Recommended for final demo)

## A. Install Python
Python 3.10 or 3.11 recommended.

Check with:
```
python3 --version
```

---

## B. Install Java (Spark requirement)

Spark requires Java 8 or Java 11.

### **Windows (WSL2 recommended)**  
Inside Ubuntu terminal:
```
sudo apt update
sudo apt install -y openjdk-11-jdk
```

Verify:
```
java -version
```

---

## C. Install Python Dependencies

Inside your project root:
```
pip install pyspark==3.5.1 streamlit plotly pandas scikit-learn
```

---

# 3. Data Preparation

## 3.1 Obtain Cleaned Dataset

Place the cleaned dataset here:

```
data/cleaned/smart_parking_clean.csv
```

This file is essential — the entire ML pipeline depends on it.

---

# 4. Running the End-to-End Batch Pipeline

The pipeline is executed using a single command:

```
./run.sh
```

Ensure executable permission:

```
chmod +x run.sh
```

---

# 4.1 What run.sh Does

The script triggers:

### 1️⃣ Data preparation  
```
python src/prepare_data.py
```
→ Converts cleaned CSV into:

- Parquet format  
- Train-ready dataset  

### 2️⃣ ML training  
```
python src/ml_pipeline.py
```
→ Trains Random Forest  
→ Generates:
- predictions.csv  
- feature_importances.csv  
- saved model in ML/models/  

Outputs are stored in:

```
data/model/
ML/models/
```

---

# 5. Verifying Outputs

After running `./run.sh`, ensure the following files are present:

### ✔ Parquet data:
```
data/cleaned/smart_parking_clean.parquet
data/train_ready/*.parquet
```

### ✔ ML artifacts:
```
data/model/predictions.csv
data/model/feature_importances.csv
ML/models/parking_rf_model_rich/
```

---

# 6. Running the Dashboard

To launch the Streamlit dashboard:

```
streamlit run dashboard.py
```

This opens a URL such as:

```
http://localhost:8501
```

---

# 6.1 Dashboard Tabs

### ✔ **Overview**
- Occupancy  
- Violation rate  
- ML accuracy & F1  
- Pie charts  

### ✔ **Demand Patterns**
- Hourly demand  
- Heatmap  
- Busiest streets  

### ✔ **Violations & Dwell Time**
- Violation hotspots  
- Boxplots  
- Dwell time distributions  

### ✔ **ML Performance**
- Confusion matrix  
- ROC curve  
- Feature importances  

---

# 7. Reproducing on Google Colab (Optional)

If using Colab:

## 7.1 Install Dependencies
```
!apt-get install openjdk-11-jdk-headless -qq
!pip install pyspark==3.5.1 streamlit pandas plotly scikit-learn
```

## 7.2 Clone the Repository
```
!git clone https://github.com/chirradhanush/CC-COURSE-PROJECT-.git
%cd CC-COURSE-PROJECT-
```

## 7.3 Upload Cleaned CSV
Upload via Colab sidebar into:
```
data/cleaned/smart_parking_clean.csv
```

## 7.4 Run Pipeline
```
!chmod +x run.sh
!./run.sh
```

---

# 8. Troubleshooting

### 8.1 "Java gateway exit" (Windows)
Occurs if Spark runs outside WSL.  
**Fix:** Use WSL2 Ubuntu.

### 8.2 No such file: cleaned CSV
Ensure:
```
data/cleaned/smart_parking_clean.csv
```
exists.

### 8.3 Dashboard not updating
Delete:
```
__pycache__/
```
and restart streamlit.


---

# 9. Summary

This reproduction guide ensures that any user can:

- Set up environment  
- Prepare data  
- Execute the pipeline  
- Train the ML model  
- Generate artifacts  
- Launch dashboard  

The entire system is reproducible using a single command and supports both local and Colab execution.


