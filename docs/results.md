
# Results

This document summarizes the results of the **Smart Parking Occupancy Prediction & Analytics Platform**, including exploratory data insights, machine learning model performance, feature importance analysis, and dashboard outputs. Screenshot placeholders are included where visualizations should be inserted.

---

# 1. Exploratory Data Analysis (EDA) Results

The cleaned and engineered dataset generated several key insights into parking behavior:

---

## 1.1 Occupancy Balance
- **Occupied:** ~50–52%  
- **Free:** ~48–50%  

```
![PLACEHOLDER: Pie chart of Occupied vs Free]
```

Interpretation:  
Parking demand is moderately balanced across the dataset. Some areas experience much higher load, while others remain under-utilized.

---

## 1.2 Violation Rate
- **Violation events:** ~3.7%  
- **Compliant events:** ~96.3%  

```
![PLACEHOLDER: Pie chart of Violations vs Compliant]
```

Interpretation:  
Violations are rare, which is typical in real parking systems, contributing to class imbalance.

---

## 1.3 Busiest Streets / High-Demand Segments

Examples:
- 13085S (~0.52 average occupancy)
- C2918 (~0.50)
- 3166N (~0.50)

```
![PLACEHOLDER: Top 10 busiest streets bar chart]
```

Interpretation:  
These streets exhibit significantly higher sensor activity and are prime candidates for resource allocation and dynamic pricing.

---

## 1.4 Dwell Time Patterns
Most dwell times fall between **0–40 minutes**, with a **long right tail** up to ~3–4 hours.

```
![PLACEHOLDER: Dwell Time Histogram]
```

Interpretation:  
Short stays dominate (errands, quick visits), but long stays contribute disproportionately to violations.

---

## 1.5 Violations vs Dwell Time
Violators have **much higher median dwell time** than compliant parkers.

```
![PLACEHOLDER: Dwell Time vs Violation boxplot]
```

Interpretation:  
Strong behavioral pattern: overstays correlate directly with violation risk.

---

# 2. ML Model Comparison Results

Extensive experimentation with different machine learning algorithms was performed. Below is the summary:

---

## 2.1 Performance Summary Table

| Model | Accuracy | F1-score | AUC | Notes |
|-------|----------|----------|------|------|
| Logistic Regression | ~58% | ~0.57 | ~0.60 | Weak linear separation |
| Linear SVC | ~60% | ~0.59 | N/A | Sensitive to imbalance |
| Decision Tree | ~62% | ~0.61 | ~0.64 | Overfits easily |
| Gradient-Boosted Trees | ~69% | ~0.68 | ~0.73 | Slow training |
| **Random Forest (Final)** | **~71%** | **~0.70** | **~0.75** | Best performance |
| Stacking Ensemble | ~70% | ~0.69 | ~0.74 | Complex, minimal gain |

---

## 2.2 Final Model Metrics (Random Forest)

### ✔ Accuracy: **~71%**  
### ✔ F1-Score: **~0.70**  
### ✔ AUC: **~0.75**  
### ✔ Balanced performance on test set  
### ✔ Low overfitting compared to decision tree  

```
![PLACEHOLDER: Confusion Matrix of Random Forest predictions]
```

---

# 3. ROC Curve & AUC

```
![PLACEHOLDER: ROC Curve]
```

Interpretation:  
The Random Forest shows strong separability between classes, with an AUC around **0.75**, meaning the model can distinguish occupied vs free ~75% of the time.

---

# 4. Feature Importance Analysis

Top predictive features include:
- **rolling_occ_N10** (recent occupancy behavior)
- **arrivals_N10**
- **Hour**
- **DwellMinutes**
- **DayOfWeek**
- **street_marker_lumped**

```
![PLACEHOLDER: Feature Importance Bar Plot]
```

Interpretation:  
Short-term parking trends are highly predictive — occupancy often follows temporal and behavioral cycles.

---

# 5. Predictions Output

The ML pipeline generates:

### 📄 `predictions.csv`  
Contains:
- prediction  
- probability  
- label (ground truth)  
- selected features  

Example columns:
```
prediction, rf_prob_occupied, label, Hour, DayOfWeek, DwellMinutes, rolling_occ_N10, ...
```

```
![PLACEHOLDER: predictions.csv preview screenshot]
```

---

# 6. Dashboard Results Summary

The dashboard consolidates key metrics and visualizations:

---

## 6.1 Overview Tab
- Occupancy rate  
- Violation rate  
- ML accuracy  
- ML F1-score  
- Pie charts  

```
![PLACEHOLDER: Dashboard Overview screenshot]
```

---

## 6.2 Demand Patterns Tab
- Hourly occupancy chart  
- Heatmap (DayOfWeek × Hour)  
- Busiest street segments  

```
![PLACEHOLDER: Demand Patterns screenshot]
```

---

## 6.3 Violations & Dwell Time Tab
- Violation rate by area  
- Dwell time histogram  
- Boxplot of Dwell Time vs Violation  

```
![PLACEHOLDER: Violations tab screenshot]
```

---

## 6.4 ML Performance Tab
- Confusion matrix  
- ROC curve  
- Feature importance  

```
![PLACEHOLDER: ML performance tab screenshot]
```

---

# 7. Interpretation & Impact

The final Random Forest model performs well given:
- Real-world noisy sensor data  
- Missing environmental factors  
- Class imbalance  
- Large number of temporal and categorical features  

Despite these constraints, the system provides actionable insights:

- **Predictive analytics for city planners**  
- **Operational guidance for enforcement teams**  
- **Evidence-based policy improvements**  
- **Foundation for real-time smart parking system**

---

# 8. Summary

This results document demonstrates that the project delivers:

✔ Strong EDA insights  
✔ A well-experimented ML pipeline with multiple models  
✔ A tuned Random Forest achieving ~71% accuracy  
✔ Useful feature-based interpretations  
✔ An interactive dashboard showing clear urban mobility patterns  

The system is effective, scalable, and ready for future enhancements such as streaming and real-time occupancy prediction.

