import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ===========================
# CONFIG
# ===========================
st.set_page_config(
    page_title="Smart Parking Analytics & ML Dashboard",
    layout="wide",
)


# ===========================
# DATA LOADING (CACHED)
# ===========================
@st.cache_data
def load_data():
    # Cleaned dataset (rich features)
    df = pd.read_csv("data/cleaned/smart_parking_clean.csv")

    # Predictions from ML pipeline
    preds = pd.read_csv("data/model/predictions.csv")

    # Feature importances from RF model
    fi = pd.read_csv("data/model/feature_importances.csv")

    return df, preds, fi


df, preds, fi = load_data()


# ===========================
# HELPER: NICE LABELS
# ===========================
DAY_MAP = {
    1: "Sun",
    2: "Mon",
    3: "Tue",
    4: "Wed",
    5: "Thu",
    6: "Fri",
    7: "Sat",
}

# Try to standardize column names used for plots
# We assume your cleaned CSV has these (from the original dataset)
if "DayOfWeek" in df.columns:
    df["DayOfWeekName"] = df["DayOfWeek"].map(DAY_MAP)
else:
    df["DayOfWeekName"] = np.nan

if "Hour" in df.columns:
    df["HourOfDay"] = df["Hour"]
elif "_hour" in df.columns:
    df["HourOfDay"] = df["_hour"]
else:
    df["HourOfDay"] = np.nan

if "InViolation" in df.columns:
    df["InViolationFlag"] = df["InViolation"].astype(int)
else:
    df["InViolationFlag"] = 0

if "DwellMinutes" in df.columns:
    df["DwellMinutesClean"] = df["DwellMinutes"]
elif "_duration_min" in df.columns:
    df["DwellMinutesClean"] = df["_duration_min"]
else:
    df["DwellMinutesClean"] = np.nan

# Occupancy/label in cleaned data
if "target_occupied" in df.columns:
    df["OccupiedLabel"] = df["target_occupied"].astype(int)
elif "_occupied" in df.columns:
    df["OccupiedLabel"] = df["_occupied"].astype(int)
else:
    df["OccupiedLabel"] = np.nan

# For top streets, use StreetMarker or lumped version
street_col = None
for c in ["StreetMarker", "street_marker_lumped", "street_marker"]:
    if c in df.columns:
        street_col = c
        break

area_col = "AreaName" if "AreaName" in df.columns else None


# ===========================
# SIDEBAR FILTERS
# ===========================
st.sidebar.title("Filters")

# Area filter
if area_col and df[area_col].notna().any():
    areas = sorted(df[area_col].dropna().unique().tolist())
    selected_areas = st.sidebar.multiselect(
        "Area",
        options=areas,
        default=areas[:5] if len(areas) > 5 else areas,
    )
else:
    selected_areas = None

# Street filter
if street_col and df[street_col].notna().any():
    streets = sorted(df[street_col].dropna().unique().tolist())
    default_streets = streets[:10] if len(streets) > 10 else streets
    selected_streets = st.sidebar.multiselect(
        "Street Marker",
        options=streets,
        default=default_streets,
    )
else:
    selected_streets = None

# Day of week filter
day_options = [DAY_MAP[d] for d in sorted(DAY_MAP.keys())]
selected_days = st.sidebar.multiselect(
    "Day of Week",
    options=day_options,
    default=day_options,
)

# Hour filter
hour_min, hour_max = st.sidebar.slider(
    "Hour range",
    min_value=0,
    max_value=23,
    value=(0, 23),
)


# Apply filters to df for EDA charts
df_filt = df.copy()

if selected_areas and area_col:
    df_filt = df_filt[df_filt[area_col].isin(selected_areas)]

if selected_streets and street_col:
    df_filt = df_filt[df_filt[street_col].isin(selected_streets)]

if selected_days:
    df_filt = df_filt[df_filt["DayOfWeekName"].isin(selected_days)]

df_filt = df_filt[(df_filt["HourOfDay"] >= hour_min) & (df_filt["HourOfDay"] <= hour_max)]


# ===========================
# TOP KPIs
# ===========================
st.title("Smart Parking Analytics & ML Dashboard")

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

# Overall occupancy rate
overall_occ = df["OccupiedLabel"].mean()
col_kpi1.metric("Avg Occupancy", f"{overall_occ:.2%}")

# Overall violation rate
overall_violation = df["InViolationFlag"].mean()
col_kpi2.metric("Avg Violation Rate", f"{overall_violation:.2%}")

# Model metrics from preds
if "label" in preds.columns and "prediction" in preds.columns:
    y_true = preds["label"].values
    y_pred = preds["prediction"].values

    acc = (y_true == y_pred).mean()

    # Simple F1 calculation (binary positive class=1)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    col_kpi3.metric("Model Accuracy", f"{acc:.2%}")
    col_kpi4.metric("Model F1-score", f"{f1:.2f}")
else:
    col_kpi3.metric("Model Accuracy", "N/A")
    col_kpi4.metric("Model F1-score", "N/A")


# ===========================
# LAYOUT: TABS
# ===========================
tab_overview, tab_demand, tab_violation, tab_model = st.tabs(
    ["Overview", "Demand Patterns", "Violations & Dwell Time", "ML Model Performance"]
)

# ------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------
with tab_overview:
    st.subheader("High-level view")

    col1, col2 = st.columns(2)

    # Pie: Occupied vs Free
    occ_counts = df_filt["OccupiedLabel"].value_counts(dropna=True)
    occ_df = pd.DataFrame({
        "status": ["Free" if v == 0 else "Occupied" for v in occ_counts.index],
        "count": occ_counts.values,
    })
    fig_occ = px.pie(occ_df, names="status", values="count", title="Occupied vs Free")
    col1.plotly_chart(fig_occ, use_container_width=True)

    # Pie: InViolation vs not
    viol_counts = df_filt["InViolationFlag"].value_counts(dropna=True)
    viol_df = pd.DataFrame({
        "status": ["Compliant" if v == 0 else "Violation" for v in viol_counts.index],
        "count": viol_counts.values,
    })
    fig_viol = px.pie(viol_df, names="status", values="count", title="Violations vs Compliant")
    col2.plotly_chart(fig_viol, use_container_width=True)


# ------------------------------------------------
# TAB 2: DEMAND PATTERNS
# ------------------------------------------------
with tab_demand:
    st.subheader("Parking Demand Patterns")

    col1, col2 = st.columns(2)

    # 1) Occupancy by hour of day
    hourly = (
        df_filt.groupby("HourOfDay")["OccupiedLabel"]
        .mean()
        .reset_index()
        .sort_values("HourOfDay")
    )
    fig_hour = px.line(
        hourly,
        x="HourOfDay",
        y="OccupiedLabel",
        markers=True,
        title="Average Occupancy by Hour of Day",
        labels={"OccupiedLabel": "Avg Occupancy"},
    )
    col1.plotly_chart(fig_hour, use_container_width=True)

    # 2) Heatmap: Day-of-week vs Hour
    heat_df = (
        df_filt.groupby(["DayOfWeekName", "HourOfDay"])["OccupiedLabel"]
        .mean()
        .reset_index()
    )
    if not heat_df.empty:
        heat_pivot = heat_df.pivot(
            index="DayOfWeekName", columns="HourOfDay", values="OccupiedLabel"
        ).reindex(index=day_options)  # ensure consistent order
        fig_heat = px.imshow(
            heat_pivot,
            aspect="auto",
            labels=dict(x="Hour", y="Day", color="Avg Occupancy"),
            title="Occupancy Heatmap (Day vs Hour)",
        )
        col2.plotly_chart(fig_heat, use_container_width=True)

    # 3) Top 10 busiest streets
    st.markdown("### Top 10 Busiest Streets (Avg Occupancy)")

    if street_col:
        street_stats = (
            df_filt.groupby(street_col)["OccupiedLabel"]
            .agg(["mean", "count"])
            .reset_index()
        )
        street_stats = street_stats[street_stats["count"] >= 50]  # minimum support
        street_stats = street_stats.sort_values("mean", ascending=False).head(10)

        fig_street = px.bar(
            street_stats,
            x="mean",
            y=street_col,
            orientation="h",
            title="Top 10 Streets by Avg Occupancy",
            labels={"mean": "Avg Occupancy"},
        )
        fig_street.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_street, use_container_width=True)
    else:
        st.info("Street column not found in dataset.")


# ------------------------------------------------
# TAB 3: VIOLATIONS & DWELL TIME
# ------------------------------------------------
with tab_violation:
    st.subheader("Violations & Dwell Time")

    col1, col2 = st.columns(2)

    # 1) Violation rate by area
    if area_col:
        area_stats = (
            df_filt.groupby(area_col)["InViolationFlag"]
            .agg(["mean", "count"])
            .reset_index()
        )
        area_stats = area_stats[area_stats["count"] >= 50]
        area_stats = area_stats.sort_values("mean", ascending=False).head(10)

        fig_area = px.bar(
            area_stats,
            x="mean",
            y=area_col,
            orientation="h",
            title="Violation Rate by Area (Top 10)",
            labels={"mean": "Violation Rate"},
        )
        fig_area.update_layout(yaxis={"categoryorder": "total ascending"})
        col1.plotly_chart(fig_area, use_container_width=True)
    else:
        col1.info("AreaName column not found; skipping violation-by-area chart.")

    # 2) Dwell time distribution
    dwell = df_filt["DwellMinutesClean"].dropna()
    dwell = dwell[(dwell >= 0) & (dwell <= 240)]  # 0-4 hours

    if not dwell.empty:
        fig_dwell = px.histogram(
            dwell,
            nbins=40,
            title="Dwell Time Distribution (0-4 hours)",
            labels={"value": "DwellMinutes"},
        )
        col2.plotly_chart(fig_dwell, use_container_width=True)

    # 3) Dwell vs violation
    st.markdown("### Dwell Time vs. Violation")

    dv = df_filt[["DwellMinutesClean", "InViolationFlag"]].dropna()
    dv = dv[(dv["DwellMinutesClean"] >= 0) & (dv["DwellMinutesClean"] <= 240)]

    if not dv.empty:
        fig_box = px.box(
            dv,
            x="InViolationFlag",
            y="DwellMinutesClean",
            points="all",
            labels={"InViolationFlag": "Violation (0=No, 1=Yes)", "DwellMinutesClean": "DwellMinutes"},
            title="Dwell Time by Violation Status",
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No valid dwell time data for violation analysis.")


# ------------------------------------------------
# TAB 4: MODEL PERFORMANCE
# ------------------------------------------------
with tab_model:
    st.subheader("ML Model Performance")

    col1, col2 = st.columns(2)

    # 1) Confusion matrix
    if "label" in preds.columns and "prediction" in preds.columns:
        y_true = preds["label"].values
        y_pred = preds["prediction"].values

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm_norm,
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                hovertemplate="Value: %{z:.2f}<extra></extra>",
                zmin=0,
                zmax=1,
            )
        )
        fig_cm.update_layout(
            title="Confusion Matrix (Normalized)",
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        col1.plotly_chart(fig_cm, use_container_width=True)

    else:
        col1.info("Predictions file missing 'label' or 'prediction' columns.")

    # 2) ROC curve
    if "prob_occupied" in preds.columns and "label" in preds.columns:
        y_true = preds["label"].values
        y_score = preds["prob_occupied"].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC curve (AUC = {roc_auc:.3f})",
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash"),
            )
        )
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor=None),
        )
        col2.plotly_chart(fig_roc, use_container_width=True)
    else:
        col2.info("Predictions file missing 'prob_occupied' or 'label' column.")

    st.markdown("### Feature Importances (Random Forest)")

    if not fi.empty:
        fi_sorted = fi.sort_values("importance", ascending=False).head(15)
        fig_fi = px.bar(
            fi_sorted,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Feature Importances",
            labels={"importance": "Importance"},
        )
        fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importances file is empty or not found.")
