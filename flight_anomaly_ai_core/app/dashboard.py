# =========================================================
# File: app/dashboard.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# Streamlit Configuration
# =========================================================

st.set_page_config(
    page_title="Flight Telemetry Dashboard",
    page_icon="✈️",
    layout="wide"
)


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_FILE = (
    DATASET_DIR
    / "predicted_anomalies.csv"
)


# =========================================================
# Validate Dataset File
# =========================================================

if not INPUT_FILE.exists():

    st.error(
        f"Prediction dataset not found:\n{INPUT_FILE}"
    )

    st.stop()


# =========================================================
# Load Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)


# =========================================================
# Sidebar Controls
# =========================================================

st.sidebar.title("Dashboard Controls")

telemetry_feature = st.sidebar.selectbox(

    "Select Telemetry Signal",

    [
        "altitude",
        "velocity",
        "pitch",
        "roll",
        "yaw"
    ]
)

show_anomalies = st.sidebar.checkbox(
    "Highlight Anomalies",
    value=True
)

num_samples = st.sidebar.slider(

    "Telemetry Samples",

    min_value=100,

    max_value=len(flight_df),

    value=1500,

    step=100
)


# =========================================================
# Filter Dataset
# =========================================================

flight_df = flight_df.head(num_samples)


# =========================================================
# Dashboard Header
# =========================================================

st.title("✈️ Flight Telemetry Intelligence Dashboard")

st.markdown(
    """
    Interactive telemetry analytics dashboard for
    real-time flight instability detection using
    Isolation Forest and PyTorch Autoencoders.
    """
)


# =========================================================
# Compute Metrics
# =========================================================

total_samples = len(flight_df)

anomaly_count = int(

    flight_df[
        "predicted_anomaly"
    ].sum()
)

normal_count = (
    total_samples
    - anomaly_count
)

mean_anomaly_score = round(

    flight_df[
        "combined_anomaly_score"
    ].mean(),

    4
)


# =========================================================
# Display Metrics
# =========================================================

metric_col1, metric_col2, metric_col3, metric_col4 = (

    st.columns(4)
)

metric_col1.metric(
    "Telemetry Samples",
    total_samples
)

metric_col2.metric(
    "Detected Anomalies",
    anomaly_count
)

metric_col3.metric(
    "Normal Samples",
    normal_count
)

metric_col4.metric(
    "Average Anomaly Score",
    mean_anomaly_score
)


# =========================================================
# Telemetry Visualization
# =========================================================

st.subheader(
    f"{telemetry_feature.capitalize()} Telemetry Analysis"
)

fig, ax = plt.subplots(
    figsize=(15, 5)
)

# Normal telemetry line

ax.plot(

    flight_df["timestamp"],

    flight_df[telemetry_feature],

    linewidth=1.5,

    label=telemetry_feature
)


# =========================================================
# Highlight Anomalies
# =========================================================

if show_anomalies:

    anomaly_df = flight_df[

        flight_df[
            "predicted_anomaly"
        ] == 1
    ]

    ax.scatter(

        anomaly_df["timestamp"],

        anomaly_df[telemetry_feature],

        s=45,

        label="Detected Anomaly"
    )


# =========================================================
# Plot Formatting
# =========================================================

ax.set_xlabel("Timestamp")

ax.set_ylabel(
    telemetry_feature.capitalize()
)

ax.set_title(
    f"{telemetry_feature.capitalize()} vs Time"
)

ax.grid(True)

ax.legend()

st.pyplot(fig)


# =========================================================
# Anomaly Score Visualization
# =========================================================

st.subheader("Anomaly Score Timeline")

fig2, ax2 = plt.subplots(
    figsize=(15, 5)
)

ax2.plot(

    flight_df["timestamp"],

    flight_df["combined_anomaly_score"],

    linewidth=1.5,

    label="Combined Anomaly Score"
)

# Threshold line

ax2.axhline(

    y=0.60,

    linestyle="--",

    label="Detection Threshold"
)

ax2.set_xlabel("Timestamp")

ax2.set_ylabel("Anomaly Score")

ax2.set_title(
    "Anomaly Score vs Time"
)

ax2.grid(True)

ax2.legend()

st.pyplot(fig2)


# =========================================================
# Top Risk Events
# =========================================================

st.subheader("Top High-Risk Flight Events")

top_risk_events = flight_df.sort_values(

    by="combined_anomaly_score",

    ascending=False
).head(20)

st.dataframe(

    top_risk_events[
        [
            "timestamp",
            "combined_anomaly_score",
            "prediction_label"
        ]
    ]
)


# =========================================================
# Telemetry Distribution
# =========================================================

st.subheader(
    f"{telemetry_feature.capitalize()} Distribution"
)

fig3, ax3 = plt.subplots(
    figsize=(10, 5)
)

ax3.hist(

    flight_df[telemetry_feature],

    bins=40
)

ax3.set_xlabel(
    telemetry_feature.capitalize()
)

ax3.set_ylabel("Frequency")

ax3.set_title(
    f"{telemetry_feature.capitalize()} Distribution"
)

st.pyplot(fig3)


# =========================================================
# Raw Dataset Viewer
# =========================================================

with st.expander("Show Raw Telemetry Dataset"):

    st.dataframe(
        flight_df.head(100)
    )


# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.markdown(
    """
    Flight Anomaly AI Core |
    Aerospace Telemetry Intelligence Platform |
    PyTorch + Isolation Forest + Real-Time ML
    """
)  # dashboard.py
