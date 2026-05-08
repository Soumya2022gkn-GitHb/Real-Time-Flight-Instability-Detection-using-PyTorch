# =========================================================
# File: app/app.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# Streamlit Page Configuration
# =========================================================

st.set_page_config(
    page_title="Flight Anomaly AI Core",
    page_icon="✈️",
    layout="wide"
)


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

PREDICTION_FILE = (
    DATASET_DIR
    / "predicted_anomalies.csv"
)


# =========================================================
# Validate Prediction File
# =========================================================

if not PREDICTION_FILE.exists():

    st.error(
        f"Prediction file not found:\n{PREDICTION_FILE}"
    )

    st.stop()


# =========================================================
# Load Dataset
# =========================================================

flight_df = pd.read_csv(PREDICTION_FILE)


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("Flight Telemetry Controls")

selected_feature = st.sidebar.selectbox(

    "Select Telemetry Feature",

    [
        "altitude",
        "velocity",
        "pitch",
        "roll",
        "yaw"
    ]
)

show_anomalies = st.sidebar.checkbox(
    "Show Anomalies",
    value=True
)

max_rows = st.sidebar.slider(
    "Number of Rows",
    min_value=100,
    max_value=len(flight_df),
    value=1000,
    step=100
)


# =========================================================
# Filter Dataset
# =========================================================

flight_df = flight_df.head(max_rows)


# =========================================================
# Dashboard Title
# =========================================================

st.title("✈️ Flight Anomaly AI Core")

st.markdown(
    """
    Real-time flight telemetry anomaly detection dashboard
    powered by Isolation Forest and PyTorch Autoencoders.
    """
)


# =========================================================
# Dataset Statistics
# =========================================================

total_samples = len(flight_df)

total_anomalies = flight_df[
    "predicted_anomaly"
].sum()

normal_samples = (
    total_samples
    - total_anomalies
)


# =========================================================
# Metrics Row
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Samples",
    total_samples
)

col2.metric(
    "Detected Anomalies",
    int(total_anomalies)
)

col3.metric(
    "Normal Samples",
    int(normal_samples)
)


# =========================================================
# Telemetry Plot
# =========================================================

st.subheader(
    f"{selected_feature.capitalize()} Telemetry"
)

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(

    flight_df["timestamp"],

    flight_df[selected_feature],

    label=selected_feature
)


# =========================================================
# Highlight Anomalies
# =========================================================

if show_anomalies:

    anomaly_df = flight_df[

        flight_df["predicted_anomaly"] == 1
    ]

    ax.scatter(

        anomaly_df["timestamp"],

        anomaly_df[selected_feature],

        s=40,

        label="Anomaly"
    )


# =========================================================
# Plot Formatting
# =========================================================

ax.set_xlabel("Timestamp")

ax.set_ylabel(selected_feature)

ax.set_title(
    f"{selected_feature.capitalize()} vs Time"
)

ax.legend()

ax.grid(True)

st.pyplot(fig)


# =========================================================
# Anomaly Score Plot
# =========================================================

st.subheader("Combined Anomaly Score")

fig2, ax2 = plt.subplots(
    figsize=(14, 5)
)

ax2.plot(

    flight_df["timestamp"],

    flight_df["combined_anomaly_score"],

    label="Anomaly Score"
)

ax2.set_xlabel("Timestamp")

ax2.set_ylabel("Score")

ax2.set_title(
    "Combined Anomaly Score vs Time"
)

ax2.grid(True)

st.pyplot(fig2)


# =========================================================
# Recent Anomalies Table
# =========================================================

st.subheader("Recent Detected Anomalies")

recent_anomalies = flight_df[

    flight_df["predicted_anomaly"] == 1

].tail(20)

st.dataframe(

    recent_anomalies[
        [
            "timestamp",
            "combined_anomaly_score",
            "prediction_label"
        ]
    ]
)


# =========================================================
# Raw Dataset Preview
# =========================================================

with st.expander("Show Raw Dataset"):

    st.dataframe(
        flight_df.head(100)
    )


# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.markdown(
    "Flight Telemetry Intelligence Platform | "
    "PyTorch + Isolation Forest + Real-Time Inference"
)  # app.py
