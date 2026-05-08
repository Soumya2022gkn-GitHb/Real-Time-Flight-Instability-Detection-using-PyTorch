# =========================================================
# File: app/telemetry_viewer.py
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
    page_title="Telemetry Viewer",
    page_icon="📡",
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
        f"Telemetry dataset not found:\n{INPUT_FILE}"
    )

    st.stop()


# =========================================================
# Load Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)


# =========================================================
# Sidebar Controls
# =========================================================

st.sidebar.title("Telemetry Viewer Controls")

selected_signal = st.sidebar.selectbox(

    "Select Telemetry Signal",

    [
        "altitude",
        "velocity",
        "pitch",
        "roll",
        "yaw"
    ]
)

sample_limit = st.sidebar.slider(

    "Number of Samples",

    min_value=100,

    max_value=len(flight_df),

    value=1000,

    step=100
)

show_anomaly_points = st.sidebar.checkbox(
    "Show Anomaly Points",
    value=True
)

show_statistics = st.sidebar.checkbox(
    "Show Statistics",
    value=True
)


# =========================================================
# Filter Dataset
# =========================================================

flight_df = flight_df.head(sample_limit)


# =========================================================
# App Header
# =========================================================

st.title("📡 Flight Telemetry Viewer")

st.markdown(
    """
    Interactive telemetry signal viewer for
    aerospace telemetry analysis and anomaly tracking.
    """
)


# =========================================================
# Dataset Statistics
# =========================================================

if show_statistics:

    st.subheader("Telemetry Statistics")

    stats_col1, stats_col2, stats_col3, stats_col4 = (

        st.columns(4)
    )

    stats_col1.metric(
        "Samples",
        len(flight_df)
    )

    stats_col2.metric(
        "Mean",
        round(
            flight_df[selected_signal].mean(),
            2
        )
    )

    stats_col3.metric(
        "Max",
        round(
            flight_df[selected_signal].max(),
            2
        )
    )

    stats_col4.metric(
        "Min",
        round(
            flight_df[selected_signal].min(),
            2
        )
    )


# =========================================================
# Signal Visualization
# =========================================================

st.subheader(
    f"{selected_signal.capitalize()} Signal"
)

fig, ax = plt.subplots(
    figsize=(15, 5)
)

# Main telemetry line

ax.plot(

    flight_df["timestamp"],

    flight_df[selected_signal],

    linewidth=1.5,

    label=selected_signal
)


# =========================================================
# Highlight Anomalies
# =========================================================

if show_anomaly_points:

    anomaly_df = flight_df[

        flight_df[
            "predicted_anomaly"
        ] == 1
    ]

    ax.scatter(

        anomaly_df["timestamp"],

        anomaly_df[selected_signal],

        s=45,

        label="Anomaly"
    )


# =========================================================
# Plot Formatting
# =========================================================

ax.set_xlabel("Timestamp")

ax.set_ylabel(
    selected_signal.capitalize()
)

ax.set_title(
    f"{selected_signal.capitalize()} Telemetry Signal"
)

ax.grid(True)

ax.legend()

st.pyplot(fig)


# =========================================================
# Signal Distribution
# =========================================================

st.subheader(
    f"{selected_signal.capitalize()} Distribution"
)

fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)

ax2.hist(

    flight_df[selected_signal],

    bins=40
)

ax2.set_xlabel(
    selected_signal.capitalize()
)

ax2.set_ylabel("Frequency")

ax2.set_title(
    f"{selected_signal.capitalize()} Distribution"
)

st.pyplot(fig2)


# =========================================================
# Anomaly Table
# =========================================================

st.subheader("Detected Anomaly Events")

anomaly_events = flight_df[

    flight_df[
        "predicted_anomaly"
    ] == 1
]

if len(anomaly_events) > 0:

    st.dataframe(

        anomaly_events[
            [
                "timestamp",
                selected_signal,
                "combined_anomaly_score",
                "prediction_label"
            ]
        ]
    )

else:

    st.info(
        "No anomaly events detected."
    )


# =========================================================
# Raw Dataset Viewer
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
    """
    Flight Telemetry Viewer |
    Aerospace Signal Analytics |
    Real-Time Telemetry Intelligence
    """
)  # telemetry_viewer.py
