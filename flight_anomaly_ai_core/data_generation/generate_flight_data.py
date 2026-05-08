# =========================================================
# File: data_generation/generate_flight_data.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

NUM_SAMPLES = 5000
TIME_STEP = 1  # seconds
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# =========================================================
# Define Project Paths
# =========================================================

# Current file:
# flight_anomaly_ai_core/data_generation/generate_flight_data.py

CURRENT_FILE = Path(__file__).resolve()

# Project root:
# flight_anomaly_ai_core/

PROJECT_ROOT = CURRENT_FILE.parent.parent

# Dataset directory:
# flight_anomaly_ai_core/dataset/

DATASET_DIR = PROJECT_ROOT / "dataset"

# Output file:
# flight_anomaly_ai_core/dataset/flight_telemetry.csv

OUTPUT_FILE = DATASET_DIR / "flight_telemetry.csv"


# =========================================================
# Create Dataset Directory
# =========================================================

DATASET_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Generate Time Axis
# =========================================================

timestamps = np.arange(
    start=0,
    stop=NUM_SAMPLES * TIME_STEP,
    step=TIME_STEP
)


# =========================================================
# Simulate Flight Telemetry
# =========================================================

# ---------------------------------------------------------
# Altitude (meters)
# ---------------------------------------------------------

altitude = (
    10000
    + 100 * np.sin(0.01 * timestamps)
    + np.random.normal(0, 5, NUM_SAMPLES)
)

# ---------------------------------------------------------
# Velocity (m/s)
# ---------------------------------------------------------

velocity = (
    250
    + 5 * np.sin(0.02 * timestamps)
    + np.random.normal(0, 1, NUM_SAMPLES)
)

# ---------------------------------------------------------
# Pitch (degrees)
# ---------------------------------------------------------

pitch = (
    2
    + 0.5 * np.sin(0.05 * timestamps)
    + np.random.normal(0, 0.2, NUM_SAMPLES)
)

# ---------------------------------------------------------
# Roll (degrees)
# ---------------------------------------------------------

roll = (
    1.5 * np.sin(0.03 * timestamps)
    + np.random.normal(0, 0.3, NUM_SAMPLES)
)

# ---------------------------------------------------------
# Yaw (degrees)
# ---------------------------------------------------------

yaw = (
    90
    + 2 * np.sin(0.01 * timestamps)
    + np.random.normal(0, 0.5, NUM_SAMPLES)
)


# =========================================================
# Create Flight Telemetry DataFrame
# =========================================================

flight_df = pd.DataFrame({
    "timestamp": timestamps,
    "altitude": altitude,
    "velocity": velocity,
    "pitch": pitch,
    "roll": roll,
    "yaw": yaw
})


# =========================================================
# Save Dataset
# =========================================================

flight_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Console Output
# =========================================================

print("\n========================================")
print(" Flight Telemetry Generated Successfully ")
print("========================================")

print(f"\nSaved File:")
print(f"{OUTPUT_FILE}")

print(f"\nDataset Shape:")
print(f"{flight_df.shape}")

print("\nColumns:")
print(list(flight_df.columns))

print("\nFirst 5 Rows:")
print(flight_df.head())
