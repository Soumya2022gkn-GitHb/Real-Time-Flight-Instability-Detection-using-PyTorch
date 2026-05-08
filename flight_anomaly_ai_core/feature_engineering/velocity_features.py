# =========================================================
# File: feature_engineering/velocity_features.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

VELOCITY_SPIKE_THRESHOLD = 15
VELOCITY_DROP_THRESHOLD = -15

ROLLING_WINDOW = 20


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_FILE = DATASET_DIR / "processed_features.csv"

OUTPUT_FILE = DATASET_DIR / "processed_features.csv"


# =========================================================
# Load Processed Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print(" Velocity Feature Engineering ")
print("========================================")

print(f"\nLoaded File:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Velocity Rate of Change
# =========================================================

flight_df["velocity_rate"] = (
    flight_df["velocity"]
    .diff()
)

# change in velocity per timestep


# =========================================================
# Velocity Acceleration
# =========================================================

flight_df["velocity_acceleration"] = (
    flight_df["velocity_rate"]
    .diff()
)

# second derivative


# =========================================================
# Velocity Gradient Magnitude
# =========================================================

flight_df["velocity_gradient_magnitude"] = np.abs(
    flight_df["velocity_rate"]
)


# =========================================================
# Detect Velocity Spikes
# =========================================================

flight_df["velocity_spike"] = (
    flight_df["velocity_rate"]
    > VELOCITY_SPIKE_THRESHOLD
).astype(int)


# =========================================================
# Detect Sudden Velocity Drops
# =========================================================

flight_df["velocity_drop"] = (
    flight_df["velocity_rate"]
    < VELOCITY_DROP_THRESHOLD
).astype(int)


# =========================================================
# Rolling Velocity Mean
# =========================================================

flight_df["velocity_trend"] = (
    flight_df["velocity"]
    .rolling(window=ROLLING_WINDOW)
    .mean()
)


# =========================================================
# Velocity Deviation from Trend
# =========================================================

flight_df["velocity_deviation"] = (
    flight_df["velocity"]
    - flight_df["velocity_trend"]
)


# =========================================================
# Rolling Velocity Volatility
# =========================================================

flight_df["velocity_volatility"] = (
    flight_df["velocity"]
    .rolling(window=ROLLING_WINDOW)
    .std()
)


# =========================================================
# Velocity Stability Score
# =========================================================

flight_df["velocity_stability_score"] = (
    1 /
    (
        1
        + flight_df["velocity_volatility"]
    )
)

# higher score = more stable velocity


# =========================================================
# Fill Missing Values
# =========================================================

flight_df = flight_df.bfill()


# =========================================================
# Save Updated Dataset
# =========================================================

flight_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Velocity Features Generated Successfully ")
print("========================================")

print(f"\nSaved File:")
print(OUTPUT_FILE)

velocity_feature_columns = [
    "velocity_rate",
    "velocity_acceleration",
    "velocity_gradient_magnitude",
    "velocity_spike",
    "velocity_drop",
    "velocity_trend",
    "velocity_deviation",
    "velocity_volatility",
    "velocity_stability_score"
]

print("\nGenerated Velocity Features:")

for feature in velocity_feature_columns:
    print(f"- {feature}")

print("\nDetected Velocity Spikes:")
print(
    flight_df["velocity_spike"]
    .sum()
)

print("\nDetected Velocity Drops:")
print(
    flight_df["velocity_drop"]
    .sum()
)

print("\nSample Velocity Features:")

print(
    flight_df[
        velocity_feature_columns
    ].head()
)  # velocity_features.py
