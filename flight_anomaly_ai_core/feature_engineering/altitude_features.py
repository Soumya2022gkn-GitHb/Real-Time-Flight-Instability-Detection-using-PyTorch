# =========================================================
# File: feature_engineering/altitude_features.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

ALTITUDE_DROP_THRESHOLD = -50
CLIMB_RATE_THRESHOLD = 50


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
print(" Altitude Feature Engineering ")
print("========================================")

print(f"\nLoaded File:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Altitude Rate of Change
# =========================================================

flight_df["altitude_rate"] = (
    flight_df["altitude"]
    .diff()
)

# meters per timestep


# =========================================================
# Altitude Acceleration
# =========================================================

flight_df["altitude_acceleration"] = (
    flight_df["altitude_rate"]
    .diff()
)

# change in climb/descent rate


# =========================================================
# Altitude Gradient Magnitude
# =========================================================

flight_df["altitude_gradient_magnitude"] = np.abs(
    flight_df["altitude_rate"]
)


# =========================================================
# Detect Sudden Altitude Drops
# =========================================================

flight_df["sudden_altitude_drop"] = (
    flight_df["altitude_rate"]
    < ALTITUDE_DROP_THRESHOLD
).astype(int)


# =========================================================
# Detect Rapid Climb Events
# =========================================================

flight_df["rapid_climb_event"] = (
    flight_df["altitude_rate"]
    > CLIMB_RATE_THRESHOLD
).astype(int)


# =========================================================
# Rolling Altitude Trend
# =========================================================

flight_df["altitude_trend"] = (
    flight_df["altitude"]
    .rolling(window=20)
    .mean()
)


# =========================================================
# Altitude Deviation from Trend
# =========================================================

flight_df["altitude_deviation"] = (
    flight_df["altitude"]
    - flight_df["altitude_trend"]
)


# =========================================================
# Rolling Altitude Volatility
# =========================================================

flight_df["altitude_volatility"] = (
    flight_df["altitude"]
    .rolling(window=20)
    .std()
)


# =========================================================
# Fill Missing Values
# =========================================================

flight_df = flight_df.bfill()


# =========================================================
# Save Updated Features
# =========================================================

flight_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Altitude Features Generated Successfully ")
print("========================================")

print(f"\nSaved File:")
print(OUTPUT_FILE)

print("\nGenerated Altitude Features:")

altitude_feature_columns = [
    "altitude_rate",
    "altitude_acceleration",
    "altitude_gradient_magnitude",
    "sudden_altitude_drop",
    "rapid_climb_event",
    "altitude_trend",
    "altitude_deviation",
    "altitude_volatility"
]

for feature in altitude_feature_columns:
    print(f"- {feature}")

print("\nDetected Sudden Altitude Drops:")

print(
    flight_df["sudden_altitude_drop"]
    .sum()
)

print("\nSample Altitude Features:")

print(
    flight_df[
        altitude_feature_columns
    ].head()
)  # altitude_features.py
