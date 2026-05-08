# =========================================================
# File: feature_engineering/oscillation_detection.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

ROLLING_WINDOW = 20

PITCH_OSCILLATION_THRESHOLD = 5
ROLL_OSCILLATION_THRESHOLD = 8

OSCILLATION_FREQUENCY_WINDOW = 30


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_FILE = DATASET_DIR / "processed_features.csv"

OUTPUT_FILE = DATASET_DIR / "processed_features.csv"


# =========================================================
# Load Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print(" Oscillation Detection Feature Engineering ")
print("========================================")

print(f"\nLoaded File:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Pitch Oscillation Features
# =========================================================

# Rolling standard deviation

flight_df["pitch_oscillation_strength"] = (
    flight_df["pitch"]
    .rolling(window=ROLLING_WINDOW)
    .std()
)

# Oscillation flag

flight_df["pitch_oscillation_detected"] = (
    flight_df["pitch_oscillation_strength"]
    > PITCH_OSCILLATION_THRESHOLD
).astype(int)


# =========================================================
# Roll Oscillation Features
# =========================================================

flight_df["roll_oscillation_strength"] = (
    flight_df["roll"]
    .rolling(window=ROLLING_WINDOW)
    .std()
)

flight_df["roll_oscillation_detected"] = (
    flight_df["roll_oscillation_strength"]
    > ROLL_OSCILLATION_THRESHOLD
).astype(int)


# =========================================================
# Oscillation Magnitude
# =========================================================

flight_df["combined_oscillation_magnitude"] = (
    flight_df["pitch_oscillation_strength"]
    + flight_df["roll_oscillation_strength"]
)


# =========================================================
# Oscillation Frequency Estimation
# =========================================================

# Detect sign changes in pitch derivative

pitch_derivative = (
    flight_df["pitch"]
    .diff()
)

pitch_sign_changes = (
    np.sign(pitch_derivative)
    .diff()
    .fillna(0)
    .ne(0)
)

flight_df["pitch_zero_crossings"] = (
    pitch_sign_changes
    .rolling(window=OSCILLATION_FREQUENCY_WINDOW)
    .sum()
)

# Approximate oscillation frequency

flight_df["pitch_oscillation_frequency"] = (
    flight_df["pitch_zero_crossings"]
    / OSCILLATION_FREQUENCY_WINDOW
)


# =========================================================
# Oscillation Stability Score
# =========================================================

flight_df["oscillation_stability_score"] = (
    1 /
    (
        1
        + flight_df["combined_oscillation_magnitude"]
    )
)

# higher score = more stable flight


# =========================================================
# Combined Oscillation Alert
# =========================================================

flight_df["oscillation_alert"] = (
    (
        flight_df["pitch_oscillation_detected"] == 1
    )
    |
    (
        flight_df["roll_oscillation_detected"] == 1
    )
).astype(int)


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
print(" Oscillation Features Generated Successfully ")
print("========================================")

print(f"\nSaved File:")
print(OUTPUT_FILE)

oscillation_feature_columns = [
    "pitch_oscillation_strength",
    "pitch_oscillation_detected",
    "roll_oscillation_strength",
    "roll_oscillation_detected",
    "combined_oscillation_magnitude",
    "pitch_oscillation_frequency",
    "oscillation_stability_score",
    "oscillation_alert"
]

print("\nGenerated Oscillation Features:")

for feature in oscillation_feature_columns:
    print(f"- {feature}")

print("\nDetected Pitch Oscillations:")
print(
    flight_df["pitch_oscillation_detected"]
    .sum()
)

print("\nDetected Roll Oscillations:")
print(
    flight_df["roll_oscillation_detected"]
    .sum()
)

print("\nTotal Oscillation Alerts:")
print(
    flight_df["oscillation_alert"]
    .sum()
)

print("\nSample Oscillation Features:")

print(
    flight_df[
        oscillation_feature_columns
    ].head()
)  # oscillation_detection.py
