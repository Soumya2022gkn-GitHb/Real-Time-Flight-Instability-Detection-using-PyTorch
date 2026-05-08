# =========================================================
# File: data_generation/inject_anomalies.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ALTITUDE_DROP_MAGNITUDE = 800
ROLL_SPIKE_MAGNITUDE = 35
OSCILLATION_AMPLITUDE = 8


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_FILE = DATASET_DIR / "flight_telemetry.csv"

OUTPUT_FILE = DATASET_DIR / "flight_telemetry_with_anomalies.csv"

LABEL_FILE = DATASET_DIR / "anomaly_labels.csv"


# =========================================================
# Load Flight Telemetry Data
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\nLoaded Flight Telemetry Data")
print(f"Shape: {flight_df.shape}")


# =========================================================
# Initialize Anomaly Labels
# =========================================================

flight_df["anomaly"] = 0

# 0 = normal
# 1 = anomaly


# =========================================================
# Inject Altitude Drop Anomaly
# =========================================================

altitude_start = 1000
altitude_end = 1050

flight_df.loc[
    altitude_start:altitude_end,
    "altitude"
] -= ALTITUDE_DROP_MAGNITUDE

flight_df.loc[
    altitude_start:altitude_end,
    "anomaly"
] = 1

print("\nInjected Altitude Drop Anomaly")


# =========================================================
# Inject Roll Instability
# =========================================================

roll_start = 2000
roll_end = 2050

flight_df.loc[
    roll_start:roll_end,
    "roll"
] += ROLL_SPIKE_MAGNITUDE

flight_df.loc[
    roll_start:roll_end,
    "anomaly"
] = 1

print("Injected Roll Instability")


# =========================================================
# Inject Pitch Oscillation
# =========================================================

oscillation_start = 3000
oscillation_end = 3080

oscillation_range = np.arange(
    oscillation_end - oscillation_start + 1
)

oscillation_signal = (
    OSCILLATION_AMPLITUDE
    * np.sin(0.5 * oscillation_range)
)

flight_df.loc[
    oscillation_start:oscillation_end,
    "pitch"
] += oscillation_signal

flight_df.loc[
    oscillation_start:oscillation_end,
    "anomaly"
] = 1

print("Injected Pitch Oscillation")


# =========================================================
# Create Anomaly Labels DataFrame
# =========================================================

anomaly_labels_df = flight_df[
    ["timestamp", "anomaly"]
]


# =========================================================
# Save Updated Telemetry Dataset
# =========================================================

flight_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved Anomaly Dataset:")
print(OUTPUT_FILE)


# =========================================================
# Save Anomaly Labels
# =========================================================

anomaly_labels_df.to_csv(
    LABEL_FILE,
    index=False
)

print(f"\nSaved Anomaly Labels:")
print(LABEL_FILE)


# =========================================================
# Summary
# =========================================================

print("\n========================================")
print(" Anomaly Injection Completed ")
print("========================================")

print(f"\nTotal Samples: {len(flight_df)}")

print(
    f"Total Anomalies: "
    f"{flight_df['anomaly'].sum()}"
)

print("\nAnomaly Types Injected:")
print("- Altitude Drop")
print("- Roll Instability")
print("- Pitch Oscillation")

print("\nSample Rows with Anomalies:")
print(
    flight_df[
        flight_df["anomaly"] == 1
    ].head()
)  # inject_anomalies.py
