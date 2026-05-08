# =========================================================
# File: data_generation/simulate_instability.py
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

OSCILLATION_AMPLITUDE = 12
ROLL_INSTABILITY_MAGNITUDE = 25
VELOCITY_FLUCTUATION = 20

INSTABILITY_DURATION = 120


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_FILE = DATASET_DIR / "flight_telemetry_with_anomalies.csv"

OUTPUT_FILE = DATASET_DIR / "flight_instability_simulation.csv"


# =========================================================
# Load Telemetry Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\nLoaded Telemetry Dataset")
print(f"Dataset Shape: {flight_df.shape}")


# =========================================================
# Ensure Anomaly Column Exists
# =========================================================

if "anomaly" not in flight_df.columns:
    flight_df["anomaly"] = 0


# =========================================================
# Simulate Flight Instability
# =========================================================

# Instability window
start_idx = 3500
end_idx = start_idx + INSTABILITY_DURATION

time_window = np.arange(INSTABILITY_DURATION)


# =========================================================
# Simulate Pitch Oscillation
# =========================================================

pitch_instability = (
    OSCILLATION_AMPLITUDE
    * np.sin(0.4 * time_window)
)

flight_df.loc[
    start_idx:end_idx - 1,
    "pitch"
] += pitch_instability


# =========================================================
# Simulate Roll Oscillation
# =========================================================

roll_instability = (
    ROLL_INSTABILITY_MAGNITUDE
    * np.sin(0.5 * time_window)
)

flight_df.loc[
    start_idx:end_idx - 1,
    "roll"
] += roll_instability


# =========================================================
# Simulate Velocity Fluctuation
# =========================================================

velocity_noise = np.random.normal(
    0,
    VELOCITY_FLUCTUATION,
    INSTABILITY_DURATION
)

flight_df.loc[
    start_idx:end_idx - 1,
    "velocity"
] += velocity_noise


# =========================================================
# Simulate Altitude Disturbance
# =========================================================

altitude_disturbance = (
    150 * np.sin(0.2 * time_window)
)

flight_df.loc[
    start_idx:end_idx - 1,
    "altitude"
] += altitude_disturbance


# =========================================================
# Mark Instability Region as Anomaly
# =========================================================

flight_df.loc[
    start_idx:end_idx - 1,
    "anomaly"
] = 1


# =========================================================
# Add Instability Type
# =========================================================

flight_df["instability_type"] = "normal"

flight_df.loc[
    start_idx:end_idx - 1,
    "instability_type"
] = "flight_instability"


# =========================================================
# Save Simulated Dataset
# =========================================================

flight_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Flight Instability Simulation Complete ")
print("========================================")

print(f"\nSaved File:")
print(OUTPUT_FILE)

print(f"\nInstability Window:")
print(f"Start Index: {start_idx}")
print(f"End Index:   {end_idx}")

print("\nInjected Instability Types:")
print("- Pitch Oscillation")
print("- Roll Oscillation")
print("- Velocity Fluctuation")
print("- Altitude Disturbance")

print(
    f"\nTotal Anomalous Samples: "
    f"{flight_df['anomaly'].sum()}"
)

print("\nSample Instability Data:")
print(
    flight_df.loc[
        start_idx:start_idx + 5,
        [
            "timestamp",
            "altitude",
            "velocity",
            "pitch",
            "roll",
            "anomaly",
            "instability_type"
        ]
    ]
)  # simulate_instability.py
