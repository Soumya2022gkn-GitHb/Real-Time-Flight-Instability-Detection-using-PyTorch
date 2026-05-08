# =========================================================
# File: feature_engineering/rolling_statistics.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import pandas as pd


# =========================================================
# Configuration
# =========================================================

ROLLING_WINDOW = 20


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_FILE = DATASET_DIR / "flight_instability_simulation.csv"

OUTPUT_FILE = DATASET_DIR / "processed_features.csv"


# =========================================================
# Load Telemetry Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print(" Rolling Statistics Feature Engineering ")
print("========================================")

print(f"\nLoaded File:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Telemetry Columns
# =========================================================

telemetry_columns = [
    "altitude",
    "velocity",
    "pitch",
    "roll",
    "yaw"
]


# =========================================================
# Generate Rolling Features
# =========================================================

for column in telemetry_columns:

    # -----------------------------------------------------
    # Rolling Mean
    # -----------------------------------------------------

    flight_df[f"{column}_rolling_mean"] = (
        flight_df[column]
        .rolling(window=ROLLING_WINDOW)
        .mean()
    )

    # -----------------------------------------------------
    # Rolling Standard Deviation
    # -----------------------------------------------------

    flight_df[f"{column}_rolling_std"] = (
        flight_df[column]
        .rolling(window=ROLLING_WINDOW)
        .std()
    )

    # -----------------------------------------------------
    # Rolling Minimum
    # -----------------------------------------------------

    flight_df[f"{column}_rolling_min"] = (
        flight_df[column]
        .rolling(window=ROLLING_WINDOW)
        .min()
    )

    # -----------------------------------------------------
    # Rolling Maximum
    # -----------------------------------------------------

    flight_df[f"{column}_rolling_max"] = (
        flight_df[column]
        .rolling(window=ROLLING_WINDOW)
        .max()
    )

    # -----------------------------------------------------
    # Rolling Range
    # -----------------------------------------------------

    flight_df[f"{column}_rolling_range"] = (
        flight_df[f"{column}_rolling_max"]
        - flight_df[f"{column}_rolling_min"]
    )


# =========================================================
# Fill Missing Values
# =========================================================

flight_df = flight_df.bfill()


# =========================================================
# Save Processed Features
# =========================================================

flight_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Rolling Feature Engineering Completed ")
print("========================================")

print(f"\nSaved Processed Features:")
print(OUTPUT_FILE)

print(f"\nRolling Window Size:")
print(ROLLING_WINDOW)

print(f"\nFinal Dataset Shape:")
print(flight_df.shape)

print("\nGenerated Features Example:")

generated_columns = [
    col for col in flight_df.columns
    if "rolling" in col
]

print(generated_columns[:10])

print("\nSample Processed Data:")
print(
    flight_df[
        generated_columns[:5]
    ].head()
)  # rolling_statistics.py
