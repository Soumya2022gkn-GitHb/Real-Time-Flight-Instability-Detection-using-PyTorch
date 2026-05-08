# =========================================================
# File: feature_engineering/normalize_features.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import joblib

import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

MODELS_DIR = PROJECT_ROOT / "models"

INPUT_FILE = DATASET_DIR / "processed_features.csv"

OUTPUT_FILE = DATASET_DIR / "normalized_features.csv"

SCALER_FILE = MODELS_DIR / "scaler.pkl"


# =========================================================
# Create Models Directory
# =========================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Load Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print(" Feature Normalization ")
print("========================================")

print(f"\nLoaded File:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Select Numerical Features
# =========================================================

exclude_columns = [
    "timestamp",
    "anomaly",
    "instability_type"
]

feature_columns = [
    column
    for column in flight_df.columns
    if column not in exclude_columns
]

print("\nSelected Feature Columns:")
print(feature_columns)


# =========================================================
# Create Scaler
# =========================================================

scaler = StandardScaler()


# =========================================================
# Normalize Features
# =========================================================

normalized_features = scaler.fit_transform(
    flight_df[feature_columns]
)


# =========================================================
# Create Normalized DataFrame
# =========================================================

normalized_df = pd.DataFrame(
    normalized_features,
    columns=feature_columns
)


# =========================================================
# Add Back Non-Normalized Columns
# =========================================================

for column in exclude_columns:

    if column in flight_df.columns:

        normalized_df[column] = flight_df[column]


# =========================================================
# Reorder Columns
# =========================================================

final_columns = (
    exclude_columns
    + feature_columns
)

final_columns = [
    column
    for column in final_columns
    if column in normalized_df.columns
]

normalized_df = normalized_df[final_columns]


# =========================================================
# Save Normalized Dataset
# =========================================================

normalized_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Save Scaler
# =========================================================

joblib.dump(
    scaler,
    SCALER_FILE
)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Feature Normalization Completed ")
print("========================================")

print(f"\nNormalized Dataset Saved:")
print(OUTPUT_FILE)

print(f"\nScaler Saved:")
print(SCALER_FILE)

print(f"\nNumber of Features Normalized:")
print(len(feature_columns))

print("\nNormalized Dataset Shape:")
print(normalized_df.shape)

print("\nSample Normalized Features:")
print(
    normalized_df[
        feature_columns[:5]
    ].head()
)  # normalize_features.py
