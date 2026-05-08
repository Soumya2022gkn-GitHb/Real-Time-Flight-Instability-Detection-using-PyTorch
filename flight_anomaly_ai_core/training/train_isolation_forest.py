# =========================================================
# File: training/train_isolation_forest.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import joblib

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 42

N_ESTIMATORS = 200

CONTAMINATION = 0.05


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

MODELS_DIR = PROJECT_ROOT / "models"

INPUT_FILE = DATASET_DIR / "normalized_features.csv"

MODEL_FILE = MODELS_DIR / "isolation_forest.pkl"

PREDICTION_FILE = DATASET_DIR / "isolation_forest_predictions.csv"


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
print(" Isolation Forest Training ")
print("========================================")

print(f"\nLoaded File:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Select Features
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

X = flight_df[feature_columns]


# =========================================================
# Ground Truth Labels
# =========================================================

if "anomaly" in flight_df.columns:

    y_true = flight_df["anomaly"]

else:

    y_true = None


# =========================================================
# Create Isolation Forest Model
# =========================================================

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_SEED
)


# =========================================================
# Train Model
# =========================================================

print("\nTraining Isolation Forest Model...")

model.fit(X)

print("Training Completed!")


# =========================================================
# Predict Anomalies
# =========================================================

predictions = model.predict(X)

# Isolation Forest Output:
#  1  = normal
# -1  = anomaly


# =========================================================
# Convert Predictions
# =========================================================

flight_df["isolation_forest_prediction"] = predictions

flight_df["predicted_anomaly"] = (
    flight_df["isolation_forest_prediction"]
    .map({
        1: 0,
        -1: 1
    })
)


# =========================================================
# Compute Anomaly Scores
# =========================================================

flight_df["anomaly_score"] = (
    model.decision_function(X)
)

# lower score = more anomalous


# =========================================================
# Save Predictions
# =========================================================

flight_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# =========================================================
# Save Trained Model
# =========================================================

joblib.dump(
    model,
    MODEL_FILE
)


# =========================================================
# Evaluation Metrics
# =========================================================

if y_true is not None:

    print("\n========================================")
    print(" Classification Report ")
    print("========================================")

    print(
        classification_report(
            y_true,
            flight_df["predicted_anomaly"]
        )
    )


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Isolation Forest Training Completed ")
print("========================================")

print(f"\nModel Saved:")
print(MODEL_FILE)

print(f"\nPredictions Saved:")
print(PREDICTION_FILE)

print(f"\nNumber of Features Used:")
print(len(feature_columns))

print(f"\nDetected Anomalies:")
print(
    flight_df["predicted_anomaly"]
    .sum()
)

print("\nSample Predictions:")

print(
    flight_df[
        [
            "timestamp",
            "anomaly_score",
            "predicted_anomaly"
        ]
    ].head()
)  # train_isolation_forest.py
