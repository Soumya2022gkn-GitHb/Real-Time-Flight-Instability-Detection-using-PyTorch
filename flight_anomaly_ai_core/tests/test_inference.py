# =========================================================
# File: tests/test_inference.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import joblib

import numpy as np
import pandas as pd


# =========================================================
# Validate PyTorch Installation
# =========================================================

try:

    import torch
    import torch.nn as nn

except ImportError:

    raise ImportError(
        "\nPyTorch not installed.\n"
        "Run:\n"
        "pip install torch torchvision torchaudio"
    )


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

MODELS_DIR = PROJECT_ROOT / "models"


# =========================================================
# Required Files
# =========================================================

NORMALIZED_DATASET_FILE = (
    DATASET_DIR
    / "normalized_features.csv"
)

PREDICTION_FILE = (
    DATASET_DIR
    / "predicted_anomalies.csv"
)

REALTIME_LOG_FILE = (
    DATASET_DIR
    / "realtime_predictions.log"
)

ISOLATION_FOREST_MODEL_FILE = (
    MODELS_DIR
    / "isolation_forest.pkl"
)

AUTOENCODER_MODEL_FILE = (
    MODELS_DIR
    / "autoencoder.pth"
)


# =========================================================
# Test Tracking
# =========================================================

total_tests = 0

passed_tests = 0

failed_tests = 0


# =========================================================
# Helper Function
# =========================================================

def run_test(test_name, condition):
    """
    Execute test condition.
    """

    global total_tests
    global passed_tests
    global failed_tests

    total_tests += 1

    if condition:

        passed_tests += 1

        print(f"[PASS] {test_name}")

    else:

        failed_tests += 1

        print(f"[FAIL] {test_name}")


# =========================================================
# Console Header
# =========================================================

print("\n========================================")
print(" Inference Pipeline Tests Started ")
print("========================================")


# =========================================================
# Test 1: Dataset Exists
# =========================================================

run_test(

    "Normalized dataset exists",

    NORMALIZED_DATASET_FILE.exists()
)


# =========================================================
# Test 2: Prediction File Exists
# =========================================================

run_test(

    "Prediction file exists",

    PREDICTION_FILE.exists()
)


# =========================================================
# Test 3: Real-Time Log File Exists
# =========================================================

run_test(

    "Real-time inference log exists",

    REALTIME_LOG_FILE.exists()
)


# =========================================================
# Test 4: Isolation Forest Model Exists
# =========================================================

run_test(

    "Isolation Forest model exists",

    ISOLATION_FOREST_MODEL_FILE.exists()
)


# =========================================================
# Test 5: Autoencoder Model Exists
# =========================================================

run_test(

    "Autoencoder model exists",

    AUTOENCODER_MODEL_FILE.exists()
)


# =========================================================
# Load Dataset
# =========================================================

if NORMALIZED_DATASET_FILE.exists():

    dataset = pd.read_csv(
        NORMALIZED_DATASET_FILE
    )

    run_test(

        "Dataset is not empty",

        len(dataset) > 0
    )

else:

    dataset = None


# =========================================================
# Load Prediction Dataset
# =========================================================

if PREDICTION_FILE.exists():

    prediction_df = pd.read_csv(
        PREDICTION_FILE
    )

    run_test(

        "Prediction dataset is not empty",

        len(prediction_df) > 0
    )

else:

    prediction_df = None


# =========================================================
# Validate Prediction Columns
# =========================================================

required_prediction_columns = [

    "combined_anomaly_score",

    "predicted_anomaly",

    "prediction_label"
]

if prediction_df is not None:

    for column in required_prediction_columns:

        run_test(

            f"Prediction column exists: {column}",

            column in prediction_df.columns
        )


# =========================================================
# Load Isolation Forest Model
# =========================================================

try:

    isolation_forest_model = joblib.load(
        ISOLATION_FOREST_MODEL_FILE
    )

    run_test(

        "Isolation Forest model loads successfully",

        isolation_forest_model is not None
    )

except Exception:

    run_test(

        "Isolation Forest model loads successfully",

        False
    )


# =========================================================
# Define Autoencoder
# =========================================================

class Autoencoder(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim=8
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed


# =========================================================
# Load Autoencoder Model
# =========================================================

try:

    if dataset is not None:

        exclude_columns = [

            "timestamp",

            "anomaly",

            "instability_type"
        ]

        feature_columns = [

            column
            for column in dataset.columns

            if column not in exclude_columns
        ]

        input_dim = len(feature_columns)

        autoencoder_model = Autoencoder(
            input_dim=input_dim
        )

        autoencoder_model.load_state_dict(

            torch.load(
                AUTOENCODER_MODEL_FILE,
                map_location="cpu"
            )
        )

        autoencoder_model.eval()

        run_test(

            "Autoencoder model loads successfully",

            autoencoder_model is not None
        )

except Exception:

    run_test(

        "Autoencoder model loads successfully",

        False
    )


# =========================================================
# Test Inference Prediction Capability
# =========================================================

try:

    if dataset is not None:

        feature_columns = [

            column
            for column in dataset.columns

            if column not in [

                "timestamp",

                "anomaly",

                "instability_type"
            ]
        ]

        X = dataset[
            feature_columns
        ].values[:5]

        predictions = isolation_forest_model.predict(X)

        run_test(

            "Inference predictions generated",

            len(predictions) == 5
        )

except Exception:

    run_test(

        "Inference predictions generated",

        False
    )


# =========================================================
# Validate Prediction Labels
# =========================================================

if prediction_df is not None:

    unique_labels = prediction_df[
        "prediction_label"
    ].unique()

    valid_labels = [

        "normal",

        "anomaly"
    ]

    run_test(

        "Prediction labels valid",

        all(
            label in valid_labels
            for label in unique_labels
        )
    )


# =========================================================
# Validate Anomaly Scores
# =========================================================

if prediction_df is not None:

    run_test(

        "Anomaly scores are numeric",

        np.issubdtype(

            prediction_df[
                "combined_anomaly_score"
            ].dtype,

            np.number
        )
    )


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Inference Test Summary ")
print("========================================")

print(f"\nTotal Tests  : {total_tests}")

print(f"Passed Tests : {passed_tests}")

print(f"Failed Tests : {failed_tests}")


# =========================================================
# Final Status
# =========================================================

if failed_tests == 0:

    print("\nAll inference tests passed!")

else:

    print("\nSome inference tests failed.")  # test_inference.py
