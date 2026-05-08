# =========================================================
# File: tests/test_training.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import joblib

import pandas as pd
import numpy as np


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

ISOLATION_FOREST_MODEL_FILE = (
    MODELS_DIR
    / "isolation_forest.pkl"
)

AUTOENCODER_MODEL_FILE = (
    MODELS_DIR
    / "autoencoder.pth"
)

LOSS_HISTORY_FILE = (
    DATASET_DIR
    / "autoencoder_training_loss.csv"
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
print(" Training Pipeline Tests Started ")
print("========================================")


# =========================================================
# Test 1: Dataset Exists
# =========================================================

run_test(

    "Normalized dataset exists",

    NORMALIZED_DATASET_FILE.exists()
)


# =========================================================
# Test 2: Isolation Forest Model Exists
# =========================================================

run_test(

    "Isolation Forest model exists",

    ISOLATION_FOREST_MODEL_FILE.exists()
)


# =========================================================
# Test 3: Autoencoder Model Exists
# =========================================================

run_test(

    "Autoencoder model exists",

    AUTOENCODER_MODEL_FILE.exists()
)


# =========================================================
# Test 4: Loss History Exists
# =========================================================

run_test(

    "Autoencoder loss history exists",

    LOSS_HISTORY_FILE.exists()
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
# Define Autoencoder Architecture
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
# Test Loss History
# =========================================================

try:

    loss_history_df = pd.read_csv(
        LOSS_HISTORY_FILE
    )

    run_test(

        "Loss history contains epochs",

        len(loss_history_df) > 0
    )

    run_test(

        "Loss column exists",

        "loss" in loss_history_df.columns
    )

    final_loss = loss_history_df[
        "loss"
    ].iloc[-1]

    run_test(

        "Final training loss is valid",

        final_loss >= 0
    )

except Exception:

    run_test(

        "Loss history validation",

        False
    )


# =========================================================
# Test Prediction Capability
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

            "Isolation Forest generates predictions",

            len(predictions) == 5
        )

except Exception:

    run_test(

        "Isolation Forest generates predictions",

        False
    )


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Training Test Summary ")
print("========================================")

print(f"\nTotal Tests  : {total_tests}")

print(f"Passed Tests : {passed_tests}")

print(f"Failed Tests : {failed_tests}")


# =========================================================
# Final Status
# =========================================================

if failed_tests == 0:

    print("\nAll training tests passed!")

else:

    print("\nSome training tests failed.")  # test_training.py
