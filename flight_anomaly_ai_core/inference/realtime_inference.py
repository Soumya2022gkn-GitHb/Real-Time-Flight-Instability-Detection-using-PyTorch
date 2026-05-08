# =========================================================
# File: inference/realtime_inference.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import time
import json
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
# Configuration
# =========================================================

STREAM_DELAY = 0.1

ANOMALY_THRESHOLD = 0.60

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

MODELS_DIR = PROJECT_ROOT / "models"

INPUT_FILE = (
    DATASET_DIR
    / "normalized_features.csv"
)

AUTOENCODER_MODEL_FILE = (
    MODELS_DIR
    / "autoencoder.pth"
)

ISOLATION_FOREST_MODEL_FILE = (
    MODELS_DIR
    / "isolation_forest.pkl"
)

OUTPUT_LOG_FILE = (
    DATASET_DIR
    / "realtime_predictions.log"
)


# =========================================================
# Validate Required Files
# =========================================================

required_files = [
    INPUT_FILE,
    AUTOENCODER_MODEL_FILE,
    ISOLATION_FOREST_MODEL_FILE
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nMissing required file:\n{file_path}"
        )


# =========================================================
# Load Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print(" Real-Time Flight Inference Started ")
print("========================================")

print(f"\nInput Dataset:")
print(INPUT_FILE)

print(f"\nTotal Samples:")
print(len(flight_df))


# =========================================================
# Select Feature Columns
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

X = flight_df[
    feature_columns
].values.astype(np.float32)

INPUT_DIM = X.shape[1]


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
# Load Models
# =========================================================

autoencoder = Autoencoder(
    input_dim=INPUT_DIM
).to(DEVICE)

autoencoder.load_state_dict(

    torch.load(
        AUTOENCODER_MODEL_FILE,
        map_location=DEVICE
    )
)

autoencoder.eval()

print("\nLoaded Autoencoder Model")


isolation_forest = joblib.load(
    ISOLATION_FOREST_MODEL_FILE
)

print("Loaded Isolation Forest Model")


# =========================================================
# Initialize Log File
# =========================================================

with open(
    OUTPUT_LOG_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("")


# =========================================================
# Start Real-Time Inference
# =========================================================

print("\nStreaming Real-Time Predictions...\n")


for index, row in flight_df.iterrows():

    # -----------------------------------------------------
    # Extract Single Sample
    # -----------------------------------------------------

    sample = row[
        feature_columns
    ].values.astype(np.float32)

    sample_tensor = torch.tensor(
        sample
    ).unsqueeze(0).to(DEVICE)

    # -----------------------------------------------------
    # Autoencoder Reconstruction Error
    # -----------------------------------------------------

    with torch.no_grad():

        reconstructed = autoencoder(
            sample_tensor
        )

    reconstruction_error = torch.mean(

        (sample_tensor - reconstructed) ** 2

    ).item()

    # -----------------------------------------------------
    # Isolation Forest Score
    # -----------------------------------------------------

    isolation_score = -isolation_forest.decision_function(
        sample.reshape(1, -1)
    )[0]

    # -----------------------------------------------------
    # Normalize Scores
    # -----------------------------------------------------

    reconstruction_score = min(
        reconstruction_error * 10,
        1.0
    )

    isolation_score_norm = min(
        isolation_score,
        1.0
    )

    # -----------------------------------------------------
    # Combined Score
    # -----------------------------------------------------

    combined_score = (

        0.5 * reconstruction_score

        +

        0.5 * isolation_score_norm
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predicted_anomaly = int(
        combined_score > ANOMALY_THRESHOLD
    )

    prediction_label = (
        "anomaly"
        if predicted_anomaly == 1
        else "normal"
    )

    # -----------------------------------------------------
    # Create Prediction Packet
    # -----------------------------------------------------

    prediction_packet = {

        "timestamp":
            int(row["timestamp"]),

        "combined_score":
            round(combined_score, 4),

        "prediction":
            prediction_label
    }

    # -----------------------------------------------------
    # Print Prediction
    # -----------------------------------------------------

    print(
        json.dumps(
            prediction_packet,
            indent=2
        )
    )

    # -----------------------------------------------------
    # Log Predictions
    # -----------------------------------------------------

    with open(
        OUTPUT_LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        json.dump(
            prediction_packet,
            file
        )

        file.write("\n")

    # -----------------------------------------------------
    # Alert for Anomalies
    # -----------------------------------------------------

    if predicted_anomaly == 1:

        print("\n===================================")
        print(" REAL-TIME ANOMALY DETECTED ")
        print("===================================")

        print(
            f"Timestamp : "
            f"{prediction_packet['timestamp']}"
        )

        print(
            f"Anomaly Score : "
            f"{prediction_packet['combined_score']}"
        )

    # -----------------------------------------------------
    # Simulate Streaming Delay
    # -----------------------------------------------------

    time.sleep(STREAM_DELAY)


# =========================================================
# Inference Complete
# =========================================================

print("\n========================================")
print(" Real-Time Inference Completed ")
print("========================================")

print(f"\nPrediction Log Saved:")
print(OUTPUT_LOG_FILE)  # realtime_inference.py
