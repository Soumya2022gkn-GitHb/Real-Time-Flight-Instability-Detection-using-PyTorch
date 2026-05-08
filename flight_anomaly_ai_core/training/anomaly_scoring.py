# =========================================================
# File: training/anomaly_scoring.py
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
# Configuration
# =========================================================

ANOMALY_THRESHOLD_PERCENTILE = 95

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

OUTPUT_FILE = (
    DATASET_DIR
    / "anomaly_scores.csv"
)


# =========================================================
# Validate Files
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
print(" Anomaly Scoring Started ")
print("========================================")

print(f"\nLoaded Dataset:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


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


# =========================================================
# Load Isolation Forest Model
# =========================================================

isolation_forest = joblib.load(
    ISOLATION_FOREST_MODEL_FILE
)

print("Loaded Isolation Forest Model")


# =========================================================
# Autoencoder Reconstruction Error
# =========================================================

tensor_X = torch.tensor(
    X
).to(DEVICE)

with torch.no_grad():

    reconstructed = autoencoder(
        tensor_X
    )

reconstruction_error = torch.mean(

    (tensor_X - reconstructed) ** 2,

    dim=1

).cpu().numpy()


# =========================================================
# Isolation Forest Scores
# =========================================================

isolation_scores = (
    -isolation_forest.decision_function(X)
)

# higher = more anomalous


# =========================================================
# Normalize Scores
# =========================================================

reconstruction_error_norm = (

    reconstruction_error
    - reconstruction_error.min()

) / (

    reconstruction_error.max()
    - reconstruction_error.min()
)

isolation_scores_norm = (

    isolation_scores
    - isolation_scores.min()

) / (

    isolation_scores.max()
    - isolation_scores.min()
)


# =========================================================
# Combined Anomaly Score
# =========================================================

combined_score = (

    0.5 * reconstruction_error_norm

    +

    0.5 * isolation_scores_norm
)

flight_df["reconstruction_error"] = (
    reconstruction_error
)

flight_df["isolation_forest_score"] = (
    isolation_scores
)

flight_df["combined_anomaly_score"] = (
    combined_score
)


# =========================================================
# Threshold-Based Detection
# =========================================================

threshold = np.percentile(

    combined_score,

    ANOMALY_THRESHOLD_PERCENTILE
)

flight_df["predicted_anomaly"] = (

    combined_score > threshold

).astype(int)


# =========================================================
# Save Results
# =========================================================

flight_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Anomaly Scoring Completed ")
print("========================================")

print(f"\nSaved File:")
print(OUTPUT_FILE)

print(f"\nAnomaly Threshold:")
print(threshold)

print(f"\nDetected Anomalies:")
print(
    flight_df["predicted_anomaly"]
    .sum()
)

print("\nGenerated Scoring Columns:")

generated_columns = [
    "reconstruction_error",
    "isolation_forest_score",
    "combined_anomaly_score",
    "predicted_anomaly"
]

for column in generated_columns:
    print(f"- {column}")

print("\nSample Scores:")

print(
    flight_df[
        [
            "timestamp",
            "combined_anomaly_score",
            "predicted_anomaly"
        ]
    ].head()
)  # anomaly_scoring.py
