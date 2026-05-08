# =========================================================
# File: utils/config.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import torch


# =========================================================
# Project Metadata
# =========================================================

PROJECT_NAME = "flight_anomaly_ai_core"

PROJECT_VERSION = "1.0.0"

PROJECT_DESCRIPTION = (
    "Flight telemetry anomaly detection platform "
    "using PyTorch and Isolation Forest."
)


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent


# =========================================================
# Core Directories
# =========================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

MODELS_DIR = PROJECT_ROOT / "models"

LOGS_DIR = PROJECT_ROOT / "logs"

VISUALIZATION_DIR = PROJECT_ROOT / "visualization"

APP_DIR = PROJECT_ROOT / "app"

TRAINING_DIR = PROJECT_ROOT / "training"

INFERENCE_DIR = PROJECT_ROOT / "inference"

FEATURE_ENGINEERING_DIR = (
    PROJECT_ROOT / "feature_engineering"
)


# =========================================================
# Dataset Files
# =========================================================

RAW_TELEMETRY_FILE = (
    DATASET_DIR
    / "flight_telemetry.csv"
)

PROCESSED_FEATURES_FILE = (
    DATASET_DIR
    / "processed_features.csv"
)

NORMALIZED_FEATURES_FILE = (
    DATASET_DIR
    / "normalized_features.csv"
)

ANOMALY_SCORES_FILE = (
    DATASET_DIR
    / "anomaly_scores.csv"
)

PREDICTED_ANOMALIES_FILE = (
    DATASET_DIR
    / "predicted_anomalies.csv"
)

REALTIME_LOG_FILE = (
    DATASET_DIR
    / "realtime_predictions.log"
)


# =========================================================
# Model Files
# =========================================================

ISOLATION_FOREST_MODEL_FILE = (
    MODELS_DIR
    / "isolation_forest.pkl"
)

AUTOENCODER_MODEL_FILE = (
    MODELS_DIR
    / "autoencoder.pth"
)

SCALER_MODEL_FILE = (
    MODELS_DIR
    / "scaler.pkl"
)


# =========================================================
# Visualization Files
# =========================================================

CONFUSION_MATRIX_PLOT = (
    VISUALIZATION_DIR
    / "confusion_matrix.png"
)

ANOMALY_DASHBOARD_PLOT = (
    VISUALIZATION_DIR
    / "anomaly_dashboard.png"
)


# =========================================================
# Training Configuration
# =========================================================

RANDOM_SEED = 42

TRAIN_TEST_SPLIT = 0.2

BATCH_SIZE = 64

LEARNING_RATE = 0.001

NUM_EPOCHS = 30

LATENT_DIM = 8

ANOMALY_THRESHOLD = 0.60


# =========================================================
# Isolation Forest Configuration
# =========================================================

N_ESTIMATORS = 200

CONTAMINATION = 0.05


# =========================================================
# Feature Engineering Configuration
# =========================================================

ROLLING_WINDOW = 20

ALTITUDE_DROP_THRESHOLD = -50

VELOCITY_SPIKE_THRESHOLD = 15

OSCILLATION_THRESHOLD = 5


# =========================================================
# Real-Time Streaming Configuration
# =========================================================

STREAM_DELAY = 0.1

REFRESH_INTERVAL = 2


# =========================================================
# Device Configuration
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Logging Configuration
# =========================================================

LOG_LEVEL = "INFO"

LOG_FILE = (
    LOGS_DIR
    / "flight_anomaly_ai_core.log"
)


# =========================================================
# Create Required Directories
# =========================================================

REQUIRED_DIRECTORIES = [

    DATASET_DIR,

    MODELS_DIR,

    LOGS_DIR,

    VISUALIZATION_DIR
]

for directory in REQUIRED_DIRECTORIES:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# Configuration Summary Function
# =========================================================

def print_config_summary():
    """
    Print project configuration summary.
    """

    print("\n========================================")
    print(" Flight Anomaly AI Core Configuration ")
    print("========================================")

    print(f"\nProject Name:")
    print(PROJECT_NAME)

    print(f"\nVersion:")
    print(PROJECT_VERSION)

    print(f"\nDevice:")
    print(DEVICE)

    print(f"\nDataset Directory:")
    print(DATASET_DIR)

    print(f"\nModels Directory:")
    print(MODELS_DIR)

    print(f"\nLogs Directory:")
    print(LOGS_DIR)

    print(f"\nBatch Size:")
    print(BATCH_SIZE)

    print(f"\nLearning Rate:")
    print(LEARNING_RATE)

    print(f"\nEpochs:")
    print(NUM_EPOCHS)

    print(f"\nAnomaly Threshold:")
    print(ANOMALY_THRESHOLD)


# =========================================================
# Example Usage
# =========================================================

if __name__ == "__main__":

    print_config_summary()  # config.py
