# =========================================================
# File: utils/helpers.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

LOGS_DIR = PROJECT_ROOT / "logs"

MODELS_DIR = PROJECT_ROOT / "models"


# =========================================================
# Directory Utilities
# =========================================================

def create_project_directories():
    """
    Create required project directories.
    """

    required_directories = [

        DATASET_DIR,

        LOGS_DIR,

        MODELS_DIR
    ]

    for directory in required_directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    print("\nProject directories verified.")


# =========================================================
# Dataset Utilities
# =========================================================

def load_csv_dataset(file_path):
    """
    Load CSV dataset safely.
    """

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{file_path}"
        )

    dataset = pd.read_csv(file_path)

    print(f"\nLoaded dataset: {file_path}")

    print(f"Dataset shape: {dataset.shape}")

    return dataset


def save_csv_dataset(dataframe, file_path):
    """
    Save DataFrame as CSV.
    """

    file_path = Path(file_path)

    dataframe.to_csv(
        file_path,
        index=False
    )

    print(f"\nDataset saved: {file_path}")


# =========================================================
# Telemetry Utilities
# =========================================================

def compute_signal_statistics(signal):
    """
    Compute telemetry signal statistics.
    """

    statistics = {

        "mean":
            np.mean(signal),

        "std":
            np.std(signal),

        "min":
            np.min(signal),

        "max":
            np.max(signal),

        "median":
            np.median(signal)
    }

    return statistics


def detect_threshold_anomalies(
    signal,
    threshold
):
    """
    Detect anomalies using threshold.
    """

    anomaly_flags = np.where(

        np.abs(signal) > threshold,

        1,

        0
    )

    return anomaly_flags


# =========================================================
# Timestamp Utilities
# =========================================================

def get_current_timestamp():
    """
    Get formatted current timestamp.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# JSON Utilities
# =========================================================

def save_json(data, file_path):
    """
    Save dictionary or list as JSON.
    """

    file_path = Path(file_path)

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print(f"\nJSON saved: {file_path}")


def load_json(file_path):
    """
    Load JSON file safely.
    """

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nJSON file not found:\n{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


# =========================================================
# Feature Engineering Utilities
# =========================================================

def normalize_signal(signal):
    """
    Normalize telemetry signal.
    """

    signal = np.array(signal)

    normalized_signal = (

        signal - signal.mean()

    ) / (

        signal.std() + 1e-8
    )

    return normalized_signal


def compute_rolling_mean(
    signal,
    window_size=10
):
    """
    Compute rolling mean.
    """

    return pd.Series(signal).rolling(
        window=window_size
    ).mean()


def compute_rolling_std(
    signal,
    window_size=10
):
    """
    Compute rolling standard deviation.
    """

    return pd.Series(signal).rolling(
        window=window_size
    ).std()


# =========================================================
# Anomaly Utilities
# =========================================================

def compute_anomaly_ratio(anomaly_labels):
    """
    Compute anomaly percentage.
    """

    anomaly_labels = np.array(
        anomaly_labels
    )

    anomaly_count = np.sum(
        anomaly_labels == 1
    )

    total_samples = len(anomaly_labels)

    anomaly_ratio = (

        anomaly_count / total_samples

    ) * 100

    return round(anomaly_ratio, 2)


# =========================================================
# Console Banner Utility
# =========================================================

def print_banner(title):
    """
    Print formatted console banner.
    """

    print("\n========================================")

    print(f" {title} ")

    print("========================================")


# =========================================================
# Example Usage
# =========================================================

if __name__ == "__main__":

    print_banner(
        "Helpers Utility Test"
    )

    create_project_directories()

    sample_signal = np.random.normal(
        0,
        1,
        100
    )

    stats = compute_signal_statistics(
        sample_signal
    )

    print("\nSignal Statistics:")

    print(stats)

    anomaly_ratio = compute_anomaly_ratio(

        np.random.randint(
            0,
            2,
            100
        )
    )

    print(f"\nAnomaly Ratio: {anomaly_ratio}%")

    print(
        f"\nCurrent Timestamp: "
        f"{get_current_timestamp()}"
    )  # helpers.py
