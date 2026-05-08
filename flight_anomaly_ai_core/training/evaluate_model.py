# =========================================================
# File: training/evaluate_model.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

VISUALIZATION_DIR = PROJECT_ROOT / "visualization"

INPUT_FILE = (
    DATASET_DIR
    / "isolation_forest_predictions.csv"
)

CONFUSION_MATRIX_PLOT = (
    VISUALIZATION_DIR
    / "confusion_matrix.png"
)

METRICS_REPORT_FILE = (
    DATASET_DIR
    / "model_evaluation_report.txt"
)


# =========================================================
# Create Visualization Directory
# =========================================================

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Validate Input File
# =========================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nPrediction file not found:\n{INPUT_FILE}"
    )


# =========================================================
# Load Prediction Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print(" Model Evaluation Started ")
print("========================================")

print(f"\nLoaded File:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Validate Required Columns
# =========================================================

required_columns = [
    "anomaly",
    "predicted_anomaly"
]

missing_columns = [

    column
    for column in required_columns

    if column not in flight_df.columns
]

if len(missing_columns) > 0:

    raise ValueError(
        f"\nMissing required columns:\n"
        f"{missing_columns}"
    )


# =========================================================
# Ground Truth and Predictions
# =========================================================

y_true = flight_df["anomaly"]

y_pred = flight_df["predicted_anomaly"]


# =========================================================
# Compute Metrics
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

conf_matrix = confusion_matrix(
    y_true,
    y_pred
)

class_report = classification_report(
    y_true,
    y_pred,
    zero_division=0
)


# =========================================================
# Print Metrics
# =========================================================

print("\n========================================")
print(" Evaluation Metrics ")
print("========================================")

print(f"\nAccuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")


# =========================================================
# Print Classification Report
# =========================================================

print("\nClassification Report:\n")

print(class_report)


# =========================================================
# Plot Confusion Matrix
# =========================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    conf_matrix,
    interpolation="nearest"
)

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = [0, 1]

plt.xticks(
    tick_marks,
    ["Normal", "Anomaly"]
)

plt.yticks(
    tick_marks,
    ["Normal", "Anomaly"]
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")


# Add values inside cells

for i in range(conf_matrix.shape[0]):

    for j in range(conf_matrix.shape[1]):

        plt.text(
            j,
            i,
            str(conf_matrix[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_PLOT)

plt.close()


# =========================================================
# Save Evaluation Report
# =========================================================

with open(
    METRICS_REPORT_FILE,
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        "========================================\n"
    )

    report_file.write(
        " Flight Anomaly Detection Evaluation\n"
    )

    report_file.write(
        "========================================\n\n"
    )

    report_file.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    report_file.write(
        f"Precision : {precision:.4f}\n"
    )

    report_file.write(
        f"Recall    : {recall:.4f}\n"
    )

    report_file.write(
        f"F1 Score  : {f1:.4f}\n\n"
    )

    report_file.write(
        "Classification Report:\n\n"
    )

    report_file.write(class_report)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Model Evaluation Completed ")
print("========================================")

print(f"\nConfusion Matrix Saved:")
print(CONFUSION_MATRIX_PLOT)

print(f"\nEvaluation Report Saved:")
print(METRICS_REPORT_FILE)  # evaluate_model.py
