# =========================================================
# File: main.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import subprocess
import sys
import time


# =========================================================
# Define Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_GENERATION_DIR = (
    PROJECT_ROOT / "data_generation"
)

FEATURE_ENGINEERING_DIR = (
    PROJECT_ROOT / "feature_engineering"
)

TRAINING_DIR = (
    PROJECT_ROOT / "training"
)

INFERENCE_DIR = (
    PROJECT_ROOT / "inference"
)

TESTS_DIR = (
    PROJECT_ROOT / "tests"
)


# =========================================================
# Helper Function
# =========================================================

def run_script(script_path):
    """
    Execute Python script safely.
    """

    print("\n========================================")
    print(f" Running: {script_path.name} ")
    print("========================================")

    try:

        result = subprocess.run(

            [sys.executable, str(script_path)],

            check=True
        )

        print(
            f"\nCompleted: {script_path.name}"
        )

        return result.returncode

    except subprocess.CalledProcessError as error:

        print(
            f"\nError while running:\n"
            f"{script_path.name}"
        )

        print(error)

        sys.exit(1)


# =========================================================
# Pipeline Execution
# =========================================================

def run_pipeline():
    """
    Execute complete telemetry anomaly pipeline.
    """

    print("\n========================================")
    print(" Flight Anomaly AI Core Pipeline ")
    print("========================================")

    start_time = time.time()

    # =====================================================
    # Step 1: Generate Telemetry Data
    # =====================================================

    run_script(

        DATA_GENERATION_DIR
        / "generate_flight_data.py"
    )

    run_script(

        DATA_GENERATION_DIR
        / "inject_anomalies.py"
    )

    run_script(

        DATA_GENERATION_DIR
        / "simulate_instability.py"
    )

    # =====================================================
    # Step 2: Feature Engineering
    # =====================================================

    run_script(

        FEATURE_ENGINEERING_DIR
        / "rolling_statistics.py"
    )

    run_script(

        FEATURE_ENGINEERING_DIR
        / "altitude_features.py"
    )

    run_script(

        FEATURE_ENGINEERING_DIR
        / "velocity_features.py"
    )

    run_script(

        FEATURE_ENGINEERING_DIR
        / "oscillation_detection.py"
    )

    run_script(

        FEATURE_ENGINEERING_DIR
        / "normalize_features.py"
    )

    # =====================================================
    # Step 3: Train Models
    # =====================================================

    run_script(

        TRAINING_DIR
        / "train_isolation_forest.py"
    )

    run_script(

        TRAINING_DIR
        / "train_autoencoder.py"
    )

    run_script(

        TRAINING_DIR
        / "evaluate_model.py"
    )

    run_script(

        TRAINING_DIR
        / "anomaly_scoring.py"
    )

    # =====================================================
    # Step 4: Inference
    # =====================================================

    run_script(

        INFERENCE_DIR
        / "predict_anomalies.py"
    )

    # =====================================================
    # Step 5: Run Tests
    # =====================================================

    run_script(

        TESTS_DIR
        / "test_feature_engineering.py"
    )

    run_script(

        TESTS_DIR
        / "test_training.py"
    )

    run_script(

        TESTS_DIR
        / "test_inference.py"
    )

    # =====================================================
    # Pipeline Summary
    # =====================================================

    total_runtime = (
        time.time() - start_time
    )

    print("\n========================================")
    print(" Pipeline Execution Completed ")
    print("========================================")

    print(
        f"\nTotal Runtime: "
        f"{round(total_runtime, 2)} seconds"
    )

    print("\nGenerated Outputs:")

    outputs = [

        "dataset/flight_telemetry.csv",

        "dataset/processed_features.csv",

        "dataset/normalized_features.csv",

        "dataset/predicted_anomalies.csv",

        "models/isolation_forest.pkl",

        "models/autoencoder.pth",

        "visualization/confusion_matrix.png"
    ]

    for output in outputs:

        print(f"- {output}")

    print("\nFlight anomaly pipeline completed successfully!")


# =========================================================
# Main Entry Point
# =========================================================

if __name__ == "__main__":

    run_pipeline()  # Main entry point
