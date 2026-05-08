import os

# Root project folder
root = "flight_anomaly_ai_core"

# Folder and file structure
project_structure = {
    "data_generation": [
        "generate_flight_data.py",
        "inject_anomalies.py",
        "simulate_instability.py",
        "generate_telemetry_stream.py"
    ],

    "dataset": [
        "flight_telemetry.csv",
        "processed_features.csv",
        "anomaly_labels.csv"
    ],

    "feature_engineering": [
        "rolling_statistics.py",
        "altitude_features.py",
        "velocity_features.py",
        "oscillation_detection.py",
        "normalize_features.py"
    ],

    "training": [
        "train_isolation_forest.py",
        "train_autoencoder.py",
        "evaluate_model.py",
        "anomaly_scoring.py"
    ],

    "models": [
        "isolation_forest.pkl",
        "autoencoder.pth",
        "scaler.pkl"
    ],

    "inference": [
        "predict_anomalies.py",
        "realtime_inference.py",
        "telemetry_monitor.py"
    ],

    "visualization": [
        "altitude_anomalies.png",
        "velocity_spikes.png",
        "oscillation_detection.png",
        "anomaly_dashboard.png"
    ],

    "app": [
        "app.py",
        "dashboard.py",
        "telemetry_viewer.py"
    ],

    "utils": [
        "logger.py",
        "helpers.py",
        "config.py"
    ],

    "tests": [
        "test_feature_engineering.py",
        "test_training.py",
        "test_inference.py"
    ]
}

# Root-level files
root_files = [
    "requirements.txt",
    "README.md",
    ".gitignore",
    "main.py"
]

# Create root folder
os.makedirs(root, exist_ok=True)

# Create folders and files
for folder, files in project_structure.items():

    folder_path = os.path.join(root, folder)

    os.makedirs(folder_path, exist_ok=True)

    for file_name in files:

        file_path = os.path.join(folder_path, file_name)

        # Create empty file
        with open(file_path, "w") as f:

            # Optional starter comments
            if file_name.endswith(".py"):
                f.write(f"# {file_name}\n")

# Create root files
for file_name in root_files:

    file_path = os.path.join(root, file_name)

    with open(file_path, "w") as f:

        if file_name == "README.md":
            f.write("# Flight Anomaly AI Core\n")

        elif file_name == "requirements.txt":
            f.write(
                "numpy\n"
                "pandas\n"
                "scikit-learn\n"
                "matplotlib\n"
                "torch\n"
            )

        elif file_name == ".gitignore":
            f.write(
                "__pycache__/\n"
                "*.pyc\n"
                "venv/\n"
                ".DS_Store\n"
            )

        elif file_name == "main.py":
            f.write("# Main entry point\n")

print(f"\nProject structure '{root}' created successfully!")
