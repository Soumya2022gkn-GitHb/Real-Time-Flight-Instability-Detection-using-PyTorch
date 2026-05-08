# =========================================================
# File: data_generation/generate_telemetry_stream.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import time
import json

import pandas as pd


# =========================================================
# Configuration
# =========================================================

STREAM_DELAY = 0.1  # seconds

ENABLE_JSON_LOGGING = True

LOOP_STREAM = False

MAX_STREAM_SAMPLES = None
# Example:
# MAX_STREAM_SAMPLES = 1000


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_FILE = (
    DATASET_DIR
    / "flight_instability_simulation.csv"
)

STREAM_OUTPUT_FILE = (
    DATASET_DIR
    / "telemetry_stream.log"
)


# =========================================================
# Validate Input File
# =========================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )


# =========================================================
# Load Telemetry Dataset
# =========================================================

telemetry_df = pd.read_csv(INPUT_FILE)

# Optional sample limit

if MAX_STREAM_SAMPLES is not None:

    telemetry_df = telemetry_df.head(
        MAX_STREAM_SAMPLES
    )


# =========================================================
# Validate Required Columns
# =========================================================

required_columns = [
    "timestamp",
    "altitude",
    "velocity",
    "pitch",
    "roll",
    "yaw",
    "anomaly"
]

missing_columns = [
    column
    for column in required_columns
    if column not in telemetry_df.columns
]

if len(missing_columns) > 0:

    raise ValueError(
        f"\nMissing required columns:\n"
        f"{missing_columns}"
    )


# =========================================================
# Console Startup Info
# =========================================================

print("\n========================================")
print(" Flight Telemetry Streaming Started ")
print("========================================")

print(f"\nInput Dataset:")
print(INPUT_FILE)

print(f"\nTotal Samples:")
print(len(telemetry_df))

print(f"\nStreaming Delay:")
print(f"{STREAM_DELAY} seconds")


# =========================================================
# Initialize Stream Log File
# =========================================================

if ENABLE_JSON_LOGGING:

    with open(
        STREAM_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("")


# =========================================================
# Streaming Function
# =========================================================

def stream_telemetry():
    """
    Stream telemetry packets one-by-one.
    """

    for index, row in telemetry_df.iterrows():

        # -------------------------------------------------
        # Build Telemetry Packet
        # -------------------------------------------------

        telemetry_packet = {

            "timestamp":
                int(row["timestamp"]),

            "altitude":
                round(row["altitude"], 2),

            "velocity":
                round(row["velocity"], 2),

            "pitch":
                round(row["pitch"], 2),

            "roll":
                round(row["roll"], 2),

            "yaw":
                round(row["yaw"], 2),

            "anomaly":
                int(row["anomaly"])
        }

        # -------------------------------------------------
        # Print Packet
        # -------------------------------------------------

        print(
            json.dumps(
                telemetry_packet,
                indent=2
            )
        )

        # -------------------------------------------------
        # Save Packet to Log File
        # -------------------------------------------------

        if ENABLE_JSON_LOGGING:

            with open(
                STREAM_OUTPUT_FILE,
                "a",
                encoding="utf-8"
            ) as file:

                json.dump(
                    telemetry_packet,
                    file
                )

                file.write("\n")

        # -------------------------------------------------
        # Highlight Anomaly Events
        # -------------------------------------------------

        if telemetry_packet["anomaly"] == 1:

            print("\n===================================")
            print(" ANOMALY DETECTED ")
            print("===================================")

            print(
                f"Timestamp : "
                f"{telemetry_packet['timestamp']}"
            )

            print(
                f"Altitude  : "
                f"{telemetry_packet['altitude']}"
            )

            print(
                f"Velocity  : "
                f"{telemetry_packet['velocity']}"
            )

            print(
                f"Pitch     : "
                f"{telemetry_packet['pitch']}"
            )

            print(
                f"Roll      : "
                f"{telemetry_packet['roll']}"
            )

        # -------------------------------------------------
        # Real-Time Delay
        # -------------------------------------------------

        time.sleep(STREAM_DELAY)


# =========================================================
# Start Streaming
# =========================================================

try:

    if LOOP_STREAM:

        while True:

            stream_telemetry()

    else:

        stream_telemetry()

except KeyboardInterrupt:

    print("\n\nStreaming Interrupted by User.")


# =========================================================
# Stream Complete
# =========================================================

print("\n========================================")
print(" Flight Telemetry Streaming Completed ")
print("========================================")

print(f"\nTelemetry Log Saved:")
print(STREAM_OUTPUT_FILE)
