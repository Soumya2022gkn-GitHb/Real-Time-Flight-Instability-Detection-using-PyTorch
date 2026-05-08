# =========================================================
# File: inference/telemetry_monitor.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import time

import pandas as pd


# =========================================================
# Configuration
# =========================================================

REFRESH_INTERVAL = 2  # seconds

MAX_RECENT_ALERTS = 10


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

LOG_FILE = (
    DATASET_DIR
    / "realtime_predictions.log"
)


# =========================================================
# Validate Log File
# =========================================================

if not LOG_FILE.exists():

    raise FileNotFoundError(
        f"\nPrediction log file not found:\n{LOG_FILE}"
    )


# =========================================================
# Console Header
# =========================================================

print("\n========================================")
print(" Flight Telemetry Monitor Started ")
print("========================================")

print(f"\nMonitoring Log File:")
print(LOG_FILE)

print(f"\nRefresh Interval:")
print(f"{REFRESH_INTERVAL} seconds")


# =========================================================
# Helper Function
# =========================================================

def load_prediction_log():
    """
    Load prediction log file safely.
    """

    try:

        records = []

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                try:

                    record = pd.read_json(
                        line,
                        typ="series"
                    )

                    records.append(record)

                except Exception:

                    continue

        if len(records) == 0:

            return pd.DataFrame()

        return pd.DataFrame(records)

    except Exception as error:

        print(
            f"\nError reading log file:\n{error}"
        )

        return pd.DataFrame()


# =========================================================
# Monitoring Loop
# =========================================================

try:

    while True:

        # -------------------------------------------------
        # Clear Console
        # -------------------------------------------------

        print("\n" * 3)

        print("========================================")
        print(" Real-Time Flight Telemetry Monitor ")
        print("========================================")

        # -------------------------------------------------
        # Load Predictions
        # -------------------------------------------------

        prediction_df = load_prediction_log()

        # -------------------------------------------------
        # Validate Data
        # -------------------------------------------------

        if prediction_df.empty:

            print("\nWaiting for telemetry data...")

            time.sleep(REFRESH_INTERVAL)

            continue

        # -------------------------------------------------
        # Total Statistics
        # -------------------------------------------------

        total_samples = len(prediction_df)

        anomaly_count = len(

            prediction_df[
                prediction_df["prediction"]
                == "anomaly"
            ]
        )

        normal_count = len(

            prediction_df[
                prediction_df["prediction"]
                == "normal"
            ]
        )

        # -------------------------------------------------
        # Print Statistics
        # -------------------------------------------------

        print(f"\nTotal Telemetry Samples : {total_samples}")

        print(f"Normal Predictions      : {normal_count}")

        print(f"Anomaly Predictions     : {anomaly_count}")

        # -------------------------------------------------
        # Recent Alerts
        # -------------------------------------------------

        anomaly_df = prediction_df[

            prediction_df["prediction"]
            == "anomaly"
        ]

        if len(anomaly_df) > 0:

            print("\n===================================")
            print(" Recent Anomaly Alerts ")
            print("===================================")

            recent_alerts = anomaly_df.tail(
                MAX_RECENT_ALERTS
            )

            print(

                recent_alerts[
                    [
                        "timestamp",
                        "combined_score",
                        "prediction"
                    ]
                ]
            )

        else:

            print("\nNo anomalies detected.")

        # -------------------------------------------------
        # Highest Risk Event
        # -------------------------------------------------

        highest_risk_event = prediction_df.loc[

            prediction_df[
                "combined_score"
            ].idxmax()
        ]

        print("\n===================================")
        print(" Highest Risk Event ")
        print("===================================")

        print(
            f"Timestamp      : "
            f"{highest_risk_event['timestamp']}"
        )

        print(
            f"Anomaly Score  : "
            f"{highest_risk_event['combined_score']:.4f}"
        )

        print(
            f"Prediction     : "
            f"{highest_risk_event['prediction']}"
        )

        # -------------------------------------------------
        # Wait Before Refresh
        # -------------------------------------------------

        time.sleep(REFRESH_INTERVAL)


# =========================================================
# Graceful Shutdown
# =========================================================

except KeyboardInterrupt:

    print("\n========================================")
    print(" Telemetry Monitor Stopped ")
    print("========================================")  # telemetry_monitor.py
