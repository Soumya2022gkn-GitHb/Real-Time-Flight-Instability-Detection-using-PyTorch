# =========================================================
# File: tests/test_feature_engineering.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

INPUT_FILE = (
    DATASET_DIR
    / "processed_features.csv"
)


# =========================================================
# Validate Dataset File
# =========================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nProcessed features dataset not found:\n"
        f"{INPUT_FILE}"
    )


# =========================================================
# Load Dataset
# =========================================================

flight_df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print(" Feature Engineering Tests Started ")
print("========================================")

print(f"\nLoaded Dataset:")
print(INPUT_FILE)

print(f"\nDataset Shape:")
print(flight_df.shape)


# =========================================================
# Test Result Tracking
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
# Test 1: Dataset Not Empty
# =========================================================

run_test(

    "Dataset is not empty",

    len(flight_df) > 0
)


# =========================================================
# Test 2: Required Columns Exist
# =========================================================

required_columns = [

    "timestamp",

    "altitude",

    "velocity",

    "pitch",

    "roll",

    "yaw"
]

for column in required_columns:

    run_test(

        f"Column exists: {column}",

        column in flight_df.columns
    )


# =========================================================
# Test 3: Rolling Features Exist
# =========================================================

rolling_features = [

    "altitude_rolling_mean",

    "velocity_rolling_std"
]

for feature in rolling_features:

    run_test(

        f"Rolling feature exists: {feature}",

        feature in flight_df.columns
    )


# =========================================================
# Test 4: Altitude Features Exist
# =========================================================

altitude_features = [

    "altitude_rate",

    "altitude_acceleration",

    "sudden_altitude_drop"
]

for feature in altitude_features:

    run_test(

        f"Altitude feature exists: {feature}",

        feature in flight_df.columns
    )


# =========================================================
# Test 5: Velocity Features Exist
# =========================================================

velocity_features = [

    "velocity_rate",

    "velocity_acceleration",

    "velocity_spike"
]

for feature in velocity_features:

    run_test(

        f"Velocity feature exists: {feature}",

        feature in flight_df.columns
    )


# =========================================================
# Test 6: Oscillation Features Exist
# =========================================================

oscillation_features = [

    "pitch_oscillation_strength",

    "roll_oscillation_strength",

    "oscillation_alert"
]

for feature in oscillation_features:

    run_test(

        f"Oscillation feature exists: {feature}",

        feature in flight_df.columns
    )


# =========================================================
# Test 7: No Missing Values
# =========================================================

run_test(

    "Dataset has no missing values",

    not flight_df.isnull().values.any()
)


# =========================================================
# Test 8: Numerical Columns Valid
# =========================================================

numerical_columns = flight_df.select_dtypes(

    include=np.number
).columns

run_test(

    "Numerical columns detected",

    len(numerical_columns) > 0
)


# =========================================================
# Test 9: Dataset Contains Anomalies
# =========================================================

if "anomaly" in flight_df.columns:

    anomaly_count = flight_df["anomaly"].sum()

    run_test(

        "Anomaly labels exist",

        anomaly_count > 0
    )


# =========================================================
# Test 10: Timestamp Monotonic
# =========================================================

run_test(

    "Timestamp is increasing",

    flight_df["timestamp"].is_monotonic_increasing
)


# =========================================================
# Console Summary
# =========================================================

print("\n========================================")
print(" Feature Engineering Test Summary ")
print("========================================")

print(f"\nTotal Tests  : {total_tests}")

print(f"Passed Tests : {passed_tests}")

print(f"Failed Tests : {failed_tests}")


# =========================================================
# Final Status
# =========================================================

if failed_tests == 0:

    print("\nAll feature engineering tests passed!")

else:

    # test_feature_engineering.py
    print("\nSome tests failed. Review feature pipeline.")
