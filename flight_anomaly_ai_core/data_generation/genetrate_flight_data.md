# ✈️ Flight Telemetry Data Generator

The script "generate_flight_data.py" is responsible for generating synthetic flight telemetry data for the flight_anomaly_ai_core project. It simulates realistic aircraft telemetry signals such as altitude, velocity, pitch, roll, and yaw over time and saves the generated dataset as a CSV file for downstream anomaly detection pipelines.

## Overview

The `generate_flight_data.py` script generates synthetic aerospace telemetry data for the `flight_anomaly_ai_core` project. The script simulates realistic aircraft flight behavior by producing time-series telemetry signals such as altitude, velocity, pitch, roll, and yaw.

| Term | What it describes | Simple motion |
|---|---|---|
| `Altitude` | Height | Up/down position |
| `Velocity` | Speed and direction | How fast and where it moves |
| `Pitch` | Nose up/down rotation | Climb/dive attitude |
| `Roll` | Wing up/down rotation | Banking left/right |
| `Yaw` | Nose left/right rotation | Turning nose sideways |

The generated telemetry dataset acts as the foundation for:
- anomaly injection,
- feature engineering,
- machine learning training,
- real-time inference,
- and telemetry visualization pipelines.

This module helps simulate flight-test telemetry streams commonly used in aerospace analytics and anomaly detection systems.

---

# 📂 File Location

```text
flight_anomaly_ai_core/
│
├── data_generation/
│   └── generate_flight_data.py
│
├── dataset/
│   └── flight_telemetry.csv
```

---

# 🚀 Features

The script generates:
- synthetic flight telemetry,
- smooth time-series signals,
- realistic oscillatory aircraft behavior,
- and stochastic telemetry noise.

Generated telemetry parameters:
- Altitude
- Velocity
- Pitch
- Roll
- Yaw

---

# ⚙️ Configuration

The script uses the following configuration:

```python
NUM_SAMPLES = 5000
TIME_STEP = 1
RANDOM_SEED = 42
```

### Description

| Parameter | Description |
|---|---|
| `NUM_SAMPLES` | Total telemetry samples generated |
| `TIME_STEP` | Sampling interval in seconds |
| `RANDOM_SEED` | Ensures reproducible telemetry generation |

---

# 📊 Telemetry Signals

## Altitude

Simulates aircraft altitude variation with sinusoidal flight movement and random noise.

```python
altitude = (
    10000
    + 100 * np.sin(0.01 * timestamps)
    + np.random.normal(0, 5, NUM_SAMPLES)
)
```

---

## Velocity

Simulates velocity fluctuations during flight.

```python
velocity = (
    250
    + 5 * np.sin(0.02 * timestamps)
    + np.random.normal(0, 1, NUM_SAMPLES)
)
```

---

## Pitch

Simulates aircraft pitch oscillations.

```python
pitch = (
    2
    + 0.5 * np.sin(0.05 * timestamps)
    + np.random.normal(0, 0.2, NUM_SAMPLES)
)
```

---

## Roll

Simulates aircraft roll movement.

```python
roll = (
    1.5 * np.sin(0.03 * timestamps)
    + np.random.normal(0, 0.3, NUM_SAMPLES)
)
```

---

## Yaw

Simulates aircraft directional heading changes.

```python
yaw = (
    90
    + 2 * np.sin(0.01 * timestamps)
    + np.random.normal(0, 0.5, NUM_SAMPLES)
)
```

---

# 🧠 Telemetry Generation Workflow

```text
Generate Time Axis
        ↓
Simulate Altitude
        ↓
Simulate Velocity
        ↓
Simulate Pitch
        ↓
Simulate Roll
        ↓
Simulate Yaw
        ↓
Create DataFrame
        ↓
Save CSV Dataset
```

---

# 📁 Output Dataset

The generated telemetry dataset is saved as:

```text
dataset/flight_telemetry.csv
```

Example columns:

| Column | Description |
|---|---|
| `timestamp` | Time index |
| `altitude` | Aircraft altitude |
| `velocity` | Aircraft velocity |
| `pitch` | Pitch angle |
| `roll` | Roll angle |
| `yaw` | Yaw angle |

---

# ▶️ Run the Script

From project root:

```bash
python data_generation/generate_flight_data.py
```

---

# 📌 Example Console Output

```text
========================================
 Flight Telemetry Generated Successfully
========================================

Saved File:
dataset/flight_telemetry.csv

Dataset Shape:
(5000, 6)
```

---

# 🔬 Applications

This telemetry generator is useful for:
- anomaly detection research,
- aerospace ML pipelines,
- telemetry analytics,
- UAV simulation,
- real-time inference systems,
- and flight-test monitoring workflows.

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Time-Series Simulation

---

# 📈 Future Improvements

Possible future enhancements:
- turbulence simulation,
- engine telemetry,
- GPS coordinates,
- weather effects,
- turbulence injection,
- multi-aircraft telemetry,
- and real-time streaming support.

---

# 🏁 Summary

The `generate_flight_data.py` module creates realistic synthetic flight telemetry data for machine learning and aerospace analytics workflows. It serves as the foundation of the Flight Anomaly AI Core platform by providing structured telemetry streams for anomaly detection, feature engineering, model training, and real-time telemetry intelligence.