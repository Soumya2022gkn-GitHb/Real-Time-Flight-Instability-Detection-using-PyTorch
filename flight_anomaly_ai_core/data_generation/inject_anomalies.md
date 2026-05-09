# 🚨 Flight Telemetry Anomaly Injection Module

The "inject_anomalies.py" script is responsible for injecting synthetic instability events and abnormal flight behavior into previously generated telemetry data. It modifies normal flight telemetry by introducing controlled anomalies such as altitude drops, roll instability, and pitch oscillations. These anomalies simulate real aerospace instability events and generate labeled datasets for anomaly detection model training and evaluation.

## Overview

The `inject_anomalies.py` script injects synthetic anomalies into flight telemetry data for the `flight_anomaly_ai_core` project. The module simulates unstable flight behavior and abnormal aerospace events by modifying telemetry signals such as altitude, roll, and pitch.

This module is a critical part of the anomaly detection pipeline because it creates labeled abnormal telemetry samples used for:
- anomaly detection model training,
- testing,
- evaluation,
- and real-time instability analysis.

The script transforms normal telemetry into realistic anomalous flight scenarios commonly encountered in aerospace telemetry systems.

---

# 📂 File Location

```text
flight_anomaly_ai_core/
│
├── data_generation/
│   └── inject_anomalies.py
│
├── dataset/
│   ├── flight_telemetry.csv
│   ├── flight_telemetry_with_anomalies.csv
│   └── anomaly_labels.csv
```

---

# 🚀 Features

The script injects multiple instability patterns into telemetry streams:

- Sudden altitude drops
- Roll instability spikes
- Pitch oscillations
- Anomaly labeling
- Synthetic instability simulation

---

# ⚙️ Configuration

```python
ALTITUDE_DROP_MAGNITUDE = 800
ROLL_SPIKE_MAGNITUDE = 35
OSCILLATION_AMPLITUDE = 8
```

### Description

| Parameter | Description |
|---|---|
| `ALTITUDE_DROP_MAGNITUDE` | Magnitude of altitude failure event |
| `ROLL_SPIKE_MAGNITUDE` | Roll instability spike |
| `OSCILLATION_AMPLITUDE` | Pitch oscillation strength |

---

# 📊 Input Dataset

The script loads telemetry generated from:

```text
dataset/flight_telemetry.csv
```

Telemetry columns:
- timestamp
- altitude
- velocity
- pitch
- roll
- yaw

---

# 🧠 Injected Anomalies

## 1. Altitude Drop Anomaly

Simulates sudden altitude loss.

```python
flight_df.loc[
    altitude_start:altitude_end,
    "altitude"
] -= ALTITUDE_DROP_MAGNITUDE
```

### Example

```text
Normal Altitude
      ↓
Sudden Drop
      ↓
Recovery
```

Applications:
- engine failure simulation,
- turbulence events,
- flight instability.

---

## 2. Roll Instability

Simulates unstable aircraft roll behavior.

```python
flight_df.loc[
    roll_start:roll_end,
    "roll"
] += ROLL_SPIKE_MAGNITUDE
```

Applications:
- lateral instability,
- turbulence,
- aggressive maneuver simulation.

---

## 3. Pitch Oscillation

Simulates oscillatory instability.

```python
oscillation_signal = (
    OSCILLATION_AMPLITUDE
    * np.sin(0.5 * oscillation_range)
)
```

Applications:
- oscillatory flight behavior,
- control instability,
- unstable pitch correction.

---

# 🏷️ Anomaly Labels

The script creates anomaly labels:

```python
flight_df["anomaly"] = 0
```

Where:
- `0` → normal telemetry
- `1` → anomaly

Generated labels are saved separately:

```text
dataset/anomaly_labels.csv
```

---

# 🧠 Anomaly Injection Workflow

```text
Load Telemetry Dataset
        ↓
Initialize Labels
        ↓
Inject Altitude Drop
        ↓
Inject Roll Instability
        ↓
Inject Pitch Oscillation
        ↓
Generate Anomaly Labels
        ↓
Save Updated Dataset
```

---

# 📁 Output Files

## Updated Telemetry Dataset

```text
dataset/flight_telemetry_with_anomalies.csv
```

Contains:
- telemetry signals,
- injected anomalies,
- anomaly labels.

---

## Anomaly Labels

```text
dataset/anomaly_labels.csv
```

Contains:
- timestamp
- anomaly label

---

# ▶️ Run the Script

From project root:

```bash
python data_generation/inject_anomalies.py
```

---

# 📌 Example Console Output

```text
Loaded Flight Telemetry Data
Shape: (5000, 6)

Injected Altitude Drop Anomaly
Injected Roll Instability
Injected Pitch Oscillation

========================================
 Anomaly Injection Completed
========================================
```

---

# 📈 Example Instability Events

## Altitude Drop

```text
Altitude
  ^
  |
  |        ________
  |       /
  |______/
  |
  +------------------> Time
```

---

## Roll Spike

```text
Roll
 ^
 |
 |        /\
 |_______/  \_______
 |
 +------------------> Time
```

---

## Pitch Oscillation

```text
Pitch
 ^
 |
 |   ~~~~ ~~~~ ~~~~
 |
 +------------------> Time
```

---

# 🔬 Applications

This module is useful for:
- aerospace anomaly detection,
- ML model training,
- telemetry simulation,
- UAV instability analysis,
- flight-test analytics,
- and real-time telemetry intelligence research.

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Time-Series Simulation

---

# 📌 Future Improvements

Possible future enhancements:
- turbulence injection,
- GPS drift simulation,
- engine failure anomalies,
- weather-based instability,
- sensor noise corruption,
- multi-aircraft telemetry,
- and streaming anomaly generation.

---

# 🏁 Summary

The `inject_anomalies.py` module introduces realistic instability events into flight telemetry streams and generates labeled anomaly datasets for machine learning workflows. It provides the foundation for training, evaluating, and testing anomaly detection models within the Flight Anomaly AI Core platform.