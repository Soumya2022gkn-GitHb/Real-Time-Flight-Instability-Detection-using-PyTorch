The simulate_instability.py script simulates complex flight instability behavior on top of previously injected telemetry anomalies. It creates realistic unstable flight conditions by modifying telemetry signals such as pitch, roll, velocity, and altitude over a defined instability window. The goal is to generate advanced instability patterns that resemble real aerospace flight disturbances for anomaly detection training and real-time telemetry analysis.

# ✈️ Flight Instability Simulation Module

## Overview

The `simulate_instability.py` script simulates advanced flight instability events within telemetry data for the `flight_anomaly_ai_core` project. The module generates realistic unstable flight behavior by injecting oscillatory motion, velocity fluctuations, altitude disturbances, and control instability into aircraft telemetry streams.

This module extends the anomaly injection pipeline by creating more complex aerospace instability scenarios commonly encountered in:
- flight testing,
- UAV systems,
- turbulence analysis,
- telemetry intelligence,
- and aerospace anomaly detection systems.

The generated instability telemetry is later used for:
- machine learning training,
- anomaly detection,
- real-time inference,
- and instability visualization.

---

# 📂 File Location

```text
flight_anomaly_ai_core/
│
├── data_generation/
│   └── simulate_instability.py
│
├── dataset/
│   ├── flight_telemetry_with_anomalies.csv
│   └── flight_instability_simulation.csv
```

---

# 🚀 Features

The script simulates:
- pitch oscillation instability,
- roll oscillation instability,
- velocity fluctuations,
- altitude disturbances,
- anomaly labeling,
- and instability classification.

---

# ⚙️ Configuration

```python
OSCILLATION_AMPLITUDE = 12
ROLL_INSTABILITY_MAGNITUDE = 25
VELOCITY_FLUCTUATION = 20
INSTABILITY_DURATION = 120
```

### Description

| Parameter | Description |
|---|---|
| `OSCILLATION_AMPLITUDE` | Pitch oscillation strength |
| `ROLL_INSTABILITY_MAGNITUDE` | Roll instability magnitude |
| `VELOCITY_FLUCTUATION` | Random velocity disturbance |
| `INSTABILITY_DURATION` | Number of unstable telemetry samples |

---

# 📊 Input Dataset

The script loads telemetry containing previous anomalies:

```text
dataset/flight_telemetry_with_anomalies.csv
```

The dataset includes:
- altitude,
- velocity,
- pitch,
- roll,
- yaw,
- and anomaly labels.

---

# 🧠 Simulated Instability Types

## 1. Pitch Oscillation

Simulates unstable oscillatory pitch motion.

```python
pitch_instability = (
    OSCILLATION_AMPLITUDE
    * np.sin(0.4 * time_window)
)
```

### Example

```text
Pitch
 ^
 |
 |   ~~~~ ~~~~ ~~~~
 |
 +------------------> Time
```

Applications:
- flight instability,
- turbulence response,
- unstable control systems.

---

## 2. Roll Oscillation

Simulates unstable lateral aircraft motion.

```python
roll_instability = (
    ROLL_INSTABILITY_MAGNITUDE
    * np.sin(0.5 * time_window)
)
```

### Example

```text
Roll
 ^
 |
 |    /\/\/\/\/\
 |
 +------------------> Time
```

Applications:
- lateral instability,
- aggressive maneuver simulation,
- unstable roll correction.

---

## 3. Velocity Fluctuation

Adds random velocity disturbances.

```python
velocity_noise = np.random.normal(
    0,
    VELOCITY_FLUCTUATION,
    INSTABILITY_DURATION
)
```

Applications:
- turbulence,
- engine instability,
- airflow disturbance simulation.

---

## 4. Altitude Disturbance

Simulates unstable altitude variation.

```python
altitude_disturbance = (
    150 * np.sin(0.2 * time_window)
)
```

Applications:
- turbulence response,
- unstable climb/descent,
- flight instability simulation.

---

# 🏷️ Instability Labeling

The script labels instability regions:

```python
flight_df["instability_type"] = "normal"
```

Unstable regions become:

```python
"flight_instability"
```

Anomaly labels are also updated:

```python
flight_df["anomaly"] = 1
```

---

# 🧠 Instability Simulation Workflow

```text
Load Telemetry Dataset
        ↓
Define Instability Window
        ↓
Inject Pitch Oscillation
        ↓
Inject Roll Oscillation
        ↓
Inject Velocity Fluctuation
        ↓
Inject Altitude Disturbance
        ↓
Label Instability Region
        ↓
Save Simulated Dataset
```

---

# 📁 Output Dataset

The generated instability dataset is saved as:

```text
dataset/flight_instability_simulation.csv
```

Additional columns include:
- anomaly
- instability_type

---

# ▶️ Run the Script

From project root:

```bash
python data_generation/simulate_instability.py
```

---

# 📌 Example Console Output

```text
========================================
 Flight Instability Simulation Complete
========================================

Injected Instability Types:
- Pitch Oscillation
- Roll Oscillation
- Velocity Fluctuation
- Altitude Disturbance
```

---

# 📈 Example Flight Instability

## Combined Instability Behavior

```text
Telemetry
 ^
 |
 |   ~~~~ /\ ~~~~ /\ ~~~~
 |
 +------------------------> Time
```

---

# 🔬 Applications

This module is useful for:
- aerospace anomaly detection,
- flight instability simulation,
- telemetry intelligence research,
- UAV monitoring,
- ML model training,
- and real-time aerospace analytics.

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Time-Series Simulation
- Aerospace Telemetry Modeling

---

# 📌 Future Improvements

Potential future enhancements:
- turbulence field simulation,
- aerodynamic instability modeling,
- weather disturbances,
- engine failure simulation,
- sensor corruption,
- multi-aircraft telemetry,
- and real-time instability streaming.

---

# 🏁 Summary

The `simulate_instability.py` module creates realistic unstable flight conditions by injecting oscillatory and disturbance-based instability patterns into telemetry streams. It provides advanced anomaly scenarios for machine learning training, telemetry analytics, and real-time aerospace anomaly detection within the Flight Anomaly AI Core platform.