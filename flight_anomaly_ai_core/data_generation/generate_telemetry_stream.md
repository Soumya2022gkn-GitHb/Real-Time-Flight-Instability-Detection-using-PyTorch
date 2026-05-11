The generate_telemetry_stream.py script simulates a real-time aerospace telemetry streaming system by continuously sending telemetry packets from a flight instability dataset. It mimics how aircraft telemetry data is streamed in production aerospace systems, enabling real-time anomaly monitoring, streaming inference, telemetry analytics, and observability workflows.

# 📡 Flight Telemetry Streaming Module

## Overview

The `generate_telemetry_stream.py` script simulates real-time flight telemetry streaming for the `flight_anomaly_ai_core` project. The module continuously streams telemetry packets one-by-one from a flight instability dataset, emulating how real aerospace telemetry systems transmit flight data during flight operations or testing.

The streaming system supports:
- real-time telemetry simulation,
- anomaly event monitoring,
- telemetry logging,
- streaming inference pipelines,
- and aerospace observability workflows.

This module acts as the real-time data source for anomaly detection and telemetry intelligence systems.

---

# 📂 File Location

```text
flight_anomaly_ai_core/
│
├── data_generation/
│   └── generate_telemetry_stream.py
│
├── dataset/
│   ├── flight_instability_simulation.csv
│   └── telemetry_stream.log
```

---

# 🚀 Features

The module provides:
- real-time telemetry streaming,
- telemetry packet simulation,
- JSON telemetry logging,
- anomaly event highlighting,
- configurable streaming delays,
- continuous streaming support,
- and telemetry replay functionality.

---

# ⚙️ Configuration

```python
STREAM_DELAY = 0.1
ENABLE_JSON_LOGGING = True
LOOP_STREAM = False
MAX_STREAM_SAMPLES = None
```

### Description

| Parameter | Description |
|---|---|
| `STREAM_DELAY` | Delay between telemetry packets |
| `ENABLE_JSON_LOGGING` | Enables telemetry packet logging |
| `LOOP_STREAM` | Enables continuous replay mode |
| `MAX_STREAM_SAMPLES` | Optional sample limit |

---

# 📊 Input Dataset

The script loads telemetry data from:

```text
dataset/flight_instability_simulation.csv
```

Required telemetry columns:
- timestamp
- altitude
- velocity
- pitch
- roll
- yaw
- anomaly

---

# 🧠 Real-Time Telemetry Workflow

```text
Load Telemetry Dataset
        ↓
Validate Columns
        ↓
Create Telemetry Packet
        ↓
Print Telemetry Packet
        ↓
Save JSON Log
        ↓
Highlight Anomalies
        ↓
Apply Streaming Delay
        ↓
Repeat
```

---

# 📦 Telemetry Packet Structure

Each telemetry packet is converted into JSON format:

```json
{
  "timestamp": 3500,
  "altitude": 10125.42,
  "velocity": 268.15,
  "pitch": 8.92,
  "roll": 17.43,
  "yaw": 91.12,
  "anomaly": 1
}
```

This structure resembles real telemetry messaging systems used in aerospace monitoring platforms.

---

# 🚨 Anomaly Detection Events

When anomaly packets are detected:

```python
if telemetry_packet["anomaly"] == 1:
```

the system prints alert information:

```text
===================================
 ANOMALY DETECTED
===================================
```

The alert includes:
- timestamp,
- altitude,
- velocity,
- pitch,
- and roll values.

---

# 📁 Output Log File

Telemetry packets are saved to:

```text
dataset/telemetry_stream.log
```

Each line contains a JSON telemetry packet.

Example:

```json
{"timestamp": 100, "altitude": 10012.5, "velocity": 251.8}
```

---

# ⏱️ Real-Time Streaming

The streaming delay simulates real telemetry systems:

```python
time.sleep(STREAM_DELAY)
```

Example:

```python
STREAM_DELAY = 0.1
```

This streams:
- 10 telemetry packets per second.

---

# 🔁 Continuous Streaming Mode

Enable infinite replay mode:

```python
LOOP_STREAM = True
```

This continuously replays telemetry streams for:
- monitoring systems,
- streaming inference pipelines,
- and observability testing.

---

# ▶️ Run the Script

From project root:

```bash
python data_generation/generate_telemetry_stream.py
```

---

# 📌 Example Console Output

```text
========================================
 Flight Telemetry Streaming Started
========================================

{
  "timestamp": 3500,
  "altitude": 10084.22,
  "velocity": 271.44,
  "pitch": 10.12,
  "roll": 22.45,
  "yaw": 91.08,
  "anomaly": 1
}
```

---

# 📈 Example Streaming Flow

```text
Telemetry Stream
        ↓
JSON Packet
        ↓
Anomaly Detection
        ↓
Real-Time Monitoring
        ↓
Telemetry Logging
```

---

# 🔬 Applications

This module is useful for:
- aerospace telemetry replay,
- real-time anomaly detection,
- UAV monitoring systems,
- telemetry intelligence platforms,
- streaming ML inference,
- and observability testing.

---

# 🛠️ Technologies Used

- Python
- Pandas
- JSON Streaming
- Time-Series Processing
- Real-Time Telemetry Simulation

---

# 📌 Future Improvements

Potential future enhancements:
- Kafka telemetry streaming,
- WebSocket telemetry servers,
- FastAPI streaming APIs,
- cloud telemetry pipelines,
- distributed streaming,
- multi-aircraft telemetry,
- and live dashboard integration.

---

# 🏁 Summary

The `generate_telemetry_stream.py` module simulates real-time aerospace telemetry transmission by streaming telemetry packets continuously from instability datasets. It provides a realistic foundation for streaming anomaly detection, telemetry analytics, and real-time aerospace intelligence systems within the Flight Anomaly AI Core platform.