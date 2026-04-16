# 🌫️ AI-Driven Edge–Fog System for Real-Time Anomaly Detection

> Team: Kshitij · Ananya 

---

## Project Structure

```
fog_project/
│
├── edge/
│   └── src/
│       └── edge_sensor.ino        ← Flash to ESP32
│
├── fog/
│   ├── __init__.py
│   ├── config.py                  ← All settings (edit this first)
│   ├── fog_engine.py              ← Main entry point (run this)
│   ├── sensor_engine.py           ← Isolation Forest anomaly detection
│   ├── camera_guard.py            ← YOLO + face recognition
│   ├── models/                    ← Saved ML model (auto-created)
│   ├── known_faces/               ← Add team photos here
│   └── intruder_snapshots/        ← Auto-saved intrusion frames
│
├── dashboard/
│   └── app.py                     ← Streamlit dashboard
│
├── tools/
│   └── simulator.py               ← ESP32 data 
│
├── requirements.txt
└── README.md
```

---

## System Architecture

```
[ESP32 + DHT11 + MPU6050]
         │
    MQTT over WiFi (port 1883)
         │
┌────────▼────────────────────────────────────────────┐
│                  LAPTOP  (Fog Node)                 │
│                                                     │
│  Mosquitto MQTT Broker                              │
│         │                                           │
│  fog_engine.py                                      │
│    ├── Thread 1 ─ sensor_engine.py                  │
│    │       Isolation Forest                         │
│    │       → anomaly score per reading              │
│    │       → ALERT back to ESP32 on anomaly         │
│    │                                                │
│    └── Thread 2 ─ camera_guard.py                   │
│            Webcam → YOLOv8 (person?)                │
│            → face_recognition (known/unknown?)      │
│            → snapshot saved                         │
│            → INTRUDER_ALERT published               │
│                                                     │
│  dashboard/app.py  (localhost:8501)                 │
│    ├── Tab 1: Intruder alerts + snapshots           │
│    └── Tab 2: Sensor charts + anomaly log           │
└─────────────────────────────────────────────────────┘

[☁️  Cloud — Future Phase]
    Long-term storage, remote access, model retraining
```

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Add known faces
Drop one photo per team member into `fog/known_faces/`:
```
fog/known_faces/
    kshitij.jpg
    ananya.jpg
    mehak.jpg
    akanshi.jpg
```
Photo tips: front-facing, good lighting, one face per image.

### Step 3 — Start Mosquitto
```bash
# Windows
mosquitto

# Linux
sudo apt install mosquitto -y && mosquitto

# Mac
brew install mosquitto && mosquitto
```

### Step 4 — Run the fog engine
```bash
# From project root:
python -m fog.fog_engine
```

### Step 5 — Run the dashboard
```bash
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

### Step 6 (optional) — Simulate ESP32
```bash
python tools/simulator.py
```

### Step 6 (hardware) — Flash ESP32
1. Open `edge/src/edge_sensor.ino` in Arduino IDE
2. Install libraries: `PubSubClient`, `DHT sensor library`, `MPU6050`, `ArduinoJson`
3. Set `WIFI_SSID`, `WIFI_PASSWORD`, `MQTT_BROKER` = your laptop's local IP
4. Flash to ESP32

---

## Hardware Components

| Component       | Role                              | Approx. Cost |
|-----------------|-----------------------------------|--------------|
| ESP32 Dev Board | Edge node (sensor + MQTT publish) | ₹300–400     |
| DHT11 / DHT22   | Temperature & Humidity            | ₹80–150      |
| MPU6050         | Accelerometer / vibration         | ₹150–200     |
| LED + Buzzer    | Local anomaly alert               | ₹50          |
| Laptop          | Fog node — all ML inference       | —            |
| Laptop Webcam   | Intruder detection                | —            |

---


