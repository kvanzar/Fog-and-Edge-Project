# ============================================================
#  tools/simulator.py  —  ESP32 Sensor Simulator
#
#  Emulates an ESP32 edge node publishing sensor data over MQTT.
#  Use this to test the fog engine + dashboard WITHOUT hardware.
#
#  Injects realistic anomalies every 30 samples automatically.
#  Run: python tools/simulator.py  (from project root)
# ============================================================

import json
import math
import random
import time

import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost"
MQTT_PORT   = 1883
MQTT_TOPIC  = "fog/sensors/node1"
DEVICE_ID   = "SIMULATOR_01"

client = mqtt.Client(client_id="simulator")
client.connect(MQTT_BROKER, MQTT_PORT)
client.loop_start()

print("=" * 50)
print("  ESP32 Simulator Running")
print(f"  Publishing to: {MQTT_TOPIC}")
print("  Anomaly injected every 30 samples")
print("  Press Ctrl+C to stop")
print("=" * 50)

t = 0
while True:
    # Realistic baseline readings with gentle sine-wave drift + noise
    temperature = 22.0 + 3.0 * math.sin(t * 0.1)  + random.gauss(0, 0.5)
    humidity    = 55.0 + 5.0 * math.sin(t * 0.07) + random.gauss(0, 1.0)
    accel_mag   =  1.0                              + random.gauss(0, 0.05)

    # Inject anomaly every 30 samples (after warm-up)
    is_anomaly = (t % 30 == 0 and t > 0)
    if is_anomaly:
        temperature += random.choice([-15, +20])
        humidity    += random.choice([-30, +35])
        accel_mag   += random.uniform(3, 6)
        print(f"[t={t:3d}] 💥 ANOMALY injected!")

    payload = {
        "device_id":   DEVICE_ID,
        "timestamp":   time.time(),
        "temperature": round(temperature, 2),
        "humidity":    round(max(0, min(100, humidity)), 2),
        "accel_mag":   round(abs(accel_mag), 4),
    }

    client.publish(MQTT_TOPIC, json.dumps(payload))
    marker = " ← ANOMALY" if is_anomaly else ""
    print(
        f"[t={t:3d}] "
        f"T={temperature:6.2f}°C  "
        f"H={humidity:5.1f}%  "
        f"A={accel_mag:.3f}"
        f"{marker}"
    )

    t += 1
    time.sleep(2)
