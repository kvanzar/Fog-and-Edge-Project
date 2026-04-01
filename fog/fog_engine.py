# ============================================================
#  fog/fog_engine.py  —  Unified Fog Node Entry Point
#
#  Three pipelines:
#    1. Sensor Anomaly     — Isolation Forest (MQTT, optional)
#    2. Camera Intruder    — YOLO + FaceRec   (main thread)
#    3. Vibration Perimeter — SW-420 + Arduino (background thread)
#
#  Run: python -m fog.fog_engine
# ============================================================

import json
import time
import logging
import threading

import paho.mqtt.client as mqtt

from fog.config           import MQTT_BROKER, MQTT_PORT, SENSOR_TOPIC
from fog.sensor_engine    import SensorAnomalyEngine
from fog.camera_guard     import CameraGuard
from fog.vibration_engine import VibrationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)-14s]  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fog_engine.log"),
    ],
)
log = logging.getLogger("FogEngine")


class FogEngine:

    def __init__(self):
        self.mqtt_client      = None
        self.sensor_engine    = None
        self.vibration_engine = None
        self.camera_guard     = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            log.info(f"MQTT connected → {MQTT_BROKER}:{MQTT_PORT}")
            client.subscribe(SENSOR_TOPIC)
        else:
            log.error(f"MQTT failed (rc={rc})")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            if self.sensor_engine:
                self.sensor_engine.process(data, msg.topic)
        except Exception as e:
            log.error(f"MQTT message error: {e}")

    def _start_mqtt(self):
        try:
            self.mqtt_client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id="fog_engine",
            )
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self.sensor_engine = SensorAnomalyEngine(self.mqtt_client)
            self.sensor_engine.start()
            threading.Thread(
                target=self.mqtt_client.loop_forever,
                name="MQTT-Loop", daemon=True,
            ).start()
            log.info("Pipeline 1 active: Sensor anomaly (Isolation Forest + MQTT)")
        except Exception as e:
            log.warning(f"MQTT not available ({e}) — sensor anomaly pipeline skipped.")
            self.mqtt_client = None

    def _start_vibration(self):
        self.vibration_engine = VibrationEngine(mqtt_client=self.mqtt_client)
        self.vibration_engine.start()
        log.info("Pipeline 3 active: Vibration perimeter (SW-420 + Arduino Uno)")

    def start(self):
        log.info("=" * 62)
        log.info("   Fog Engine — Starting")
        log.info("   Pipeline 1 : Sensor Anomaly  (Isolation Forest)")
        log.info("   Pipeline 2 : Camera Intruder (YOLO + FaceRec)")
        log.info("   Pipeline 3 : Vibration Guard (SW-420 + Arduino)")
        log.info("=" * 62)

        self._start_mqtt()
        self._start_vibration()

        self.camera_guard = CameraGuard(
            mqtt_client=self.mqtt_client,
            on_alert_callback=self.vibration_engine.notify_camera_alert,
            vibration_engine=self.vibration_engine,
        )

        log.info("Starting camera on main thread (required on macOS)...")
        log.info("Press Q in the camera window to quit.")

        try:
            self.camera_guard.start(show_window=True)
        except KeyboardInterrupt:
            pass
        finally:
            log.info("Shutdown signal received.")
            self.camera_guard.stop()
            self.vibration_engine.stop()
            if self.mqtt_client:
                self.mqtt_client.disconnect()
            log.info("Fog Engine stopped cleanly.")


def main():
    FogEngine().start()

if __name__ == "__main__":
    main()