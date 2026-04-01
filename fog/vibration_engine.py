# ============================================================
#  fog/vibration_engine.py  —  SW-420 Vibration Processing
# ============================================================

import json
import time
import logging
import threading
import subprocess
from datetime import datetime
from collections import defaultdict

import serial
import serial.tools.list_ports

from fog.config import (
    SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT,
    VIBRATION_THREAT, CORRELATION_WINDOW_S,
    RESULTS_TOPIC,
)

log = logging.getLogger("VibrationEngine")

VIBRATION_ALERT_LEVEL = {
    "PERIMETER_KNOCK":  "LOW",
    "PERIMETER_FORCE":  "MEDIUM",
    "PERIMETER_BREACH": "HIGH",
    "CONFIRMED_BREACH": "CRITICAL",
}

WARNING_DISPLAY_SECONDS = 5


class VibrationEngine:

    def __init__(self, mqtt_client=None):
        self.mqtt_client       = mqtt_client
        self.running           = False
        self._serial           = None
        self._last_vibration_t = 0.0
        self._last_camera_t    = 0.0
        self._lock             = threading.Lock()

        self.warning_active = False
        self.warning_text   = ""
        self.warning_level  = ""
        self._warning_until = 0.0

        self.stats = {
            "total_events":     0,
            "knock":            0,
            "force":            0,
            "breach":           0,
            "confirmed_breach": 0,
            "arduino_online":   False,
        }

    # ── SOUND ─────────────────────────────────────────────────

    def _play_sound(self, level: str):
        sound_map = {
            "LOW":      "/System/Library/Sounds/Tink.aiff",
            "MEDIUM":   "/System/Library/Sounds/Sosumi.aiff",
            "HIGH":     "/System/Library/Sounds/Basso.aiff",
            "CRITICAL": "/System/Library/Sounds/Funk.aiff",
        }
        def _play():
            try:
                subprocess.run(
                    ["afplay", sound_map.get(level, sound_map["HIGH"])],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                print("\a", end="", flush=True)
        threading.Thread(target=_play, daemon=True).start()

    # ── ON-SCREEN WARNING ─────────────────────────────────────

    def _set_warning(self, text: str, level: str):
        self.warning_text   = text
        self.warning_level  = level
        self.warning_active = True
        self._warning_until = time.time() + WARNING_DISPLAY_SECONDS

    def tick_warning(self):
        if self.warning_active and time.time() > self._warning_until:
            self.warning_active = False
            self.warning_text   = ""
            self.warning_level  = ""

    # ── CROSS-SENSOR CORRELATION ──────────────────────────────

    def notify_camera_alert(self, threat_level: str):
        now = time.time()
        with self._lock:
            self._last_camera_t = now
            time_since_vib = now - self._last_vibration_t
        if 0 < time_since_vib <= CORRELATION_WINDOW_S:
            self._raise_confirmed_breach(threat_level, round(time_since_vib, 1))

    def _check_vibration_correlation(self):
        now = time.time()
        with self._lock:
            time_since_cam = now - self._last_camera_t
        if 0 < time_since_cam <= CORRELATION_WINDOW_S:
            self._raise_confirmed_breach("recent camera alert", round(time_since_cam, 1))

    def _raise_confirmed_breach(self, camera_threat: str, delta_s: float):
        self.stats["confirmed_breach"] += 1
        msg = f"CONFIRMED BREACH — vibration + camera both triggered within {delta_s}s"
        self._set_warning(msg, "CRITICAL")
        self._play_sound("CRITICAL")
        self._publish({
            "type":         "VIBRATION_ALERT",
            "threat_level": "CONFIRMED_BREACH",
            "alert_level":  "CRITICAL",
            "timestamp":    datetime.now().isoformat(),
            "source":       "vibration_engine+camera_guard",
            "message":      msg,
        })
        log.critical(msg)

    # ── EVENT PROCESSING ──────────────────────────────────────

    def _process_event(self, data: dict):
        event = data.get("event", "")

        if event == "HEARTBEAT":
            self.stats["arduino_online"] = True
            return
        if event == "READY":
            self.stats["arduino_online"] = True
            log.info("Arduino Uno online.")
            return
        if event != "VIBRATION":
            return

        intensity    = data.get("intensity", "LOW")
        pulse_count  = data.get("pulse_count", 0)
        threat_label = VIBRATION_THREAT.get(intensity, "PERIMETER_KNOCK")
        alert_level  = VIBRATION_ALERT_LEVEL.get(threat_label, "LOW")

        self.stats["total_events"] += 1
        if intensity == "LOW":      self.stats["knock"]  += 1
        elif intensity == "MEDIUM": self.stats["force"]  += 1
        elif intensity == "HIGH":   self.stats["breach"] += 1

        with self._lock:
            self._last_vibration_t = time.time()

        warning_messages = {
            "LOW":    f"VIBRATION DETECTED — light knock  (pulses: {pulse_count})",
            "MEDIUM": f"VIBRATION ALERT — repeated force  (pulses: {pulse_count})",
            "HIGH":   f"BREACH ALERT — heavy impact on sensor!  (pulses: {pulse_count})",
        }
        self._set_warning(warning_messages.get(intensity, "Vibration detected"), alert_level)
        self._play_sound(alert_level)
        self._publish({
            "type":         "VIBRATION_ALERT",
            "threat_level": threat_label,
            "alert_level":  alert_level,
            "intensity":    intensity,
            "pulse_count":  pulse_count,
            "timestamp":    datetime.now().isoformat(),
            "source":       "vibration_engine",
        })
        log.warning(f"VIBRATION | {threat_label} | intensity={intensity} | pulses={pulse_count}")
        self._check_vibration_correlation()

    def _publish(self, payload: dict):
        if self.mqtt_client:
            try:
                self.mqtt_client.publish(RESULTS_TOPIC, json.dumps(payload))
            except Exception as e:
                log.error(f"MQTT publish failed: {e}")

    # ── SERIAL ────────────────────────────────────────────────

    def _connect_serial(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=SERIAL_PORT, baudrate=SERIAL_BAUD, timeout=SERIAL_TIMEOUT,
            )
            time.sleep(2)
            log.info(f"Serial connected: {SERIAL_PORT} @ {SERIAL_BAUD} baud")
            return True
        except serial.SerialException as e:
            available = [str(p) for p in serial.tools.list_ports.comports()]
            log.warning(
                f"Cannot open '{SERIAL_PORT}': {e}\n"
                f"  Available ports: {available}\n"
                f"  Update SERIAL_PORT in fog/config.py"
            )
            return False

    def _read_loop(self):
        while self.running:
            if self._serial is None or not self._serial.is_open:
                time.sleep(5)
                self._connect_serial()
                continue
            try:
                raw  = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("{"):
                    self._process_event(json.loads(line))
            except json.JSONDecodeError:
                pass
            except serial.SerialException as e:
                log.error(f"Serial error: {e}")
                self._serial = None
                self.stats["arduino_online"] = False
            except Exception as e:
                log.error(f"Read loop error: {e}")

    # ── START / STOP ──────────────────────────────────────────

    def start(self):
        connected = self._connect_serial()
        if not connected:
            log.warning(
                "Vibration engine started without Arduino. "
                "Connect Arduino via USB and set SERIAL_PORT in config.py."
            )
        self.running = True
        threading.Thread(target=self._read_loop, name="VibrationEngine", daemon=True).start()
        log.info("Vibration engine running.")

    def stop(self):
        self.running = False
        if self._serial and self._serial.is_open:
            self._serial.close()