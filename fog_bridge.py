"""
fog_bridge.py  —  FogSec Web Bridge
====================================
Bridges the fog engine (camera + vibration) with the ESP32 sensor
and serves everything to the web dashboard via Flask + WebSocket.

Run:
    pip install flask flask-socketio requests opencv-python
    python fog_bridge.py

Then open http://localhost:5000 in your browser.

Config:
    Set ESP32_IP below to your ESP32's IP address (shown in Arduino Serial Monitor).
    Set AQI_API_KEY to your OpenWeatherMap API key (free at openweathermap.org).
    Set LAT/LON to your location.
"""

import os
import cv2
import time
import json
import math
import base64
import logging
import threading
import requests
from datetime import datetime
from collections import deque
from pathlib import Path

from flask import Flask, render_template, send_from_directory, jsonify
from flask_socketio import SocketIO, emit

# ── CONFIGURATION ─────────────────────────────────────────────
ESP32_IP        = "192.168.1.100"      # ← Change to your ESP32's IP
ESP32_PORT      = 80
SENSOR_POLL_S   = 2                    # Poll ESP32 every 2 seconds

AQI_API_KEY     = "87c413047dbc069abdcaba6d9c043324"   # ← Free at openweathermap.org
LAT             = 12.9184              # Chennai latitude (change to yours)
LON             = 79.13255           # Chennai longitude
AQI_POLL_S      = 300                 # Poll AQI every 5 minutes

CAMERA_INDEX    = 0
KNOWN_FACES_DIR = "fog/known_faces"
SNAPSHOTS_DIR   = "fog/intruder_snapshots"
FLAGGED_FILE    = "fog/flagged_faces.json"   # Stores flagged encodings

TEMP_HISTORY_MAX = 120   # Keep last 120 readings for ML model

# ── FLASK + SOCKETIO ──────────────────────────────────────────
app    = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "fogsec-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-16s] %(levelname)s %(message)s"
)
log = logging.getLogger("FogBridge")

# ── SHARED STATE ──────────────────────────────────────────────
STATE = {
    "temp":          None,
    "humidity":      None,
    "heat_index":    None,
    "sensor_ok":     False,
    "temp_history":  deque(maxlen=TEMP_HISTORY_MAX),
    "hum_history":   deque(maxlen=TEMP_HISTORY_MAX),
    "ts_history":    deque(maxlen=TEMP_HISTORY_MAX),
    "aqi":           None,
    "aqi_category":  "Unknown",
    "aqi_ts":        None,
    "alerts":        deque(maxlen=200),   # All events
    "snapshots":     [],                  # {id, path, b64, threat, ts, resolved, name}
    "flagged":       [],                  # Flagged face encodings (stored as lists)
    "stats": {
        "known": 0, "visitors": 0, "suspicious": 0,
        "intruders": 0, "anomalies": 0, "total_events": 0
    },
    "known_persons": [],
    "uptime_start":  time.time(),
}

# ── SENSOR POLLING ────────────────────────────────────────────

def poll_esp32():
    """Poll ESP32 HTTP endpoint every SENSOR_POLL_S seconds."""
    while True:
        try:
            url = f"http://{ESP32_IP}:{ESP32_PORT}/sensor"
            r   = requests.get(url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                temp = data.get("temperature")
                hum  = data.get("humidity")
                hi   = data.get("heat_index", compute_heat_index(temp, hum))

                STATE["temp"]       = temp
                STATE["humidity"]   = hum
                STATE["heat_index"] = hi
                STATE["sensor_ok"]  = True

                ts = datetime.now().strftime("%H:%M:%S")
                STATE["temp_history"].append(temp)
                STATE["hum_history"].append(hum)
                STATE["ts_history"].append(ts)

                # Run ML prediction
                prediction = predict_temperature()

                # Emit to dashboard
                socketio.emit("sensor_update", {
                    "temp":       temp,
                    "humidity":   hum,
                    "heat_index": hi,
                    "ts":         ts,
                    "prediction": prediction,
                    "history": {
                        "temps":  list(STATE["temp_history"]),
                        "hums":   list(STATE["hum_history"]),
                        "labels": list(STATE["ts_history"]),
                    }
                })
                add_alert("SENSOR", "DHT22/ESP32",
                          f"temp:{temp:.1f}°C  hum:{hum:.1f}%  heat_index:{hi:.1f}°C", "info")

        except requests.exceptions.ConnectionError:
            if STATE["sensor_ok"]:
                log.warning(f"ESP32 unreachable at {ESP32_IP}")
                STATE["sensor_ok"] = False
                socketio.emit("sensor_offline", {})
        except Exception as e:
            log.error(f"Sensor poll error: {e}")

        time.sleep(SENSOR_POLL_S)

# ── HEAT INDEX ────────────────────────────────────────────────

def compute_heat_index(T, H):
    if T is None or H is None or T < 27:
        return T
    hi = (-8.78469475556 + 1.61139411*T + 2.33854883889*H
          - 0.14611605*T*H - 0.012308094*T*T - 0.0164248277778*H*H
          + 0.002211732*T*T*H + 0.00072546*T*H*H - 0.000003582*T*T*H*H)
    return max(T, hi)

# ── ML TEMPERATURE PREDICTION ─────────────────────────────────

def predict_temperature():
    """
    Simple linear regression on recent temperature history.
    Predicts next 6 hours using trend extrapolation.
    Returns dict with predictions + outdoor advisory.
    """
    temps = list(STATE["temp_history"])
    if len(temps) < 10:
        return {"status": "insufficient_data", "min_needed": 10, "have": len(temps)}

    n = len(temps)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(temps) / n

    # Linear regression: y = mx + b
    num   = sum((x[i] - x_mean) * (temps[i] - y_mean) for i in range(n))
    denom = sum((x[i] - x_mean) ** 2 for i in range(n))
    m     = num / denom if denom != 0 else 0
    b     = y_mean - m * x_mean

    # Predict next 6h (180 readings at 2s cadence = 6min, scaled to 6h)
    steps_per_hour = 1800 / SENSOR_POLL_S  # readings per hour
    forecasts = []
    for h in [1, 2, 3, 4, 6]:
        future_x    = n + h * steps_per_hour
        predicted_t = m * future_x + b
        # Clamp to realistic range
        predicted_t = max(10, min(50, predicted_t))
        forecasts.append({"hour": f"+{h}h", "temp": round(predicted_t, 1)})

    current = temps[-1]

    # Advisory logic
    advisory, go_out_score = get_outdoor_advisory(current, forecasts)

    return {
        "status":      "ok",
        "current":     round(current, 1),
        "trend_per_h": round(m * steps_per_hour, 2),
        "forecasts":   forecasts,
        "advisory":    advisory,
        "go_out_score": go_out_score,  # 0-100, higher = better time to go out
        "r_squared":   compute_r_squared(temps, m, b, x_mean, y_mean),
    }

def compute_r_squared(temps, m, b, x_mean, y_mean):
    n       = len(temps)
    ss_res  = sum((temps[i] - (m*i + b))**2 for i in range(n))
    ss_tot  = sum((t - y_mean)**2 for t in temps)
    return round(1 - ss_res/ss_tot, 3) if ss_tot > 0 else 0

def get_outdoor_advisory(current_temp, forecasts):
    """Returns (advisory_text, go_out_score 0-100)."""
    score = 100

    if current_temp > 40:
        return ("🔴 EXTREME HEAT — Do not go outside. Dangerous heat levels.", 0)
    if current_temp > 37:
        score -= 60
        msg = "🔴 VERY HOT — Avoid outdoor activity. Risk of heat exhaustion."
    elif current_temp > 33:
        score -= 35
        msg = "🟠 HOT — Limit time outside. Carry water, wear sunscreen."
    elif current_temp > 28:
        score -= 15
        msg = "🟡 WARM — Comfortable for short outings. Avoid peak noon hours."
    elif current_temp > 18:
        msg = "🟢 IDEAL — Great conditions to go outside."
    elif current_temp > 10:
        score -= 10
        msg = "🔵 COOL — Pleasant. Light jacket recommended."
    else:
        score -= 40
        msg = "🔵 COLD — Bundle up well before going out."

    # Check forecast trend
    if forecasts:
        future_max = max(f["temp"] for f in forecasts)
        if future_max > 38 and current_temp < 35:
            msg += f" ⚠ Warning: temperature forecast to reach {future_max}°C — go out early."
            score -= 10

    return (msg, max(0, min(100, score)))

# ── AQI POLLING ───────────────────────────────────────────────

def poll_aqi():
    """Poll OpenWeatherMap Air Pollution API every AQI_POLL_S seconds."""
    while True:
        try:
            if AQI_API_KEY == "YOUR_OPENWEATHERMAP_KEY":
                # Demo mode — simulate AQI
                simulated_aqi = 2
                STATE["aqi"]          = simulated_aqi
                STATE["aqi_category"] = aqi_category(simulated_aqi)
                STATE["aqi_ts"]       = datetime.now().strftime("%H:%M")
                socketio.emit("aqi_update", {
                    "aqi":      simulated_aqi,
                    "category": STATE["aqi_category"],
                    "ts":       STATE["aqi_ts"],
                    "demo":     True,
                })
            else:
                url = (f"http://api.openweathermap.org/data/2.5/air_pollution"
                       f"?lat={LAT}&lon={LON}&appid={AQI_API_KEY}")
                r   = requests.get(url, timeout=5)
                if r.status_code == 200:
                    data     = r.json()
                    aqi_val  = data["list"][0]["main"]["aqi"]
                    comps    = data["list"][0]["components"]
                    STATE["aqi"]          = aqi_val
                    STATE["aqi_category"] = aqi_category(aqi_val)
                    STATE["aqi_ts"]       = datetime.now().strftime("%H:%M")
                    socketio.emit("aqi_update", {
                        "aqi":        aqi_val,
                        "category":   STATE["aqi_category"],
                        "ts":         STATE["aqi_ts"],
                        "components": {
                            "pm2_5": round(comps.get("pm2_5", 0), 1),
                            "pm10":  round(comps.get("pm10", 0), 1),
                            "no2":   round(comps.get("no2", 0), 1),
                            "o3":    round(comps.get("o3", 0), 1),
                        }
                    })
                    log.info(f"AQI updated: {aqi_val} ({STATE['aqi_category']})")
        except Exception as e:
            log.error(f"AQI poll error: {e}")
        time.sleep(AQI_POLL_S)

def aqi_category(aqi):
    return {1:"Good",2:"Fair",3:"Moderate",4:"Poor",5:"Very Poor"}.get(aqi,"Unknown")

# ── CAMERA STREAM ─────────────────────────────────────────────

def camera_stream_thread():
    """Captures frames and emits MJPEG-style base64 frames via WebSocket."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        log.error("Camera not available")
        socketio.emit("camera_offline", {})
        return

    log.info("Camera stream started")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1

        # Encode frame as JPEG → base64
        _, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64     = base64.b64encode(buf).decode("utf-8")

        # Only emit every 3rd frame to reduce bandwidth (~10fps)
        if frame_count % 3 == 0:
            socketio.emit("camera_frame", {"frame": b64})

        time.sleep(0.033)   # ~30fps capture rate

# ── ALERTS ────────────────────────────────────────────────────

def add_alert(level, source, message, alert_type="info", snapshot_id=None):
    alert = {
        "id":          f"evt_{int(time.time()*1000)}",
        "ts":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level":       level,
        "source":      source,
        "message":     message,
        "type":        alert_type,
        "snapshot_id": snapshot_id,
    }
    STATE["alerts"].appendleft(alert)
    STATE["stats"]["total_events"] += 1
    socketio.emit("new_alert", alert)

# ── SNAPSHOT MANAGEMENT ───────────────────────────────────────

def load_snapshots():
    """Load existing snapshots from disk."""
    Path(SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
    snaps = []
    for f in sorted(Path(SNAPSHOTS_DIR).glob("*.jpg"), reverse=True)[:50]:
        try:
            img = cv2.imread(str(f))
            if img is not None:
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64    = base64.b64encode(buf).decode("utf-8")
                snaps.append({
                    "id":       f.stem,
                    "path":     str(f),
                    "b64":      b64,
                    "ts":       datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "threat":   "UNKNOWN" if "intruder" in f.stem else "SUSPICIOUS",
                    "resolved": False,
                    "name":     None,
                })
        except Exception:
            pass
    STATE["snapshots"] = snaps
    log.info(f"Loaded {len(snaps)} snapshots")

def load_known_persons():
    Path(KNOWN_FACES_DIR).mkdir(parents=True, exist_ok=True)
    names = [f.stem for f in Path(KNOWN_FACES_DIR).glob("*.jpg")]
    STATE["known_persons"] = names
    log.info(f"Known persons: {names}")

def load_flagged():
    if Path(FLAGGED_FILE).exists():
        with open(FLAGGED_FILE) as f:
            STATE["flagged"] = json.load(f)
        log.info(f"Loaded {len(STATE['flagged'])} flagged persons")

def save_flagged():
    Path(FLAGGED_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(FLAGGED_FILE, "w") as f:
        json.dump(STATE["flagged"], f)

# ── FLASK ROUTES ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "dashboard.html")

@app.route("/api/state")
def api_state():
    """Full initial state dump for when dashboard first loads."""
    return jsonify({
        "sensor": {
            "temp":       STATE["temp"],
            "humidity":   STATE["humidity"],
            "heat_index": STATE["heat_index"],
            "ok":         STATE["sensor_ok"],
        },
        "aqi":      STATE["aqi"],
        "aqi_cat":  STATE["aqi_category"],
        "alerts":   list(STATE["alerts"])[:100],
        "snapshots": [{k:v for k,v in s.items() if k!="b64"}
                      for s in STATE["snapshots"]],
        "known_persons": STATE["known_persons"],
        "stats":    STATE["stats"],
        "uptime":   int(time.time() - STATE["uptime_start"]),
        "history": {
            "temps":  list(STATE["temp_history"]),
            "hums":   list(STATE["hum_history"]),
            "labels": list(STATE["ts_history"]),
        },
        "prediction": predict_temperature(),
    })

@app.route("/api/snapshots")
def api_snapshots():
    """Return snapshots with base64 images."""
    return jsonify(STATE["snapshots"][:30])

@app.route("/api/snapshot/<snap_id>/image")
def api_snapshot_image(snap_id):
    snap = next((s for s in STATE["snapshots"] if s["id"] == snap_id), None)
    if not snap:
        return jsonify({"error": "not found"}), 404
    return jsonify({"b64": snap["b64"]})

# ── SOCKETIO EVENTS ───────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    log.info("Dashboard client connected")
    # Send full state on connect
    emit("init_state", {
        "sensor": {
            "temp":       STATE["temp"],
            "humidity":   STATE["humidity"],
            "heat_index": STATE["heat_index"],
            "ok":         STATE["sensor_ok"],
        },
        "aqi":      STATE["aqi"],
        "aqi_cat":  STATE["aqi_category"],
        "alerts":   list(STATE["alerts"])[:50],
        "snapshots": STATE["snapshots"][:20],
        "known_persons": STATE["known_persons"],
        "stats":    STATE["stats"],
        "history": {
            "temps":  list(STATE["temp_history"]),
            "hums":   list(STATE["hum_history"]),
            "labels": list(STATE["ts_history"]),
        },
    })

@socketio.on("authorise_person")
def on_authorise(data):
    """User authorises a snapshot person as known."""
    snap_id = data.get("snap_id")
    name    = data.get("name", "").strip()
    if not name:
        return

    snap = next((s for s in STATE["snapshots"] if s["id"] == snap_id), None)
    if not snap:
        return

    # Save face photo to known_faces/
    safe_name = "".join(c for c in name if c.isalnum() or c in ("_","-"))
    save_path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.jpg")
    img_data  = base64.b64decode(snap["b64"])
    with open(save_path, "wb") as f:
        f.write(img_data)

    snap["resolved"] = True
    snap["name"]     = safe_name
    STATE["known_persons"].append(safe_name)
    STATE["stats"]["known"] += 1

    add_alert("INFO", "WebUI", f"Person authorised as: {safe_name}", "success")
    emit("person_authorised", {"snap_id": snap_id, "name": safe_name}, broadcast=True)
    log.info(f"Person authorised: {safe_name} from snapshot {snap_id}")

@socketio.on("flag_person")
def on_flag(data):
    """Flag a person — next appearance triggers red alert."""
    snap_id     = data.get("snap_id")
    description = data.get("description", "flagged by operator")

    snap = next((s for s in STATE["snapshots"] if s["id"] == snap_id), None)
    if not snap:
        return

    STATE["flagged"].append({
        "snap_id":     snap_id,
        "description": description,
        "ts":          datetime.now().isoformat(),
    })
    save_flagged()
    snap["resolved"] = True
    snap["threat"]   = "FLAGGED"

    add_alert("CRITICAL", "WebUI",
              f"Person FLAGGED — will trigger red alert on re-entry. ID: {snap_id}", "error")
    emit("person_flagged", {"snap_id": snap_id}, broadcast=True)
    log.warning(f"Person flagged: {snap_id}")

@socketio.on("dismiss_alert")
def on_dismiss(data):
    alert_id = data.get("id")
    STATE["alerts"] = deque(
        (a for a in STATE["alerts"] if a["id"] != alert_id),
        maxlen=200
    )

# ── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 52)
    log.info("  FogSec Web Bridge Starting")
    log.info(f"  ESP32 target: http://{ESP32_IP}:{ESP32_PORT}/sensor")
    log.info(f"  Dashboard:    http://localhost:5001")
    log.info("=" * 52)

    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    load_snapshots()
    load_known_persons()
    load_flagged()

    # Start background threads
    threading.Thread(target=poll_esp32,          daemon=True, name="SensorPoller").start()
    threading.Thread(target=poll_aqi,            daemon=True, name="AQIPoller").start()
    threading.Thread(target=camera_stream_thread, daemon=True, name="CameraStream").start()

    add_alert("INFO", "FogBridge", "System started — all services online", "info")

    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)