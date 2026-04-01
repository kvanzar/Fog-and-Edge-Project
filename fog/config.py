# ============================================================
#  fog/config.py  —  Single source of truth for all settings
# ============================================================

# ── MQTT ──────────────────────────────────────────────────────
MQTT_BROKER       = "localhost"
MQTT_PORT         = 1883
SENSOR_TOPIC      = "fog/sensors/#"
RESULTS_TOPIC     = "fog/results"
CAMERA_TOPIC      = "fog/camera/intruder"
COMMAND_TOPIC_TPL = "fog/commands/{node_id}"

# ── SENSOR ANOMALY DETECTION ──────────────────────────────────
FEATURES            = ["temperature", "humidity", "accel_mag"]
CONTAMINATION       = 0.05
N_ESTIMATORS        = 100
TRAIN_BUFFER_SIZE   = 200
RETRAIN_INTERVAL_S  = 300
MODEL_PATH          = "fog/models/isolation_forest.pkl"

# ── CAMERA / INTRUDER DETECTION ───────────────────────────────
CAMERA_INDEX        = 0
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
PROCESS_EVERY_N     = 5
ALERT_COOLDOWN_SEC  = 10
YOLO_MODEL          = "yolov8n.pt"
YOLO_CONFIDENCE     = 0.5
FACE_TOLERANCE      = 0.5
KNOWN_FACES_DIR     = "fog/known_faces"
SNAPSHOTS_DIR       = "fog/intruder_snapshots"

# ── VIBRATION SENSOR (Arduino Uno + SW-420) ───────────────────
SERIAL_PORT      = "/dev/tty.usbserial-1130"   # ← your port
SERIAL_BAUD      = 9600
SERIAL_TIMEOUT   = 2

VIBRATION_THREAT = {
    "LOW":    "PERIMETER_KNOCK",
    "MEDIUM": "PERIMETER_FORCE",
    "HIGH":   "PERIMETER_BREACH",
}

CORRELATION_WINDOW_S = 15