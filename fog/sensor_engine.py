import json
import time
import logging
import threading
from datetime import datetime
from collections import deque

import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

from fog.config import (
    FEATURES, CONTAMINATION, N_ESTIMATORS,
    TRAIN_BUFFER_SIZE, RETRAIN_INTERVAL_S,
    MODEL_PATH, RESULTS_TOPIC, COMMAND_TOPIC_TPL
)

log = logging.getLogger("SensorEngine")


class SensorAnomalyEngine:
    """
    Isolation Forest–based anomaly detector for ESP32 sensor streams.

    Usage:
        engine = SensorAnomalyEngine(mqtt_client)
        engine.start()                       # starts retraining thread
        engine.process(data_dict, topic)     # call from MQTT on_message
    """

    def __init__(self, mqtt_client):
        self.mqtt_client  = mqtt_client
        self.model        = None
        self.is_trained   = False
        self.train_buffer = deque(maxlen=TRAIN_BUFFER_SIZE)
        self.lock         = threading.Lock()
        self.stats = {
            "total_received":   0,
            "anomalies_detected": 0,
            "last_anomaly":     None,
        }

    # ── MODEL ─────────────────────────────────────────────────

    def load_or_create_model(self):
        """Load saved model if it exists, otherwise create fresh."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        if os.path.exists(MODEL_PATH):
            self.model      = joblib.load(MODEL_PATH)
            self.is_trained = True
            log.info(f"Loaded existing model from {MODEL_PATH}")
        else:
            self.model = IsolationForest(
                n_estimators=N_ESTIMATORS,
                contamination=CONTAMINATION,
                random_state=42,
            )
            log.info("New Isolation Forest created — waiting for training data.")

    def _train(self):
        """Fit model on current buffer contents."""
        if len(self.train_buffer) < 50:
            log.warning(f"Not enough data ({len(self.train_buffer)} samples). Need 50+.")
            return False

        X = np.array([[s[f] for f in FEATURES] for s in self.train_buffer])
        log.info(f"Training Isolation Forest on {len(X)} samples...")
        self.model.fit(X)
        self.is_trained = True
        joblib.dump(self.model, MODEL_PATH)
        log.info(f"Model trained and saved → {MODEL_PATH}")
        return True

    def _predict(self, features: dict) -> dict:
        """Run inference on one sample. Returns anomaly flag + score."""
        if not self.is_trained:
            return {"anomaly": False, "score": None, "status": "untrained"}

        X          = np.array([[features[f] for f in FEATURES]])
        prediction = self.model.predict(X)[0]       # 1=normal, -1=anomaly
        score      = float(self.model.score_samples(X)[0])  # more negative = worse

        return {
            "anomaly": prediction == -1,
            "score":   score,
            "status":  "ok",
        }

    # ── PROCESSING ────────────────────────────────────────────

    def process(self, data: dict, topic: str):
        """
        Main entry point — called for every incoming MQTT message.
        Validates data, buffers it, runs inference, publishes result.
        """
        device_id = data.get("device_id", "unknown")

        if not all(f in data for f in FEATURES):
            log.warning(f"Missing features from {device_id}: {list(data.keys())}")
            return

        features = {f: float(data[f]) for f in FEATURES}

        with self.lock:
            self.stats["total_received"] += 1
            self.train_buffer.append(features)

            # Auto-train when buffer first fills up
            if not self.is_trained and len(self.train_buffer) >= TRAIN_BUFFER_SIZE:
                log.info("Training buffer full — starting initial training.")
                self._train()

            result = self._predict(features)

            # Build unified result payload
            payload = {
                "type":            "SENSOR",
                "device_id":       device_id,
                "timestamp":       datetime.now().isoformat(),
                "features":        features,
                "anomaly":         result["anomaly"],
                "score":           result["score"],
                "status":          result["status"],
                "total_received":  self.stats["total_received"],
                "total_anomalies": self.stats["anomalies_detected"],
                "alert_level":     "NORMAL",
            }

            if result["anomaly"]:
                self.stats["anomalies_detected"] += 1
                self.stats["last_anomaly"] = datetime.now().isoformat()
                payload["alert_level"] = "HIGH"

                # Send ALERT back to the originating ESP32 node
                node_id = topic.split("/")[-1]
                cmd_topic = COMMAND_TOPIC_TPL.format(node_id=node_id)
                self.mqtt_client.publish(cmd_topic, "ALERT")

                log.warning(
                    f"ANOMALY | {device_id} | "
                    f"score={result['score']:.4f} | {features}"
                )
            else:
                score_str = f"{result['score']:.4f}" if result["score"] is not None else "N/A"
                log.info(f"Normal  | {device_id} | score={score_str}")

            self.mqtt_client.publish(RESULTS_TOPIC, json.dumps(payload))

    # ── BACKGROUND RETRAINING ─────────────────────────────────

    def start_retrain_scheduler(self):
        """Spawn a daemon thread that periodically retrains the model."""
        t = threading.Thread(target=self._retrain_loop, daemon=True)
        t.start()
        log.info(f"Retraining scheduler started (every {RETRAIN_INTERVAL_S}s).")

    def _retrain_loop(self):
        while True:
            time.sleep(RETRAIN_INTERVAL_S)
            with self.lock:
                if len(self.train_buffer) >= 50:
                    log.info("Scheduled retrain triggered.")
                    self._train()

    def start(self):
        self.load_or_create_model()
        self.start_retrain_scheduler()
