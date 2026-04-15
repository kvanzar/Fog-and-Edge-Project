import cv2
import json
import time
import os
import logging
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import face_recognition
from ultralytics import YOLO

from fog.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    PROCESS_EVERY_N, ALERT_COOLDOWN_SEC,
    YOLO_MODEL, YOLO_CONFIDENCE, FACE_TOLERANCE,
    KNOWN_FACES_DIR, SNAPSHOTS_DIR,
    CAMERA_TOPIC, RESULTS_TOPIC,
)

log = logging.getLogger("CameraGuard")

class Threat:
    KNOWN       = "KNOWN"
    VISITOR     = "VISITOR"
    SUSPICIOUS  = "SUSPICIOUS"
    INTRUDER    = "INTRUDER"

THREAT_COLOUR = {
    Threat.KNOWN:      (0,   200,   0),
    Threat.VISITOR:    (200, 150,   0),
    Threat.SUSPICIOUS: (0,   200, 255),
    Threat.INTRUDER:   (0,     0, 255),
}

THREAT_BASE_SCORE = {
    Threat.KNOWN:      0,
    Threat.VISITOR:    20,
    Threat.SUSPICIOUS: 55,
    Threat.INTRUDER:   90,
}

ALERT_LEVELS = {Threat.VISITOR, Threat.SUSPICIOUS, Threat.INTRUDER}
LOITER_SECONDS = 10
DAYTIME_START = 6
DAYTIME_END   = 20
MIN_FACE_BODY_RATIO = 0.04

class CameraGuard:
    def __init__(self, mqtt_client=None, on_alert_callback=None, vibration_engine=None):
        self.mqtt_client        = mqtt_client
        self._on_alert_callback = on_alert_callback
        self._vibration_engine  = vibration_engine
        self.yolo            = None
        self.known_encodings = []
        self.known_names     = []
        self.running         = False

        self._loiter: dict[str, float] = {}
        self._last_alert: dict[str, float] = defaultdict(float)

        self.stats = {
            "frames_processed":  0,
            "persons_detected":  0,
            "known_seen":        0,
            "visitors":          0,
            "suspicious":        0,
            "intruders":         0,
        }

        # Thread-safe frame storage for the Web UI
        self.frame_lock = threading.Lock()
        self.current_jpeg = None

        Path(SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(KNOWN_FACES_DIR).mkdir(parents=True, exist_ok=True)

        self._register_msg       = ("", True)
        self._register_msg_until = 0.0

    def load_models(self):
        log.info("Loading YOLOv8 nano model...")
        self.yolo = YOLO(YOLO_MODEL)
        log.info("YOLO ready.")
        self._load_known_faces()

    def _load_known_faces(self):
        self.known_encodings = []
        self.known_names     = []
        supported = {".jpg", ".jpeg", ".png"}
        face_files = [f for f in Path(KNOWN_FACES_DIR).iterdir() if f.suffix.lower() in supported and not f.name.startswith(".")]
        for fpath in face_files:
            img  = face_recognition.load_image_file(str(fpath))
            encs = face_recognition.face_encodings(img)
            if encs:
                self.known_encodings.append(encs[0])
                self.known_names.append(fpath.stem)
        log.info(f"Known faces loaded: {self.known_names}")

    def reload_known_faces(self):
        self._load_known_faces()

    def _is_daytime(self) -> bool:
        return DAYTIME_START <= datetime.now().hour < DAYTIME_END

    def _track_id(self, bbox) -> str:
        x1, y1, x2, y2 = bbox
        return f"{round((x1 + x2) / 2 / 40) * 40}_{round((y1 + y2) / 2 / 40) * 40}"

    def _loiter_seconds(self, track_id: str) -> float:
        if track_id not in self._loiter:
            self._loiter[track_id] = time.time()
        return time.time() - self._loiter[track_id]

    def _cleanup_loiter(self, active_ids: set):
        for k in [k for k in self._loiter if k not in active_ids]:
            del self._loiter[k]

    def _face_covered(self, face_locs, person_bbox) -> bool:
        x1, y1, x2, y2 = person_bbox
        body_area = max(1, (x2 - x1) * (y2 - y1))
        for (top, right, bottom, left) in face_locs:
            if x1 <= (left + right) / 2 <= x2 and y1 <= (top + bottom) / 2 <= y2:
                if ((right - left) * (bottom - top)) / body_area >= MIN_FACE_BODY_RATIO:
                    return False
        return True

    def _classify(self, frame, persons) -> list:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb, model="hog")
        face_encs = face_recognition.face_encodings(rgb, face_locs)

        is_day = self._is_daytime()
        unknown_count = 0
        results = []
        active_ids = set()

        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            track_id = self._track_id(person["bbox"])
            active_ids.add(track_id)
            loiter_secs = self._loiter_seconds(track_id)

            matched_name = None
            is_known = False
            face_detected = False

            for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                if x1 <= (left + right) / 2 <= x2 and y1 <= (top + bottom) / 2 <= y2:
                    face_detected = True
                    if self.known_encodings:
                        dists = face_recognition.face_distance(self.known_encodings, enc)
                        best = int(np.argmin(dists))
                        if dists[best] <= FACE_TOLERANCE:
                            matched_name = self.known_names[best]
                            is_known = True

            covered = self._face_covered(face_locs, person["bbox"])
            loitering = loiter_secs >= LOITER_SECONDS
            signals = []
            risk = 0

            if is_known:
                threat, risk = Threat.KNOWN, 0
            elif covered:
                threat, risk = Threat.INTRUDER, 95
                signals.append("face covered")
            elif not face_detected:
                threat, risk = Threat.SUSPICIOUS, 60
                signals.append("face not visible")
            else:
                threat, risk = Threat.VISITOR, THREAT_BASE_SCORE[Threat.VISITOR]
                if not is_day:
                    risk += 30
                    signals.append("nighttime")
                    threat = Threat.SUSPICIOUS
                if loitering:
                    risk += 20
                    signals.append(f"loitering {loiter_secs:.0f}s")
                    if threat != Threat.INTRUDER: threat = Threat.SUSPICIOUS
                if not is_day and loitering:
                    threat, risk = Threat.INTRUDER, min(100, risk + 15)
                    signals.append("night+loiter")

            if not is_known: unknown_count += 1

            results.append({
                "bbox": person["bbox"], "confidence": person["confidence"],
                "name": matched_name or ("Unknown" if face_detected else "Face Hidden"),
                "is_known": is_known, "threat": threat, "risk_score": min(100, risk),
                "signals": signals, "loiter_secs": round(loiter_secs, 1),
                "face_covered": covered, "face_detected": face_detected, "is_daytime": is_day
            })

        if unknown_count >= 2:
            for d in results:
                if not d["is_known"] and d["threat"] == Threat.VISITOR:
                    d["threat"], d["risk_score"] = Threat.SUSPICIOUS, min(100, d["risk_score"] + 25)
                    d["signals"].append(f"group ({unknown_count} unknowns)")

        self._cleanup_loiter(active_ids)
        return results

    def _maybe_alert(self, frame, detections: list):
        alertable = [d for d in detections if d["threat"] in ALERT_LEVELS]
        if not alertable: return

        if any(d["threat"] == Threat.INTRUDER for d in alertable):
            highest = Threat.INTRUDER
        elif any(d["threat"] == Threat.SUSPICIOUS for d in alertable):
            highest = Threat.SUSPICIOUS
        else:
            highest = Threat.VISITOR

        now = time.time()
        if now - self._last_alert[highest] < ALERT_COOLDOWN_SEC: return
        self._last_alert[highest] = now

        for d in alertable:
            if d["threat"] == Threat.INTRUDER: self.stats["intruders"] += 1
            elif d["threat"] == Threat.SUSPICIOUS: self.stats["suspicious"] += 1
            elif d["threat"] == Threat.VISITOR: self.stats["visitors"] += 1

        snapshot_name = f"{highest.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_name)
        cv2.imwrite(snapshot_path, frame)

        if self.mqtt_client:
            payload = json.dumps({"type": "CAMERA_ALERT", "threat_level": highest, "timestamp": datetime.now().isoformat(), "snapshot": snapshot_path})
            self.mqtt_client.publish(CAMERA_TOPIC, payload)

        if self._on_alert_callback:
            try: self._on_alert_callback(highest)
            except: pass
        alertable = [d for d in detections if d["threat"] in ALERT_LEVELS]
        if not alertable: return

        highest = Threat.INTRUDER if any(d["threat"] == Threat.INTRUDER for d in alertable) else Threat.SUSPICIOUS
        now = time.time()
        if now - self._last_alert[highest] < ALERT_COOLDOWN_SEC: return
        self._last_alert[highest] = now

        for d in alertable:
            if d["threat"] == Threat.INTRUDER: self.stats["intruders"] += 1
            elif d["threat"] == Threat.SUSPICIOUS: self.stats["suspicious"] += 1

        snapshot_name = f"{highest.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_name)
        cv2.imwrite(snapshot_path, frame)

        if self.mqtt_client:
            payload = json.dumps({"type": "CAMERA_ALERT", "threat_level": highest, "timestamp": datetime.now().isoformat(), "snapshot": snapshot_path})
            self.mqtt_client.publish(CAMERA_TOPIC, payload)

        if self._on_alert_callback:
            try: self._on_alert_callback(highest)
            except: pass

    def _draw_overlay(self, frame, detections: list):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            colour = THREAT_COLOUR[d["threat"]]
            
            # Calculate center coordinates and radius for the circle
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            # Make the radius large enough to encompass the person's upper body/face
            radius = int(max(x2 - x1, y2 - y1) / 2) + 10
            
            # Draw the circle (thickness 3 for intruders, 2 for others)
            thickness = 3 if d["threat"] == Threat.INTRUDER else 2
            cv2.circle(frame, (cx, cy), radius, colour, thickness)
            
            # Draw the classification label above the circle
            label = f"{d['threat']} | Risk: {d['risk_score']} | {d['name']}"
            
            # Background block for text readability
            cv2.rectangle(frame, (cx - radius, cy - radius - 28), (cx + radius, cy - radius), colour, -1)
            cv2.putText(frame, label, (cx - radius + 4, cy - radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
        return frame

    def start(self, show_window: bool = True):
        self.load_models()
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.running = True
        frame_count = 0
        detections = [] # <--- 1. ADD THIS HERE

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret: continue

                frame_count += 1
                
                # Only run heavy ML every N frames
                if frame_count % PROCESS_EVERY_N == 0:
                    persons = self._detect_persons(frame)
                    detections = self._classify(frame, persons)
                    self._maybe_alert(frame.copy(), detections)
                
                # <--- 2. WE REMOVED THE "else: detections = []" FROM HERE
                # Now it will just use the old detections for the in-between frames!

                annotated = self._draw_overlay(frame.copy(), detections)
                
                # Encode frame for Web UI
                ret_jpg, buffer = cv2.imencode('.jpg', annotated)
                if ret_jpg:
                    with self.frame_lock:
                        self.current_jpeg = buffer.tobytes()

                if show_window:
                    cv2.imshow("Fog Threat Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

        finally:
            self.running = False
            cap.release()
            if show_window: cv2.destroyAllWindows()

    def _detect_persons(self, frame) -> list:
        results = self.yolo(frame, verbose=False)[0]
        persons = []
        
        for b in results.boxes:
            if int(b.cls[0]) == 0 and float(b.conf[0]) >= YOLO_CONFIDENCE:
                # 1. Extract tensor and force it into a standard Python list
                coords = b.xyxy[0].tolist()
                
                # 2. Hardcast each coordinate to a standard integer
                x1 = int(coords[0])
                y1 = int(coords[1])
                x2 = int(coords[2])
                y2 = int(coords[3])
                
                # 3. Save as a standard immutable tuple
                persons.append({
                    "bbox": (x1, y1, x2, y2), 
                    "confidence": float(b.conf[0])
                })
                
        return persons

    def get_video_frame(self):
        """Returns the latest encoded JPEG for the Flask web stream."""
        with self.frame_lock:
            return self.current_jpeg

    def stop(self):
        self.running = False