import cv2
import json
import time
import os
import logging
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

# Colour per threat level (BGR)
THREAT_COLOUR = {
    Threat.KNOWN:      (0,   200,   0),   # green
    Threat.VISITOR:    (200, 150,   0),   # blue
    Threat.SUSPICIOUS: (0,   200, 255),   # yellow
    Threat.INTRUDER:   (0,     0, 255),   # red
}

# Base risk score per level (refined further by signals)
THREAT_BASE_SCORE = {
    Threat.KNOWN:      0,
    Threat.VISITOR:    20,
    Threat.SUSPICIOUS: 55,
    Threat.INTRUDER:   90,
}

# Only alert for these levels
ALERT_LEVELS = {Threat.SUSPICIOUS, Threat.INTRUDER}

# Loiter threshold — if unknown person stays in frame this long → escalate
LOITER_SECONDS = 10

# Day hours (inclusive). Outside this range = night.
DAYTIME_START = 6    # 06:00
DAYTIME_END   = 20   # 20:00

# Face-area ratio: if YOLO detects a person but detected face bbox
# is smaller than this fraction of body bbox → likely face covered
MIN_FACE_BODY_RATIO = 0.04


class CameraGuard:
    """
    Webcam pipeline with multi-signal threat classification.

    Signals evaluated per detection:
      • Face recognition  — known / unknown / undetectable
      • Time of day       — daytime vs nighttime
      • Loiter tracking   — how long unknown has been in frame
      • Group detection   — multiple unknowns simultaneously
      • Face visibility   — face covered / obscured
    """

    def __init__(self, mqtt_client, on_alert_callback=None, vibration_engine=None):
        self.mqtt_client        = mqtt_client
        self._on_alert_callback  = on_alert_callback  # called when camera alert fires
        self._vibration_engine  = vibration_engine    # for on-screen warnings
        self.yolo            = None
        self.known_encodings = []
        self.known_names     = []
        self.running         = False

        # Per-person loiter tracking: track_id → first_seen timestamp
        # We use a simple spatial hash (rounded bbox center) as a proxy ID
        self._loiter: dict[str, float] = {}

        # Cooldown per threat level (seconds between alerts of same type)
        self._last_alert: dict[str, float] = defaultdict(float)

        self.stats = {
            "frames_processed":  0,
            "persons_detected":  0,
            "known_seen":        0,
            "visitors":          0,
            "suspicious":        0,
            "intruders":         0,
        }

        Path(SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(KNOWN_FACES_DIR).mkdir(parents=True, exist_ok=True)

        # Registration feedback banner state
        self._register_msg       = ("", True)
        self._register_msg_until = 0.0

    # ── MODEL LOADING ─────────────────────────────────────────

    def load_models(self):
        log.info("Loading YOLOv8 nano model...")
        self.yolo = YOLO(YOLO_MODEL)
        log.info("YOLO ready.")
        self._load_known_faces()

    def _load_known_faces(self):
        self.known_encodings = []
        self.known_names     = []
        supported = {".jpg", ".jpeg", ".png"}
        face_files = [
            f for f in Path(KNOWN_FACES_DIR).iterdir()
            if f.suffix.lower() in supported and not f.name.startswith(".")
        ]
        if not face_files:
            log.warning(
                f"No photos in '{KNOWN_FACES_DIR}/'. "
                "Add photos like kshitij.jpg to register known people. "
                "Until then unknown faces will be classified by context."
            )
            return
        for fpath in face_files:
            img  = face_recognition.load_image_file(str(fpath))
            encs = face_recognition.face_encodings(img)
            if encs:
                self.known_encodings.append(encs[0])
                self.known_names.append(fpath.stem)
                log.info(f"  Loaded face: {fpath.stem}")
            else:
                log.warning(
                    f"  No face detected in {fpath.name} — skipping. "
                    "Use a clear front-facing portrait photo."
                )
        log.info(f"Known faces loaded: {self.known_names}")

    def reload_known_faces(self):
        log.info("Reloading known faces...")
        self._load_known_faces()

    # ── HELPERS ───────────────────────────────────────────────

    def _is_daytime(self) -> bool:
        hour = datetime.now().hour
        return DAYTIME_START <= hour < DAYTIME_END

    def _track_id(self, bbox) -> str:
        """Rough spatial ID — centre of bbox rounded to 40px grid."""
        x1, y1, x2, y2 = bbox
        cx = round((x1 + x2) / 2 / 40) * 40
        cy = round((y1 + y2) / 2 / 40) * 40
        return f"{cx}_{cy}"

    def _loiter_seconds(self, track_id: str) -> float:
        if track_id not in self._loiter:
            self._loiter[track_id] = time.time()
        return time.time() - self._loiter[track_id]

    def _cleanup_loiter(self, active_ids: set):
        """Remove stale loiter entries for people no longer in frame."""
        stale = [k for k in self._loiter if k not in active_ids]
        for k in stale:
            del self._loiter[k]

    def _face_covered(self, face_locs, person_bbox) -> bool:
        """
        Returns True if no face was found inside the person bounding box,
        suggesting the face is covered or turned away.
        We only flag this when the person detection is high confidence
        (confident there IS a person) but no face is detected.
        """
        x1, y1, x2, y2 = person_bbox
        body_area = max(1, (x2 - x1) * (y2 - y1))
        for (top, right, bottom, left) in face_locs:
            # Face centre inside person bbox?
            fc_x = (left + right) / 2
            fc_y = (top + bottom) / 2
            if x1 <= fc_x <= x2 and y1 <= fc_y <= y2:
                face_area = (right - left) * (bottom - top)
                if face_area / body_area >= MIN_FACE_BODY_RATIO:
                    return False   # valid visible face found
        return True   # no detectable face inside person bbox

    # ── CORE CLASSIFICATION ───────────────────────────────────

    def _classify(self, frame, persons) -> list:
        """
        Run face recognition + multi-signal threat scoring on each person.
        Returns list of detection dicts with threat_level + risk_score.
        """
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb, model="hog")
        face_encs = face_recognition.face_encodings(rgb, face_locs)

        is_day        = self._is_daytime()
        unknown_count = 0
        results       = []
        active_ids    = set()

        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            track_id        = self._track_id(person["bbox"])
            active_ids.add(track_id)
            loiter_secs     = self._loiter_seconds(track_id)

            # ── Step 1: Face recognition ──────────────────────
            matched_name  = None
            is_known      = False
            face_detected = False

            for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                fc_x = (left + right) / 2
                fc_y = (top + bottom) / 2
                if x1 <= fc_x <= x2 and y1 <= fc_y <= y2:
                    face_detected = True
                    if self.known_encodings:
                        dists    = face_recognition.face_distance(self.known_encodings, enc)
                        best     = int(np.argmin(dists))
                        if dists[best] <= FACE_TOLERANCE:
                            matched_name = self.known_names[best]
                            is_known     = True

            # ── Step 2: Signals ───────────────────────────────
            covered     = self._face_covered(face_locs, person["bbox"])
            loitering   = loiter_secs >= LOITER_SECONDS

            # ── Step 3: Threat classification ─────────────────
            signals = []
            risk    = 0

            if is_known:
                threat = Threat.KNOWN
                risk   = 0

            elif covered:
                # Person detected but no face visible — highest risk
                threat = Threat.INTRUDER
                risk   = 95
                signals.append("face covered")

            elif not face_detected and not covered:
                # Face turned away — slightly lower but still suspicious
                threat = Threat.SUSPICIOUS
                risk   = 60
                signals.append("face not visible")

            else:
                # Unknown face visible — assess context
                threat = Threat.VISITOR   # start optimistic
                risk   = THREAT_BASE_SCORE[Threat.VISITOR]

                if not is_day:
                    risk   += 30
                    signals.append("nighttime")
                    threat  = Threat.SUSPICIOUS

                if loitering:
                    risk   += 20
                    signals.append(f"loitering {loiter_secs:.0f}s")
                    if threat != Threat.INTRUDER:
                        threat = Threat.SUSPICIOUS

                if not is_day and loitering:
                    # Nighttime + loitering = intruder
                    threat = Threat.INTRUDER
                    risk   = min(100, risk + 15)
                    signals.append("night+loiter")

            if not is_known:
                unknown_count += 1

            risk = min(100, risk)

            results.append({
                "bbox":         person["bbox"],
                "confidence":   person["confidence"],
                "name":         matched_name or ("Unknown" if face_detected else "Face Hidden"),
                "is_known":     is_known,
                "threat":       threat,
                "risk_score":   risk,
                "signals":      signals,
                "loiter_secs":  round(loiter_secs, 1),
                "face_covered": covered,
                "face_detected": face_detected,
                "is_daytime":   is_day,
            })

        # ── Step 4: Group escalation ──────────────────────────
        # Multiple unknowns at once → escalate all unknowns to at least SUSPICIOUS
        if unknown_count >= 2:
            for d in results:
                if not d["is_known"] and d["threat"] == Threat.VISITOR:
                    d["threat"]     = Threat.SUSPICIOUS
                    d["risk_score"] = min(100, d["risk_score"] + 25)
                    d["signals"].append(f"group ({unknown_count} unknowns)")

        self._cleanup_loiter(active_ids)
        return results

    # ── ALERT ─────────────────────────────────────────────────

    def _maybe_alert(self, frame, detections: list):
        """Fire alert for SUSPICIOUS and INTRUDER detections, with per-level cooldown."""
        alertable = [d for d in detections if d["threat"] in ALERT_LEVELS]
        if not alertable:
            return

        # Use the highest threat level present
        highest = (
            Threat.INTRUDER
            if any(d["threat"] == Threat.INTRUDER for d in alertable)
            else Threat.SUSPICIOUS
        )

        now = time.time()
        if now - self._last_alert[highest] < ALERT_COOLDOWN_SEC:
            return

        self._last_alert[highest] = now

        # Update stats
        for d in alertable:
            if d["threat"] == Threat.INTRUDER:
                self.stats["intruders"] += 1
            elif d["threat"] == Threat.SUSPICIOUS:
                self.stats["suspicious"] += 1

        # Save snapshot
        timestamp     = datetime.now().isoformat()
        snapshot_name = f"{highest.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_name)
        cv2.imwrite(snapshot_path, frame)

        payload = {
            "type":        "CAMERA_ALERT",
            "threat_level": highest,
            "timestamp":   timestamp,
            "source":      "camera_guard",
            "alert_level": "CRITICAL" if highest == Threat.INTRUDER else "WARNING",
            "detections": [
                {
                    "name":       d["name"],
                    "threat":     d["threat"],
                    "risk_score": d["risk_score"],
                    "signals":    d["signals"],
                    "loiter_secs": d["loiter_secs"],
                }
                for d in alertable
            ],
            "snapshot":    snapshot_path,
            "is_daytime":  self._is_daytime(),
        }

        if self.mqtt_client:
            self.mqtt_client.publish(CAMERA_TOPIC,  json.dumps(payload))
            self.mqtt_client.publish(RESULTS_TOPIC, json.dumps(payload))

        log.warning(
            f"{highest} ALERT | "
            f"{len(alertable)} person(s) | "
            f"Signals: {[s for d in alertable for s in d['signals']]} | "
            f"Snapshot → {snapshot_path}"
        )

        # Notify vibration engine for cross-sensor correlation
        if self._on_alert_callback:
            try:
                self._on_alert_callback(highest)
            except Exception as e:
                log.debug(f"Alert callback error: {e}")

    # ── REGISTER UNKNOWN AS KNOWN ─────────────────────────────

    def _register_unknown(self, frame, detections: list):
        """
        Called when user presses R.
        Instantly saves the unknown person closest to frame centre
        as a known person. Auto-named: person_YYYYMMDD_HHMMSS
        Takes effect immediately — no terminal input, no restart.
        """
        h, w = frame.shape[:2]
        frame_cx, frame_cy = w // 2, h // 2

        unknowns = [
            d for d in detections
            if not d["is_known"] and d["face_detected"] and not d["face_covered"]
        ]

        if not unknowns:
            log.info("Register: no unknown face visible to register.")
            self._register_msg       = ("No unknown face in view", False)
            self._register_msg_until = time.time() + 3
            return

        # Pick the unknown closest to frame centre
        def dist_to_centre(d):
            x1, y1, x2, y2 = d["bbox"]
            return (((x1+x2)/2 - frame_cx)**2 + ((y1+y2)/2 - frame_cy)**2) ** 0.5

        target   = min(unknowns, key=dist_to_centre)
        x1, y1, x2, y2 = target["bbox"]

        # Crop with padding for a better face photo
        pad  = 20
        crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]

        # Verify face can be encoded before saving
        encs = face_recognition.face_encodings(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not encs:
            log.warning("Register: could not encode face — try better lighting.")
            self._register_msg       = ("Encoding failed — try better lighting", False)
            self._register_msg_until = time.time() + 3
            return

        # Auto-name: person_YYYYMMDD_HHMMSS
        auto_name = f"person_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_path = os.path.join(KNOWN_FACES_DIR, f"{auto_name}.jpg")
        cv2.imwrite(save_path, crop)
        log.info(f"Register: saved → {save_path}")

        self.reload_known_faces()
        log.info(f"Register: {auto_name} is now a known person.")

        self._register_msg       = (f"Registered as: {auto_name}", True)
        self._register_msg_until = time.time() + 4

    # ── DRAW OVERLAY ──────────────────────────────────────────

    def _draw_overlay(self, frame, detections: list):
        h, w = frame.shape[:2]

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            threat  = d["threat"]
            colour  = THREAT_COLOUR[threat]
            risk    = d["risk_score"]
            loiter  = d["loiter_secs"]

            # Bounding box — thicker for higher threats
            thickness = 3 if threat == Threat.INTRUDER else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

            # Label bar
            label = f"{threat}  risk:{risk}  {d['name']}"
            if d["signals"]:
                label += f"  [{', '.join(d['signals'])}]"

            cv2.rectangle(frame, (x1, y1 - 28), (x2, y1), colour, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Loiter bar — fills left-to-right as person loiters
            if not d["is_known"] and loiter > 0:
                bar_w   = int((min(loiter, LOITER_SECONDS) / LOITER_SECONDS) * (x2 - x1))
                bar_col = (0, 200, 255) if loiter < LOITER_SECONDS else (0, 0, 255)
                cv2.rectangle(frame, (x1, y2), (x1 + bar_w, y2 + 4), bar_col, -1)
                cv2.putText(frame, f"{loiter:.0f}s", (x1 + 2, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_col, 1)

        # ── Status bar (bottom strip) ──────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        time_label = "DAY" if self._is_daytime() else "NIGHT"
        status = (
            f"Fog Node  |  {time_label}  |  "
            f"Known:{self.stats['known_seen']}  "
            f"Visitors:{self.stats['visitors']}  "
            f"Suspicious:{self.stats['suspicious']}  "
            f"Intruders:{self.stats['intruders']}"
        )
        cv2.putText(frame, status, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        # Legend (top-right)
        legend = [
            (Threat.KNOWN,      "Known person"),
            (Threat.VISITOR,    "Visitor (unknown, day)"),
            (Threat.SUSPICIOUS, "Suspicious"),
            (Threat.INTRUDER,   "Intruder"),
        ]
        for i, (level, desc) in enumerate(legend):
            col = THREAT_COLOUR[level]
            y   = 20 + i * 20
            cv2.rectangle(frame, (w - 210, y - 12), (w - 196, y + 2), col, -1)
            cv2.putText(frame, desc, (w - 190, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        # ── Vibration warning banner (top of frame) ──────────
        if self._vibration_engine:
            self._vibration_engine.tick_warning()
            if self._vibration_engine.warning_active:
                lvl = self._vibration_engine.warning_level
                txt = self._vibration_engine.warning_text
                # Banner colour by level
                banner_col = {
                    "LOW":      (0, 180, 255),
                    "MEDIUM":   (0, 100, 255),
                    "HIGH":     (0, 0, 255),
                    "CRITICAL": (0, 0, 200),
                }.get(lvl, (0, 0, 255))
                # Flashing effect — blink every 0.4s for CRITICAL/HIGH
                show = True
                if lvl in ("CRITICAL", "HIGH"):
                    show = int(time.time() * 2.5) % 2 == 0
                if show:
                    overlay2 = frame.copy()
                    cv2.rectangle(overlay2, (0, 0), (w, 52), banner_col, -1)
                    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
                    cv2.putText(frame, f"[SENSOR] {txt}",
                                (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.60, (255, 255, 255), 2)

        # ── Registration feedback banner ──────────────────────
        if time.time() < self._register_msg_until:
            msg, success = self._register_msg
            banner_col = (0, 160, 0) if success else (0, 0, 200)
            reg_overlay = frame.copy()
            cv2.rectangle(reg_overlay, (0, h//2 - 30), (w, h//2 + 30), banner_col, -1)
            cv2.addWeighted(reg_overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, f"[REGISTER] {msg}",
                        (20, h//2 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

        return frame

    # ── MAIN LOOP ─────────────────────────────────────────────

    def start(self, show_window: bool = True):
        self.load_models()

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            log.error("Cannot open webcam. Check CAMERA_INDEX in config.py.")
            return

        log.info(f"Webcam open ({FRAME_WIDTH}x{FRAME_HEIGHT}). Threat classification running.")
        self.running  = True
        frame_count   = 0
        detections    = []

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                frame_count                    += 1
                self.stats["frames_processed"] += 1

                if frame_count % PROCESS_EVERY_N == 0:
                    persons    = self._detect_persons(frame)
                    detections = self._classify(frame, persons)

                    if persons:
                        self.stats["persons_detected"] += len(persons)

                    for d in detections:
                        if d["is_known"]:
                            self.stats["known_seen"] += 1
                            log.info(f"Known: {d['name']}")
                        elif d["threat"] == Threat.VISITOR:
                            self.stats["visitors"] += 1

                    self._maybe_alert(frame.copy(), detections)

                annotated = self._draw_overlay(frame.copy(), detections)

                if show_window:
                    cv2.imshow("Fog Threat Detection  [Q to quit | R to register face]", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        log.info("Quit key pressed.")
                        break
                    elif key == ord("r"):
                        self._register_unknown(frame.copy(), detections)

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            cap.release()
            if show_window:
                cv2.destroyAllWindows()
            log.info(f"Camera stopped. Session stats: {self.stats}")

    def _detect_persons(self, frame) -> list:
        results = self.yolo(frame, verbose=False)[0]
        persons = []
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= YOLO_CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append({"bbox": (x1, y1, x2, y2), "confidence": float(box.conf[0])})
        return persons

    def stop(self):
        self.running = False