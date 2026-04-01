# ============================================================
#  dashboard/app.py  —  Real-Time Streamlit Dashboard
#
#  Subscribes to fog/results and renders two tabs:
#    Tab 1 — Camera: intruder snapshots + event log
#    Tab 2 — Sensors: live charts + anomaly log
#
#  Run: streamlit run dashboard/app.py  (from project root)
# ============================================================

import json
import os
import time
from collections import deque
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import paho.mqtt.client as mqtt
import streamlit as st
from PIL import Image

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fog Security Dashboard",
    page_icon="🌫️",
    layout="wide",
)

MQTT_BROKER   = "localhost"
MQTT_PORT     = 1883
RESULTS_TOPIC = "fog/results"
MAX_POINTS    = 100

# ── SESSION STATE DEFAULTS ────────────────────────────────────
_defaults = {
    "sensor_buf":       deque(maxlen=MAX_POINTS),
    "sensor_anomalies": deque(maxlen=50),
    "camera_alerts":    deque(maxlen=20),
    "total_sensor_rx":  0,
    "total_sensor_anm": 0,
    "total_intruders":  0,
    "mqtt_ok":          False,
    "latest_snapshot":  None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── MQTT (cached singleton) ────────────────────────────────────
@st.cache_resource
def _get_mqtt():
    client = mqtt.Client(client_id="streamlit_dashboard")

    def on_connect(c, _u, _f, rc):
        if rc == 0:
            c.subscribe(RESULTS_TOPIC)
            st.session_state.mqtt_ok = True

    def on_message(_c, _u, msg):
        try:
            data     = json.loads(msg.payload.decode())
            msg_type = data.get("type", "SENSOR")

            if msg_type == "INTRUDER_ALERT":
                st.session_state.camera_alerts.appendleft(data)
                st.session_state.total_intruders += 1
                snap = data.get("snapshot")
                if snap and os.path.exists(snap):
                    st.session_state.latest_snapshot = snap

            else:   # SENSOR message
                data["_ts"] = datetime.now().isoformat()
                st.session_state.sensor_buf.append(data)
                st.session_state.total_sensor_rx += 1
                if data.get("anomaly"):
                    st.session_state.total_sensor_anm += 1
                    st.session_state.sensor_anomalies.appendleft(data)

        except Exception:
            pass

    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
        st.session_state.mqtt_ok = True
    except Exception:
        st.session_state.mqtt_ok = False
    return client


_get_mqtt()   # ensure connection on every rerun

# ── HEADER ────────────────────────────────────────────────────
st.title("🌫️ Fog Security & Anomaly Dashboard")
st.caption("Laptop Fog Node  |  Sensor Anomaly Detection  +  Camera Intruder Detection")

# Status bar
s1, s2, _ = st.columns([1, 1, 4])
with s1:
    st.success("🟢 MQTT Online") if st.session_state.mqtt_ok else st.error("🔴 MQTT Offline")
with s2:
    if st.button("🔄 Refresh now"):
        st.rerun()

# ── GLOBAL ALERT BANNER ───────────────────────────────────────
if st.session_state.total_intruders:
    latest_cam = list(st.session_state.camera_alerts)[0]
    st.error(
        f"🚨 **INTRUDER DETECTED**  |  "
        f"{latest_cam.get('timestamp','')}  |  "
        f"Alert #{st.session_state.total_intruders}"
    )

# ── KPI ROW ───────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
buf    = list(st.session_state.sensor_buf)
latest = buf[-1].get("features", {}) if buf else {}

k1.metric("📦 Sensor Packets",   st.session_state.total_sensor_rx)
k2.metric("⚠️ Sensor Anomalies", st.session_state.total_sensor_anm)
k3.metric("🚨 Intruder Alerts",  st.session_state.total_intruders,
          delta="CRITICAL" if st.session_state.total_intruders else None,
          delta_color="inverse")
k4.metric("🌡️ Latest Temp (°C)", latest.get("temperature", "--"))

st.divider()

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_cam, tab_sensor = st.tabs(
    ["📷  Camera — Intruder Detection", "📡  Sensors — Anomaly Detection"]
)

# ─────────────────────────────────────────────────────────────
# TAB 1 — CAMERA
# ─────────────────────────────────────────────────────────────
with tab_cam:
    st.subheader("📷 Intruder Detection Feed")
    st.caption("YOLOv8 person detection → face_recognition (Known vs Unknown)")

    col_snap, col_log = st.columns(2)

    with col_snap:
        st.markdown("**Latest Intrusion Snapshot**")
        snap = st.session_state.latest_snapshot
        if snap and os.path.exists(snap):
            st.image(Image.open(snap), caption=snap, use_column_width=True)
        else:
            st.info(
                "No intrusion snapshot yet.\n\n"
                "When an unknown person is detected the saved frame "
                "will appear here automatically."
            )

    with col_log:
        st.markdown("**Intrusion Event Log**")
        alerts = list(st.session_state.camera_alerts)
        if alerts:
            rows = []
            for a in alerts:
                names = ", ".join(
                    d.get("name", "Unknown")
                    for d in a.get("detections", [])
                ) or "Unknown"
                rows.append({
                    "Timestamp":   a.get("timestamp", ""),
                    "Person(s)":   names,
                    "Snapshot":    os.path.basename(a.get("snapshot", "")),
                    "Alert Level": a.get("alert_level", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.success("✅ No intrusions detected yet.")

    with st.expander("ℹ️ How Camera Detection Works"):
        st.markdown("""
**Pipeline (all on laptop — Fog layer):**

1. Webcam captures frames continuously
2. **YOLOv8 nano** runs every 5 frames — detects if a *person* is present
3. **face_recognition** compares face against photos in `fog/known_faces/`
   - Match → ✅ Known (green box, name shown, no alert)
   - No match → 🚨 Intruder (red box, snapshot saved, MQTT alert fired)
4. Alert published to `fog/results` → appears here in real-time
5. MQTT `ALERT` command sent to ESP32 → LED + buzzer fires

**To add team members:** drop a photo named `yourname.jpg` in `fog/known_faces/` and restart the engine.
        """)

# ─────────────────────────────────────────────────────────────
# TAB 2 — SENSORS
# ─────────────────────────────────────────────────────────────
with tab_sensor:
    st.subheader("📡 Sensor Anomaly Detection")
    st.caption("Isolation Forest ML model running live on ESP32 data")

    if buf:
        df = pd.DataFrame([
            {
                "time":        d.get("_ts", ""),
                "temperature": d.get("features", {}).get("temperature", 0),
                "humidity":    d.get("features", {}).get("humidity", 0),
                "accel_mag":   d.get("features", {}).get("accel_mag", 0),
                "score":       d.get("score") or 0,
                "anomaly":     d.get("anomaly", False),
            }
            for d in buf
        ])
        anm_df = df[df["anomaly"]]

        c1, c2 = st.columns(2)

        # Temperature chart
        with c1:
            st.markdown("**🌡️ Temperature (°C)**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["time"], y=df["temperature"],
                mode="lines+markers", name="Temp",
                line=dict(color="#FF6B6B"),
            ))
            if not anm_df.empty:
                fig.add_trace(go.Scatter(
                    x=anm_df["time"], y=anm_df["temperature"],
                    mode="markers", name="Anomaly",
                    marker=dict(color="red", size=12, symbol="x"),
                ))
            fig.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Humidity chart
        with c2:
            st.markdown("**💧 Humidity (%)**")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["time"], y=df["humidity"],
                mode="lines+markers", name="Humidity",
                line=dict(color="#4ECDC4"),
            ))
            if not anm_df.empty:
                fig2.add_trace(go.Scatter(
                    x=anm_df["time"], y=anm_df["humidity"],
                    mode="markers", name="Anomaly",
                    marker=dict(color="red", size=12, symbol="x"),
                ))
            fig2.update_layout(height=240, margin=dict(l=0, r=0, t=8, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Anomaly score chart
        st.markdown("**📉 Isolation Forest Anomaly Score**")
        st.caption("More negative = more anomalous. Red dashed line = alert threshold.")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["time"], y=df["score"],
            mode="lines+markers", fill="tozeroy", name="Score",
            line=dict(color="#A855F7"),
        ))
        fig3.add_hline(
            y=-0.1, line_dash="dash", line_color="red",
            annotation_text="Anomaly Threshold",
        )
        fig3.update_layout(height=200, margin=dict(l=0, r=0, t=8, b=0))
        st.plotly_chart(fig3, use_container_width=True)

        # Anomaly log table
        st.markdown("**📋 Sensor Anomaly Log**")
        anm_list = list(st.session_state.sensor_anomalies)
        if anm_list:
            rows = []
            for a in anm_list:
                f = a.get("features", {})
                rows.append({
                    "Timestamp":    a.get("timestamp", ""),
                    "Device":       a.get("device_id", ""),
                    "Temp (°C)":    f.get("temperature", ""),
                    "Humidity (%)": f.get("humidity", ""),
                    "Accel Mag":    f.get("accel_mag", ""),
                    "Score": f"{a['score']:.4f}" if a.get("score") else "N/A",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.success("✅ No sensor anomalies logged yet.")

    else:
        st.info("⏳ Waiting for ESP32 data… (or run `python tools/simulator.py`)")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🌫️ System Status")
    st.markdown("""
**Active Pipelines:**
- 📡 Sensor Anomaly (Isolation Forest)
- 📷 Camera Intruder (YOLO + FaceRec)

**Fog Node:** Laptop
**Edge Node:** ESP32
**Cloud:** *Not connected — future phase*
    """)
    st.divider()
    st.metric("Sensor packets",   st.session_state.total_sensor_rx)
    st.metric("Sensor anomalies", st.session_state.total_sensor_anm)
    st.metric("Intruder alerts",  st.session_state.total_intruders)
    st.divider()
    auto = st.checkbox("Auto-refresh every 5s", value=True)

# ── AUTO-REFRESH ──────────────────────────────────────────────
if auto:
    time.sleep(5)
    st.rerun()
