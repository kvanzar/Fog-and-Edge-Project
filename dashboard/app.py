import os
import time
import threading
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from sklearn.linear_model import LinearRegression
from fog.camera_guard import CameraGuard

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

KNOWN_FACES_DIR = os.path.abspath(os.path.join(current_dir, '..', 'fog', 'known_faces'))
SNAPSHOTS_DIR = os.path.abspath(os.path.join(current_dir, '..', 'fog', 'intruder_snapshots'))
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

current_temp = 0.0
current_hum = 0.0
temp_history = []

camera = CameraGuard()
threading.Thread(target=camera.start, kwargs={"show_window": False}, daemon=True).start()

def gen_frames():
    while True:
        frame = camera.get_video_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sensor', methods=['POST'])
def receive_sensor_data():
    global current_temp, current_hum, temp_history
    data = request.json
    current_temp = float(data.get("temperature", current_temp))
    current_hum = float(data.get("humidity", current_hum))
    temp_history.append((time.time(), current_temp))
    if len(temp_history) > 50: temp_history.pop(0)
    return jsonify({"status": "ok"})

@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOTS_DIR, filename)

@app.route('/api/mark_known', methods=['POST'])
def mark_known():
    data = request.json
    old_path = os.path.join(SNAPSHOTS_DIR, data.get("filename"))
    new_path = os.path.join(KNOWN_FACES_DIR, f"{data.get('name').replace(' ', '_')}.jpg")
    if os.path.exists(old_path): os.rename(old_path, new_path)
    camera.reload_known_faces()
    return jsonify({"status": "success"})

@app.route('/api/mark_unknown', methods=['POST'])
def mark_unknown():
    old_path = os.path.join(SNAPSHOTS_DIR, request.json.get("filename"))
    if os.path.exists(old_path): os.remove(old_path)
    return jsonify({"status": "success"})

@app.route('/api/data')
def get_data():
    alerts_log = []
    try:
        for f in sorted(os.listdir(SNAPSHOTS_DIR), reverse=True)[:15]:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                parts = f.replace('.jpg', '').replace('.png', '').split('_')
                threat_type = parts[0].capitalize() if len(parts) > 0 else "Alert"
                time_str = f"{parts[1]} {parts[2][:2]}:{parts[2][2:4]}" if len(parts) >= 3 else "Recent"
                alerts_log.append({"filename": f, "time": time_str, "type": threat_type})
    except Exception: pass

    suggestion, prediction = "Collecting data...", current_temp
    if len(temp_history) > 5:
        X = np.array([t[0] for t in temp_history]).reshape(-1, 1)
        model = LinearRegression().fit(X, np.array([t[1] for t in temp_history]))
        prediction = model.predict(np.array([[time.time() + 3600]]))[0]
        if prediction > 32: suggestion = "Temperature rising. Not a good time to go out."
        elif prediction < 22: suggestion = "Cool weather approaching. Great time to go out."
        else: suggestion = "Weather is stable and optimal."
    
    try: aqi = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality?latitude=12.9165&longitude=79.1325&current=us_aqi", timeout=2).json()['current']['us_aqi']
    except: aqi = "N/A"

    return jsonify({"temperature": current_temp, "humidity": current_hum, "aqi": aqi, "prediction": round(prediction, 1), "suggestion": suggestion, "alerts": alerts_log})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)