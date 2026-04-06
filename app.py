import os
import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from flask import Flask, request, send_from_directory
from datetime import datetime

# Project-relative paths (EB runs from project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
OUTPUT_DIR = os.path.join(BACKEND_DIR, "output")
RESULT_HTML_PATH = os.path.join(FRONTEND_DIR, "result.html")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# Load models/data (keep files in project root)
ensemble_model = joblib.load(os.path.join(BASE_DIR, "ensemble_sensor_model.pkl"))
sensor_data = pd.read_csv(os.path.join(BASE_DIR, "fused_sensor_training_data_high_altitude.csv"))
yolo_model = YOLO("yolov8l.pt")

@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.get("/output/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.post("/")
def process_image():
    if "model-file" not in request.files:
        return "No image uploaded", 400

    file = request.files["model-file"]
    if file.filename == "":
        return "Empty filename", 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return "Invalid image", 400

    results = yolo_model(image, classes=[0], conf=0.15, imgsz=1280)
    boxes = results[0].boxes
    if not boxes:
        boxes = []

    confirmed = False

    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0]) * 100
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

            box_data = sensor_data[
                (sensor_data['Pixel_X'] >= x) & (sensor_data['Pixel_X'] <= x + w) &
                (sensor_data['Pixel_Y'] >= y) & (sensor_data['Pixel_Y'] <= y + h)
            ]

            if not box_data.empty:
                features = box_data[['Pixel_X','Pixel_Y','Depth_mm','Temperature_C','ToF_Amplitude']]
                predictions = ensemble_model.predict(features)
                human_pixel_count = (predictions == 'Victim').sum()

                if human_pixel_count > 0:
                    confirmed = True
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image, f"VERIFIED VICTIM {confidence:.1f}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(image, "POTENTIAL VICTIM", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 1)
                cv2.putText(image, "NO SENSOR DATA", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"output_{ts}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, image)

    msg_text = "Confirmed: Victim Identified" if confirmed else "No victims found"
    msg_color = "#ff4d4d" if confirmed else "#41d07f"
    image_url = f"/output/{out_name}"

    with open(RESULT_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("{MSG_TEXT}", msg_text)
    html = html.replace("{MSG_COLOR}", msg_color)
    html = html.replace("{IMAGE_URL}", image_url)

    return html

if __name__ == "__main__":
    # EB sets PORT automatically
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
