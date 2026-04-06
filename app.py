import os
import cv2
import numpy as np
import pandas as pd
import joblib
import boto3
import tempfile
from ultralytics import YOLO
from flask import Flask, request, send_from_directory, abort
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
RESULT_HTML_PATH = os.path.join(FRONTEND_DIR, "result.html")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Load models/data
ensemble_model = joblib.load(os.path.join(BASE_DIR, "ensemble_sensor_model.pkl"))
sensor_data = pd.read_csv(os.path.join(BASE_DIR, "fused_sensor_training_data_high_altitude.csv"))
yolo_model = YOLO("yolov8l.pt")

# S3 client
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET")
if not OUTPUT_BUCKET:
    raise RuntimeError("OUTPUT_BUCKET env var is not set")

s3 = boto3.client("s3")

@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

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
    if boxes is None:
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

    # Save to temp file and upload to S3
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        s3.upload_file(tmp.name, OUTPUT_BUCKET, out_name)
        tmp_path = tmp.name

    # Cleanup temp file
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    # Presigned URL (valid 1 hour)
    image_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": OUTPUT_BUCKET, "Key": out_name},
        ExpiresIn=3600
    )

    msg_text = "Confirmed: Victim Identified" if confirmed else "No victims found"
    msg_color = "#ff4d4d" if confirmed else "#41d07f"

    if not os.path.exists(RESULT_HTML_PATH):
        abort(500, description="result.html not found")

    with open(RESULT_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("{MSG_TEXT}", msg_text)
    html = html.replace("{MSG_COLOR}", msg_color)
    html = html.replace("{IMAGE_URL}", image_url)

    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
