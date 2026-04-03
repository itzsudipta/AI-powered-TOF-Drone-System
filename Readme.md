# Efficient victim detection model

Synthetic multi‑sensor data + vision experiments for a disaster‑response drone. The core work lives in `main.ipynb`, which walks through data generation, classic ML baselines, and a YOLO‑based detection demo with a simple Flask UI.

## Contents
- `main.ipynb`: end‑to‑end notebook (data generation, Random Forest training, HOG baseline notes, YOLO demo, and a Flask + ngrok backend section).
- `fused_sensor_training_data.csv`: synthetic sensor‑fusion dataset.
- `fused_sensor_training_data_high_altitude.csv`: alternate synthetic dataset (same schema).
- `templates/`: Flask HTML templates for upload + results.
- `Static/Uploads/`: upload target folder for the Flask demo.
- `data/`: sample images used for testing/visualization.

## Dataset Schema
Both CSVs share the same columns:
- `Timestamp`
- `Pixel_X`
- `Pixel_Y`
- `Depth_mm`
- `Temperature_C`
- `ToF_Amplitude`
- `Target_Label` (e.g., Victim / Surroundings / Heated Object)

## Notebook Requirements
The notebook imports the following Python packages:
- `numpy`, `pandas`, `scikit-learn`
- `opencv-python` (`cv2`)
- `ultralytics` (YOLOv8)
- `flask`, `werkzeug`, `pyngrok`
- `joblib` / `pickle`

## Quick Start
1. Open `main.ipynb` in Jupyter or VS Code.
2. Run cells in order to generate data, train a baseline model, and explore the vision demo.

## Flask UI (Templates)
The HTML templates are in `templates/` and expect a Flask app to:
- Accept an uploaded image at `/`
- Render `result.html` with `victim_found` and `image_url`

