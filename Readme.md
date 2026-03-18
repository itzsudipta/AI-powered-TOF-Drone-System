# Disaster Drone - Dummy Datasets

This repo contains scripts to generate large synthetic datasets for a multi-sensor disaster-response drone and a baseline model demo.

## Datasets
- `data/sensor_fusion_dataset.csv` includes TOF, IR, thermal, and environment features with labels `victim_present` and `victim_count`.
- `data/battery_dataset.csv` includes battery telemetry with label `battery_status`.

## Generate Data
```bash
python generate_datasets.py
```

Optional size controls:
```bash
python generate_datasets.py --sensor_rows 300000 --battery_rows 200000
```

## Baseline Algorithms
Run the baseline surveillance and battery status rules:
```bash
python main.py
```

## Train/Test Split Example (50/50)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data/sensor_fusion_dataset.csv")
X = data.drop(columns=["victim_present", "victim_count"])
y = data["victim_present"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
```
