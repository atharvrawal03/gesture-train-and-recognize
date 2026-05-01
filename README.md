# Gesture Control with ML

Just a fun AIML lab project – real‑time hand gesture recognition using MediaPipe + 16 ML models.

## What it does
- Tracks hand landmarks, normalises them (size & distance invariant)
- Trains and compares Random Forest, XGBoost, SVM, KNN, etc.
- Real‑time control: mouse, media, system shortcuts
- Mode switching, voice feedback, double‑blink screenshot

## How to run
1. `pip install -r requirements.txt`
2. `python3 collect_data.py` (press 1‑7 to record gestures)
3. `python3 train_models_robust.py`
4. `python3 evaluate.py`
5. `python3 main.py`

## Results
Best model hits ~96% test accuracy (XGBoost).

Made by me for the AIML lab.
