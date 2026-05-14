# ✨ Gesture-Controlled Human Computer Interaction System ✨

> 🚀 A real-time AI-powered touchless interaction system using Computer Vision, Machine Learning, and Human Computer Interaction (HCI).

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green?style=for-the-badge\&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-HandTracking-orange?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Working-success?style=for-the-badge)

---

## 🌟 Project Highlights

✅ Real-time hand gesture recognition

✅ Touchless mouse & media control

✅ AI-based gesture classification

✅ Blink-based screenshot capture

✅ Hybrid ML + Rule-Based architecture

✅ Lightweight landmark-based learning

✅ Multiple ML model benchmarking

---

# Gesture-Controlled Human Computer Interaction System

## Overview

This project is a real-time gesture-controlled human computer interaction system built using Computer Vision and Machine Learning. The system uses webcam input to detect hand gestures and facial landmarks, then converts those gestures into actions such as mouse control, media control, and system operations.

The project combines:

* MediaPipe for hand and face landmark detection
* Machine Learning for gesture classification
* OpenCV for real-time webcam processing
* PyAutoGUI for controlling the operating system
* Voice feedback using pyttsx3

The system supports both:

1. Machine Learning-based gesture recognition
2. Rule-based fallback gesture detection

This hybrid architecture improves robustness and reliability.

---

# 🎯 Features

## Hand Gesture Recognition

The system recognizes gestures such as:

* Point Up
* Fist Hand
* Rock On
* Palm Open
* Two Fingers
* Thumbs Up
* Peace Sign

---

## Mouse Control Mode

* Move cursor using index finger
* Left click using pinch gesture
* Right click using pinch gesture
* Drag and drop using two fingers
* Scroll control

---

## Media Control Mode

* Volume control
* Play/Pause media
* Next track
* Previous track

---

## System Control Mode

* Application switching
* Show desktop
* Lock screen
* Open terminal

---

## Face and Blink Detection

* Double blink screenshot capture

---

## Machine Learning Features

* Multi-model training and comparison
* Automatic best model selection
* Cross-validation evaluation
* Confusion matrix generation
* Learning curve visualization

---

# 🛠️ Technologies Used

| Technology   | Purpose                          |
| ------------ | -------------------------------- |
| Python       | Core programming language        |
| OpenCV       | Webcam and image processing      |
| MediaPipe    | Hand and face landmark detection |
| NumPy        | Numerical computations           |
| Scikit-learn | Machine learning models          |
| XGBoost      | Boosting classifier              |
| LightGBM     | Gradient boosting framework      |
| CatBoost     | Boosting algorithm               |
| Matplotlib   | Graph plotting                   |
| Seaborn      | Visualization                    |
| PyAutoGUI    | Mouse and keyboard automation    |
| pyttsx3      | Voice feedback                   |
| Joblib       | Model serialization              |

---

# 🧠 System Architecture

```text
Webcam Input
      ↓
MediaPipe Hand & Face Detection
      ↓
21 Hand Landmarks Extraction
      ↓
Feature Normalization
      ↓
Machine Learning Model
      ↓
Gesture Prediction
      ↓
Mouse / Media / System Actions
```

---

# ✋ Hand Landmarks

The system uses MediaPipe Hands to detect 21 hand landmarks.

Each landmark contains:

* x coordinate
* y coordinate
* z coordinate

Total features:

```text
21 landmarks × 3 coordinates = 63 features
```

These 63 normalized values form the feature vector used by machine learning models.

---

# 📏 Feature Normalization

Raw coordinates are affected by:

* hand size
* camera distance
* hand position

To solve this, coordinates are normalized relative to:

* wrist position
* hand size

Normalization improves:

* scale invariance
* position invariance
* model robustness

---

# 📂 Project Files

## 1. collect_data.py

### Purpose

Collect gesture samples and create the dataset.

### Workflow

1. Webcam captures hand image
2. MediaPipe extracts hand landmarks
3. Landmarks are normalized
4. User presses gesture key
5. Features and labels are stored
6. Dataset saved as:

```text
gesture_data.pkl
```

### Gestures Mapping

| Key | Gesture     |
| --- | ----------- |
| 1   | point_up    |
| 2   | fist_hand   |
| 3   | rock_on     |
| 4   | palm_open   |
| 5   | two_fingers |
| 6   | thumbs_up   |
| 7   | peace_sign  |

---

## 2. train_models_robust.py

### Purpose

Train and compare multiple machine learning models.

### Workflow

1. Load dataset
2. Encode labels
3. Split dataset into training and testing sets
4. Scale features when required
5. Train multiple classifiers
6. Evaluate models using cross-validation and test accuracy
7. Select best model automatically
8. Save trained model

### Models Used

* KNN
* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* AdaBoost
* SVM (RBF)
* SVM (Linear)
* XGBoost
* LightGBM
* CatBoost
* MLP Classifier
* Gaussian Naive Bayes
* Ridge Classifier

### Output Files

* best_model.pkl
* scaler.pkl
* label_encoder.pkl
* model_ranking.csv

---

## 3. evaluate.py

### Purpose

Evaluate model performance.

### Features

* Accuracy comparison chart
* Confusion matrix
* Classification report
* Learning curve

### Output Files

* accuracy_chart.png
* confusion_matrix.png
* classification_report.txt
* learning_curve.png

---

## 4. main.py

### Purpose

Run the real-time gesture control system.

### Workflow

1. Webcam captures frames
2. Hand and face landmarks detected
3. Features extracted and normalized
4. ML model predicts gesture
5. Gesture smoothing applied
6. Actions executed

---

# 🔄 Gesture Recognition Pipeline

```text
Webcam Frame
      ↓
MediaPipe Detection
      ↓
21 Landmarks
      ↓
63 Normalized Features
      ↓
ML Model Prediction
      ↓
Stable Gesture Selection
      ↓
Action Execution
```

---

# 🎛️ Gesture Smoothing

Real-time predictions can fluctuate due to:

* hand movement
* lighting variation
* detection noise

To reduce instability, the system stores recent predictions using a deque buffer and selects the most frequent gesture.

Benefits:

* smoother control
* fewer false actions
* stable user interaction

---

# 👁️ Blink Detection

The system uses MediaPipe FaceMesh to detect eye landmarks.

Eye Aspect Ratio (EAR) is calculated to determine:

* eye open
* eye closed

Two rapid blinks trigger screenshot capture.

---

# 🤖 Machine Learning Concepts Used

## Supervised Learning

The model learns from:

* input features
* labeled gestures

Example:

```text
63 normalized features → point_up
```

---

## Label Encoding

Text labels are converted into numerical values.

Example:

```text
point_up → 0
fist_hand → 1
```

---

## Standardization

Feature scaling ensures:

* zero mean
* unit variance

Important for:

* SVM
* KNN
* Logistic Regression

---

## Cross Validation

The dataset is divided into multiple folds to evaluate model consistency.

Benefits:

* reliable evaluation
* reduced overfitting
* better model comparison

---

# 🚀 How to Run the Project

## Step 1: Install Dependencies

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn \
xgboost lightgbm catboost pyautogui pyttsx3 seaborn matplotlib joblib
```

---

## Step 2: Collect Gesture Data

```bash
python collect_data.py
```

This creates:

```text
gesture_data.pkl
```

---

## Step 3: Train Models

```bash
python train_models_robust.py
```

This creates:

* best_model.pkl
* scaler.pkl
* label_encoder.pkl
* model_ranking.csv

---

## Step 4: Evaluate Models

```bash
python evaluate.py
```

---

## Step 5: Run Real-Time System

```bash
python main.py
```

---

# 🎮 Controls

## Mouse Mode

| Gesture                  | Action          |
| ------------------------ | --------------- |
| Point Up                 | Cursor Movement |
| Pinch                    | Left Click      |
| Pinch Variation          | Right Click     |
| Two Fingers              | Drag            |
| Finger Vertical Movement | Scroll          |

---

## Media Mode

| Gesture         | Action         |
| --------------- | -------------- |
| Finger Distance | Volume Control |
| Fist            | Play/Pause     |
| Rock On         | Next Track     |
| Thumbs Up       | Previous Track |

---

## System Mode

| Gesture   | Action        |
| --------- | ------------- |
| Rock On   | Alt+Tab       |
| Palm Open | Show Desktop  |
| Fist      | Lock Screen   |
| Thumbs Up | Open Terminal |

---

# ✅ Advantages of the System

* Touchless interaction
* Real-time processing
* Lightweight landmark-based approach
* Multi-mode functionality
* Hybrid ML + rule-based architecture
* Easy extensibility

---

# 🚀 Future Improvements

* Cross-platform support
* Deep learning-based gesture sequences
* GUI dashboard
* User-specific calibration
* Dynamic gesture customization
* Multi-hand support
* Confidence thresholding
* Improved feature engineering

---

# 🌍 Applications

* Smart home control
* Accessibility systems
* Touchless interfaces
* Gaming interaction
* Presentation control
* Virtual interaction systems

---

# 📌 Conclusion

## 🧩 Unique Features of This Project

### 🔥 Hybrid Gesture Recognition

The system combines:

* Machine Learning prediction
* Rule-based fallback logic

This ensures the system continues working even if the ML model confidence becomes unstable.

---

### ⚡ Landmark-Based Learning Instead of Raw Images

Instead of using full images, the project uses:

```text
21 Hand Landmarks → 63 Features
```

Benefits:

* Faster processing
* Lower memory usage
* Real-time performance
* Better scalability

---

### 🎙️ Voice Feedback System

The project provides voice confirmations such as:

* "Click"
* "Right Click"
* "Screenshot"
* "Volume 50"

This improves user interaction and accessibility.

---

### 👀 Blink-Based Screenshot System

The system detects double eye blinks using facial landmarks and automatically captures screenshots.

This combines:

* Face landmark detection
* Eye Aspect Ratio analysis
* Event triggering

---

### 🎯 Multi-Mode Architecture

The project supports three independent modes:

| Mode      | Function                      |
| --------- | ----------------------------- |
| 🖱️ Mouse | Cursor & click control        |
| 🎵 Media  | Volume & playback control     |
| 💻 System | Desktop & application control |

---

### 📊 Automatic Best Model Selection

The training pipeline automatically:

* trains multiple ML models
* compares accuracies
* selects the best-performing classifier
* saves deployment-ready model files

This mimics real-world ML experimentation pipelines.

---

This project demonstrates the integration of Computer Vision, Machine Learning, and Human Computer Interaction into a real-time gesture-controlled system. By combining MediaPipe landmark detection with multiple machine learning classifiers, the system achieves efficient and lightweight gesture recognition capable of controlling mouse, media, and system functions in real time.

The hybrid architecture using both machine learning and rule-based fallback improves robustness and reliability, making the system practical for real-world interaction scenarios.

---

