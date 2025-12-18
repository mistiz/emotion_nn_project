# 🎭 Multimodal Emotion Detection System
A real‑time multimodal emotion recognition system that analyzes **facial expressions** and **vocal tone** using deep learning, computer vision, and speech processing. The system uses your webcam and microphone to detect emotions live, fusing both modalities for more accurate and human‑like predictions.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Models](#models)
- [Logging & Saved Outputs](#logging--saved-outputs)
- [Diagrams](#diagrams)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🧠 Overview
This project performs **real‑time emotion detection** by combining:
- CNN‑based **facial emotion recognition**
- Whisper‑based **speech‑to‑text**
- A **vocal emotion classifier**
- A **fusion engine** that merges both signals

The result is a more robust and realistic emotion detection system that mimics how humans interpret emotions.

---

## ✅ Features

### 🎥 Facial Emotion Recognition
- Real‑time webcam processing  
- Haar Cascade face detection  
- CNN model for emotion classification  
- Prediction smoothing for stability  

### 🎤 Voice Emotion Recognition
- Whisper for speech‑to‑text  
- Vocal emotion classifier  
- Runs in a background thread  

### 🔗 Fusion Engine
- Combines face + voice emotion  
- Rule‑based decision logic  
- Handles mismatches and neutral cases  

### 🖥️ User Interface
- Live webcam overlay  
- Displays:
  - Face emotion  
  - Voice emotion  
  - Fused emotion  
  - Whisper transcription  

### 📝 Logging & Saving
- Logs all events to `emotion_log.txt`  
- Saves high‑confidence face images to `saved_faces/`  

---

## 🏗️ System Architecture

+-------------------------------------------------------------+ | Multimodal Emotion System | +-------------------------------------------------------------+ | | | +-------------------+ +---------------------------+ | | | UI Layer | | Logging Module | | | | (Webcam Overlay) | | (Events, Speech, Images) | | | +-------------------+ +---------------------------+ | | | | +-------------------------------------------------------+ | | | Application Controller | | | | (Main Loop, Threading, Fusion, Saving, Orchestration) | | | +--------------------+------------------+----------------+ | | | | | | +---------------+ +----------------+ | | | | | ▼ ▼ | | +-------------------+ +----------------------------+ | | | Face Processing | | Voice Processing | | | | (Haar, CNN, Smooth)| | (Whisper, Vocal Emotion) | | | +-------------------+ +----------------------------+ | | | +-------------------------------------------------------------+

Code

---

## 📁 Project Structure

emotion_nn_project/ │ ├── run_emotion_detector.py ├── audio_emotion.py ├── fusion_logic.py ├── requirements.txt ├── README.md ├── .gitignore ├── .gitattributes │ ├── models/ │ ├── emotion_model.h5 │ ├── emotion_model.weights.h5 │ └── haarcascade_frontalface_default.xml │ ├── launch/ │ ├── run_emotion.bat │ └── setup_env.bat │ ├── diagrams/ │ ├── architecture.png │ ├── component_diagram.png │ └── fusion_flowchart.png │ ├── slides/ │ └── presentation.pptx │ └── docs/ ├── system_overview.md ├── installation_guide.md └── usage_guide.md

Code

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/emotion_nn_project.git
cd emotion_nn_project
2️⃣ Create and activate a virtual environment
bash
python -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies
bash
pip install -r requirements.txt
▶️ Running the Application
✅ Option 1 — One‑click launcher
bash
launch/run_emotion.bat
✅ Option 2 — Manual run
bash
python run_emotion_detector.py
🔍 How It Works
1. Capture
Webcam captures video

Microphone captures audio

2. Analyze
Face → Haar Cascade → CNN → Emotion

Voice → Whisper → Vocal Emotion

3. Fuse
Rule‑based fusion engine combines both signals

4. Display
Live overlay with all predictions

5. Log & Save
Logs events

Saves high‑confidence face images

🧩 Models
✅ Facial Emotion Model
emotion_model.h5

emotion_model.weights.h5

✅ Face Detection
haarcascade_frontalface_default.xml

✅ Whisper
Installed via pip

No model files stored in repo

📝 Logging & Saved Outputs
Logs stored in:

Code
emotion_log.txt
Saved face images stored in:

Code
saved_faces/
Both are excluded from GitHub via .gitignore.

🖼️ Diagrams
All diagrams are stored in:

Code
diagrams/
Includes:

Architecture Diagram

Component Diagram

Data Flow Diagram

Fusion Flowchart

📦 Requirements
All dependencies are listed in:

Code
requirements.txt
Key libraries:

TensorFlow / Keras

OpenCV

Whisper

NumPy

SciPy

PyAudio or sounddevice

📚 Dataset
This project uses:

FER2013 for facial emotion training

A small curated dataset for vocal emotion (optional)

⚠️ Datasets are not included due to size.

🚀 Future Improvements
Transformer‑based facial emotion models

ML‑based fusion

Multi‑face detection and tracking

GUI dashboard (Tkinter / PyQt)

REST API for remote emotion detection

Analytics mode for emotion timelines

📄 License
MIT License You are free to use, modify, and distribute this project.

🙌 Acknowledgements
TensorFlow

OpenCV

Whisper

FER2013 dataset

Python community


