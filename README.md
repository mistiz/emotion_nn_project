# emotion_nn_project
“Multimodal Emotion Detection (Face + Voice)”
## 🏗️ Architecture Overview

This project implements a real‑time multimodal emotion detection system that combines:

- Facial emotion recognition (CNN model + OpenCV)
- Vocal emotion recognition (Whisper + audio classifier)
- Threaded background processing for smooth performance
- Rule‑based fusion of face + voice emotion
- Unified logging and automatic face image saving

---

## 📦 Project Structure

emotion_nn_project/
│
├── run_emotion_detector.py        # Main application (webcam + fusion + threading)
├── audio_emotion.py               # Whisper STT + vocal emotion analysis
├── train_model.py                 # (Optional) CNN training script
├── emotion_log.txt                # Unified log (face + voice + fused)
│
├── models/
│   ├── emotion_cnn.h5             # Trained CNN model
│   ├── emotion_model.weights.h5   # Optional weights
│   └── haarcascade_frontalface_default.xml
│
├── saved_faces/                   # Auto‑saved high‑confidence face crops
├── launch/                        # Batch launcher + shortcut
└── venv/                          # Virtual environment

---

## 🧩 Components

### **Application Controller**
Handles webcam loop, CNN inference, Whisper threading, fusion logic, logging, and UI overlays.

### **Face Processing**
Haar cascade → preprocessing → CNN → smoothing → confidence filtering.

### **Voice Processing**
Background thread running Whisper STT + vocal emotion classifier.

### **Fusion Engine**
Combines face + voice emotion using rule‑based logic.

### **Logging System**
Records face emotion, voice emotion, speech text, fused emotion, and saved image events.

---

## 🔄 High‑Level Flow

1. Capture webcam frame  
2. Detect face  
3. Predict facial emotion (CNN)  
4. Smooth predictions  
5. Whisper thread runs every X seconds  
6. Extract speech + vocal emotion  
7. Fuse face + voice emotion  
8. Update UI overlays  
9. Log events  
10. Save high‑confidence face images
