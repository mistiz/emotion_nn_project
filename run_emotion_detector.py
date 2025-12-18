import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime
import threading
import time
import os

from audio_emotion import transcribe_and_analyze  # returns (text, vocal_emotion)

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "models/emotion_model.h5"
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

SMOOTHING_WINDOW = 10          # number of recent predictions to smooth over
CONFIDENCE_THRESHOLD = 0.75    # for face emotion
SAVE_IMAGES = True
SAVE_DIR = "saved_faces"

LOG_FILE = "emotion_log.txt"

VOICE_INTERVAL = 10.0          # seconds between voice analyses (Option A)
VOICE_TIMEOUT_LOG = 2.0        # seconds to wait before considering voice "Pending"

# -----------------------------
# Global state for voice thread
# -----------------------------

voice_lock = threading.Lock()
latest_vocal_emotion = None
latest_speech_text = None
last_voice_time = 0.0
voice_thread_running = False

# -----------------------------
# Logging helper
# -----------------------------

def log_event(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {message}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")

# -----------------------------
# Voice worker (runs in thread)
# -----------------------------

def voice_worker():
    global latest_vocal_emotion, latest_speech_text, last_voice_time, voice_thread_running

    try:
        text, vocal_emotion = transcribe_and_analyze()  # blocking call
        with voice_lock:
            latest_speech_text = text
            latest_vocal_emotion = vocal_emotion
            last_voice_time = time.time()

        log_event(f"Voice update | Speech: {text!r} | Voice Emotion: {vocal_emotion}")
    except Exception as e:
        log_event(f"Voice error: {e}")
    finally:
        voice_thread_running = False

# -----------------------------
# Helper: load model
# -----------------------------

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# -----------------------------
# Helper: preprocess face image
# -----------------------------

def preprocess_face(gray_frame, face_box):
    (x, y, w, h) = face_box
    face = gray_frame[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)  # (48, 48, 1)
    face = np.expand_dims(face, axis=0)   # (1, 48, 48, 1)
    return face

# -----------------------------
# Helper: get smoothed emotion
# -----------------------------

def get_stable_emotion(history):
    if not history:
        return None
    # majority vote
    values, counts = np.unique(history, return_counts=True)
    return values[np.argmax(counts)]

# -----------------------------
# Helper: fuse face + voice emotions
# -----------------------------

def fuse_emotions(face_emotion, face_confidence, voice_emotion):
    """
    Simple fusion rule:
    - If no voice emotion: use face.
    - If face and voice agree: use that.
    - If they differ: prefer the one that is not Neutral (if any).
    - If still ambiguous: prefer face (more frequent).
    """
    if voice_emotion is None:
        return face_emotion

    if face_emotion is None:
        return voice_emotion

    # Agreement
    if face_emotion == voice_emotion:
        return face_emotion

    # Disagreement: prefer non-Neutral
    if face_emotion != "Neutral" and voice_emotion == "Neutral":
        return face_emotion
    if voice_emotion != "Neutral" and face_emotion == "Neutral":
        return voice_emotion

    # Both non-neutral but different: keep face as primary
    return face_emotion

# -----------------------------
# Main
# -----------------------------

def main():
    global voice_thread_running

    # Ensure save directory exists
    if SAVE_IMAGES and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    # Load model
    log_event("Loading emotion recognition model...")
    model = load_model()
    log_event("Model loaded.")

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log_event("Error: Could not open webcam.")
        return

    # Face detector (Haar cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    emotion_history = deque(maxlen=SMOOTHING_WINDOW)
    last_logged_emotion = None

    log_event("Starting real-time emotion detection with threaded voice analysis...")

    while True:
        ret, frame = cap.read()
        if not ret:
            log_event("Error: Failed to read frame from webcam.")
            break

        # Optionally downscale for speed
        frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_emotion = None
        face_confidence = 0.0

        for (x, y, w, h) in faces:
            face_input = preprocess_face(gray, (x, y, w, h))

            preds = model.predict(face_input, verbose=0)[0]
            max_idx = int(np.argmax(preds))
            face_emotion = EMOTION_LABELS[max_idx]
            face_confidence = float(preds[max_idx])

            # Update smoothing history
            emotion_history.append(face_emotion)
            stable_emotion = get_stable_emotion(emotion_history)

            # Draw bounding box + label
            label = f"{stable_emotion} ({face_confidence:.2f})"
            color = (0, 255, 0) if face_confidence >= CONFIDENCE_THRESHOLD else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Save face image if high confidence
            if SAVE_IMAGES and face_confidence >= CONFIDENCE_THRESHOLD:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{SAVE_DIR}/{timestamp}_{stable_emotion}_{face_confidence:.2f}.jpg"
                face_color = frame[y:y+h, x:x+w]
                cv2.imwrite(filename, face_color)
                log_event(f"Saved face image: {filename}")

            break  # currently only using first detected face

        # Get current stable face emotion from history
        stable_emotion = get_stable_emotion(emotion_history)

        # -----------------------------
        # Start voice thread periodically
        # -----------------------------
        current_time = time.time()
        time_since_last_voice = current_time - last_voice_time

        if (not voice_thread_running) and (time_since_last_voice > VOICE_INTERVAL):
            voice_thread_running = True
            threading.Thread(target=voice_worker, daemon=True).start()
            log_event("Started voice analysis thread...")

        # -----------------------------
        # Read latest voice state
        # -----------------------------
        with voice_lock:
            voice_emotion_snapshot = latest_vocal_emotion
            speech_snapshot = latest_speech_text
            last_voice_time_snapshot = last_voice_time

        # Determine voice status text (for overlay/log clarity)
        voice_status = "None"
        if voice_emotion_snapshot is not None:
            voice_status = voice_emotion_snapshot
        elif current_time - last_voice_time_snapshot < VOICE_TIMEOUT_LOG:
            voice_status = "Pending"

        # -----------------------------
        # Fuse emotions
        # -----------------------------
        fused_emotion = fuse_emotions(stable_emotion, face_confidence, voice_emotion_snapshot)

        # -----------------------------
        # Overlay info on frame
        # -----------------------------
        overlay_y = 20

        cv2.putText(
            frame,
            f"Face: {stable_emotion if stable_emotion else 'None'} ({face_confidence:.2f})",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        overlay_y += 25

        cv2.putText(
            frame,
            f"Voice: {voice_status}",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        overlay_y += 25

        cv2.putText(
            frame,
            f"Fused: {fused_emotion if fused_emotion else 'None'}",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # -----------------------------
        # Log when fused emotion changes
        # -----------------------------
        if fused_emotion is not None and fused_emotion != last_logged_emotion:
            log_event(
                f"Fused update | Face: {stable_emotion} ({face_confidence:.2f}) | "
                f"Voice: {voice_status} | Speech: {speech_snapshot!r} | Fused: {fused_emotion}"
            )
            last_logged_emotion = fused_emotion

        # Show frame
        cv2.imshow("Fused Emotion Detector (Threaded)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log_event("User requested exit (q pressed).")
            break

    cap.release()
    cv2.destroyAllWindows()
    log_event("Shutting down.")


if __name__ == "__main__":
    main()
