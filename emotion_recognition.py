# emotion_recognition.py

import cv2
import random

# Predefined list of emotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load Haar cascade for face detection
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise Exception("Failed to load Haar cascade for face detection!")

def detect_emotion(frame):
    """
    Detect faces and return random emotions for demonstration.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Assign a random emotion
        emotion = random.choice(EMOTIONS)
        results.append((x, y, w, h, emotion))
        # Put the emotion text on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)
    
    return frame, results

# Simple test if run standalone
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, faces = detect_emotion(frame)
        cv2.imshow("Emotion Recognition Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
