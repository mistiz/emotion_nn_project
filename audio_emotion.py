import sounddevice as sd
import numpy as np
import whisper

model = whisper.load_model("base")

def record_audio(duration=3, fs=16000):
    print("Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def transcribe_and_analyze():
    audio = record_audio()
    result = model.transcribe(audio, fp16=False)

    text = result["text"].strip()

    # Simple vocal emotion heuristic
    # (You can upgrade this later)
    if "!" in text or text.isupper():
        vocal_emotion = "Excited"
    elif any(word in text.lower() for word in ["sad", "tired", "upset"]):
        vocal_emotion = "Sad"
    else:
        vocal_emotion = "Neutral"

    return text, vocal_emotion
