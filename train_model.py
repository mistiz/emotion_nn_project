
# train_model.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

DATASET_PATH = "dataset/fer2013.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_cnn.h5")

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def load_fer2013():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    # FER2013 has columns: emotion, pixels, Usage
    train_data = data[data["Usage"] == "Training"]
    val_data = data[data["Usage"] == "PublicTest"]

    def process_subset(subset):
        pixels = subset["pixels"].tolist()
        X = np.array([np.fromstring(p, sep=" ") for p in pixels], dtype="float32")
        X = X.reshape((-1, 48, 48, 1)) / 255.0
        y = subset["emotion"].values
        y = to_categorical(y, num_classes=len(EMOTIONS))
        return X, y

    X_train, y_train = process_subset(train_data)
    X_val, y_val = process_subset(val_data)

    return X_train, y_train, X_val, y_val

def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Loading FER2013 dataset...")
    X_train, y_train, X_val, y_val = load_fer2013()
    print("Dataset loaded.")
    print("Train:", X_train.shape, "Val:", X_val.shape)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64
    )

    print(f"Saving model to {MODEL_PATH} ...")
    model.save(MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
