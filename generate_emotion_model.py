import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "models/emotion_model.weights.h5"

# Ensure dataset exists
for base in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(base):
        raise FileNotFoundError(f"{base} not found")

# Data loaders
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(TRAIN_DIR, target_size=(48,48),
                                         color_mode="grayscale", class_mode="categorical", batch_size=32)
test_data = datagen.flow_from_directory(TEST_DIR, target_size=(48,48),
                                        color_mode="grayscale", class_mode="categorical", batch_size=32)

# Model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(train_data, epochs=1, validation_data=test_data)

# Save weights
model.save_weights(MODEL_PATH)
print(f"✅ Model weights saved to {MODEL_PATH}")
