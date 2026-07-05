import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf

from model import build_model
from utils import VideoDataGenerator

# ======================================================
# DATASET PATH
# ======================================================

dataset_path = "/content/FaceForensics++_C23"

MAX_VIDEOS_PER_CLASS = 100

# ======================================================
# LOAD VIDEOS
# ======================================================

real_videos = []

original_path = os.path.join(dataset_path, "original")

for video in os.listdir(original_path):

    if video.endswith(".mp4"):

        real_videos.append(
            os.path.join(original_path, video)
        )

real_videos = real_videos[:MAX_VIDEOS_PER_CLASS]

# ------------------------------------------------------

fake_folders = [

    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"

]

fake_videos = []

for folder in fake_folders:

    folder_path = os.path.join(dataset_path, folder)

    count = 0

    for video in os.listdir(folder_path):

        if video.endswith(".mp4"):

            fake_videos.append(
                os.path.join(folder_path, video)
            )

            count += 1

            if count >= MAX_VIDEOS_PER_CLASS:
                break

# ======================================================
# LABELS
# ======================================================

video_paths = real_videos + fake_videos

labels = [0] * len(real_videos) + [1] * len(fake_videos)

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

train_paths, test_paths, train_labels, test_labels = train_test_split(

    video_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels

)

# ======================================================
# DATA GENERATORS
# ======================================================

train_generator = VideoDataGenerator(

    train_paths,
    train_labels,
    batch_size=8

)

test_generator = VideoDataGenerator(

    test_paths,
    test_labels,
    batch_size=8,
    shuffle=False

)

# ======================================================
# BUILD MODEL
# ======================================================

model = build_model()

model.summary()

# ======================================================
# CLASS WEIGHTS
# ======================================================

weights = compute_class_weight(

    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels

)

class_weights = {

    0: weights[0],
    1: weights[1]

}

# ======================================================
# CALLBACKS
# ======================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",
        patience=5,
        restore_best_weights=True

    ),

    tf.keras.callbacks.ModelCheckpoint(

        "best_deepfake_model.keras",
        save_best_only=True,
        monitor="val_accuracy"

    )

]

# ======================================================
# TRAIN
# ======================================================

history = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=20,

    class_weight=class_weights,

    callbacks=callbacks

)

# ======================================================
# SAVE MODEL
# ======================================================

model.save("../saved_model/deepfake_detector.keras")

print("Training Completed Successfully!")
