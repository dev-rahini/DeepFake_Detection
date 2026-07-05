import numpy as np
import tensorflow as tf

from utils import extract_frames

# ======================================================
# LOAD TRAINED MODEL
# ======================================================

MODEL_PATH = "../saved_model/deepfake_detector.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ======================================================
# PREDICT VIDEO
# ======================================================

def predict_video(video_path):

    frames = extract_frames(video_path)

    video = np.expand_dims(frames, axis=0)

    prediction = model.predict(video, verbose=0)[0][0]

    print(f"Prediction Score : {prediction:.4f}")

    if prediction >= 0.5:
        print("Prediction : Fake Video")
    else:
        print("Prediction : Real Video")

# ======================================================
# EXAMPLE
# ======================================================

if __name__ == "__main__":

    test_video = "test_video.mp4"   # Change to your video path

    predict_video(test_video)
