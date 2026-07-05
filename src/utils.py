import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

# -----------------------------
# PARAMETERS
# -----------------------------

IMG_SIZE = 64
MAX_FRAMES = 10
FRAME_SKIP = 5

# ======================================================
# FRAME EXTRACTION
# ======================================================

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

            if len(frames) == MAX_FRAMES:
                break

        frame_count += 1

    cap.release()

    while len(frames) < MAX_FRAMES:

        frames.append(
            np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        )

    return np.array(frames)


# ======================================================
# VIDEO DATA GENERATOR
# ======================================================

class VideoDataGenerator(Sequence):

    def __init__(
        self,
        video_paths,
        labels,
        batch_size=8,
        shuffle=True
    ):

        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(video_paths))

        self.on_epoch_end()

    def __len__(self):

        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):

        batch_indices = self.indices[
            index*self.batch_size:
            (index+1)*self.batch_size
        ]

        X = []

        y = []

        for i in batch_indices:

            frames = extract_frames(self.video_paths[i])

            X.append(frames)

            y.append(self.labels[i])

        return np.array(X), np.array(y)

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(self.indices)
