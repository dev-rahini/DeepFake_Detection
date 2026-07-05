import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    LSTM,
    Dense,
    Dropout
)


def build_model(max_frames=10, img_size=64):

    model = Sequential()

    model.add(
        Input(shape=(max_frames, img_size, img_size, 3))
    )

    # CNN Block 1
    model.add(
        TimeDistributed(
            Conv2D(32, (3,3), activation="relu")
        )
    )

    model.add(
        TimeDistributed(
            MaxPooling2D((2,2))
        )
    )

    model.add(
        TimeDistributed(
            BatchNormalization()
        )
    )

    # CNN Block 2
    model.add(
        TimeDistributed(
            Conv2D(64, (3,3), activation="relu")
        )
    )

    model.add(
        TimeDistributed(
            MaxPooling2D((2,2))
        )
    )

    model.add(
        TimeDistributed(
            BatchNormalization()
        )
    )

    # CNN Block 3
    model.add(
        TimeDistributed(
            Conv2D(96, (3,3), activation="relu")
        )
    )

    model.add(
        TimeDistributed(
            MaxPooling2D((2,2))
        )
    )

    model.add(
        TimeDistributed(
            Flatten()
        )
    )

    # LSTM
    model.add(
        LSTM(64)
    )

    # Dense Layers
    model.add(
        Dense(128, activation="relu")
    )

    model.add(
        Dropout(0.5)
    )

    model.add(
        Dense(64, activation="relu")
    )

    model.add(
        Dropout(0.3)
    )

    # Output Layer
    model.add(
        Dense(1, activation="sigmoid")
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
