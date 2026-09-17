"""The network. Only the Keras API is used, so any Keras 3 backend can run it."""

from __future__ import annotations

from typing import Any

import keras
from keras import layers

# Batch normalisation uses batch statistics while training and moving averages at
# inference. The moving variance starts at 1.0 while the real value after the first
# convolution is near 0.03, so until the average has caught up the model answers at
# chance on anything it is evaluated on, including its own training data. With the
# Keras default momentum of 0.99 that takes several hundred steps; with 0.9 it took
# about 45 in measurements on the synthetic data set (docs/testing.md).
BATCH_NORM_MOMENTUM = 0.9
# Early stopping must not count epochs before that point, or it stops a run that is
# learning well. train.py converts this number of steps into epochs.
BATCH_NORM_WARMUP_STEPS = 50


def build_cnn(
    image_size: int,
    num_classes: int,
    *,
    width: int = 16,
    dropout: float = 0.2,
) -> Any:
    """Four convolution blocks, global average pooling, softmax.

    Global pooling instead of Flatten keeps the head at a few hundred weights whatever
    the input size; the coursework version spent 11 of its 11.2 million parameters in
    one Dense layer after Flatten. Rescaling is a layer, so a saved model takes raw
    0-255 pixels and cannot be fed wrongly scaled input.
    """
    inputs = keras.Input(shape=(image_size, image_size, 3), name="pixels")
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (width, width * 2, width * 4, width * 8):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="small_cnn")
