"""Feed prepared PNG files to Keras in batches, without tf.data and so without TensorFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np
from numpy.typing import NDArray
from PIL import Image


class ImageBatches(keras.utils.PyDataset):  # type: ignore[misc]
    """Batches of (pixels 0-255 as float32, integer labels).

    Decoded images stay in memory after first use. A prepared data set is small
    (27,000 images of 64 px are 330 MB), and decoding PNG files again every epoch
    would otherwise dominate CPU training time.
    """

    def __init__(
        self,
        root: Path,
        paths: list[str],
        labels: list[int],
        *,
        batch_size: int,
        shuffle: bool,
        augment: bool = False,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.root = root
        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.cache: dict[int, NDArray[np.uint8]] = {}
        self.order = np.arange(len(paths))
        self._reorder()

    def _reorder(self) -> None:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            self.order = rng.permutation(len(self.paths))

    def _pixels(self, index: int) -> NDArray[np.uint8]:
        if index not in self.cache:
            with Image.open(self.root / self.paths[index]) as image:
                self.cache[index] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return self.cache[index]

    def __len__(self) -> int:
        return (len(self.paths) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch: int) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        chosen = self.order[batch * self.batch_size : (batch + 1) * self.batch_size]
        pixels = np.stack([self._pixels(int(i)) for i in chosen]).astype(np.float32)
        if self.augment:
            # Horizontal flips, decided by (seed, epoch, batch): cheap, and the same run
            # sees the same flips again. Doing it here instead of with a
            # keras.layers.RandomFlip also keeps the saved model free of training-only
            # layers, so what is served is what was validated.
            rng = np.random.default_rng((self.seed, self.epoch, batch))
            mirrored = rng.random(len(chosen)) < 0.5
            pixels[mirrored] = pixels[mirrored, :, ::-1]
        return pixels, self.labels[chosen]

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._reorder()
