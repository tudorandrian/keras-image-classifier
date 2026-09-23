"""Generate a small labelled data set of drawn shapes.

It needs no download and no licence, trains in seconds, and is hard enough that a
broken pipeline fails on it: position, size, colour and background all vary, so
only the shape separates the classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from keras_image_classifier import KicError

SHAPES = ("circle", "square", "triangle")


def draw_shape(shape: str, size: int, rng: np.random.Generator) -> Image.Image:
    if shape not in SHAPES:
        raise KicError(f"unknown shape {shape!r}")
    noise = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
    image = Image.fromarray(noise)
    radius = int(rng.integers(size // 5, size // 3))
    cx, cy = (int(v) for v in rng.integers(radius + 1, size - radius - 1, 2))
    red, green, blue = (int(v) for v in rng.integers(120, 256, 3))
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse(box, fill=(red, green, blue))
    elif shape == "square":
        draw.rectangle(box, fill=(red, green, blue))
    else:
        corners = [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)]
        draw.polygon(corners, fill=(red, green, blue))
    return image


def generate(destination: Path, *, per_class: int = 200, size: int = 48, seed: int = 0) -> int:
    """Write per_class PNG files for each shape under destination/<shape>/; return the total."""
    if per_class < 3:
        raise KicError("per-class count must be at least 3")
    if not 16 <= size <= 512:
        raise KicError("size must be between 16 and 512")
    if destination.exists() and any(destination.iterdir()):
        raise KicError(f"{destination} is not empty; choose a new directory or delete it")
    rng = np.random.default_rng(seed)
    for shape in SHAPES:
        directory = destination / shape
        directory.mkdir(parents=True, exist_ok=True)
        for number in range(per_class):
            draw_shape(shape, size, rng).save(directory / f"{shape}_{number:05d}.png")
    return per_class * len(SHAPES)
