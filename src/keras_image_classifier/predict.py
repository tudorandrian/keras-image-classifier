"""Classify new image files with a trained run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from keras_image_classifier import KicError
from keras_image_classifier.images import letterbox, load_rgb
from keras_image_classifier.train import load_model, load_run


def predict(run_dir: Path, files: list[Path], *, top_k: int = 3) -> list[dict[str, Any]]:
    """One record per file, in order: the top_k labels, or the reason it was unreadable.

    New files go through the same decoder and letterbox as the training data, so the
    model sees what it was trained on.
    """
    if top_k < 1:
        raise KicError("top-k must be at least 1")
    run = load_run(run_dir)
    classes: list[str] = run["classes"]
    records: list[dict[str, Any]] = []
    pixels = []
    for file in files:
        try:
            image = letterbox(load_rgb(file), run["image_size"])
        except KicError as error:
            records.append({"file": str(file), "error": str(error)})
            continue
        records.append({"file": str(file)})
        pixels.append(np.asarray(image, dtype=np.float32))
    if pixels:
        probabilities = load_model(run_dir).predict(np.stack(pixels), verbose=0)
        readable = (record for record in records if "error" not in record)
        for record, row in zip(readable, probabilities, strict=True):
            best = np.argsort(row)[::-1][:top_k]
            record["predictions"] = [
                {"label": classes[int(i)], "probability": round(float(row[int(i)]), 4)}
                for i in best
            ]
    return records
