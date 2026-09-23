"""Classify new image files with a trained run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from keras_image_classifier import KicError
from keras_image_classifier.images import letterbox, load_rgb
from keras_image_classifier.train import load_model, load_run


def predict(
    run_dir: Path, files: list[Path], *, top_k: int = 3, batch_size: int = 128
) -> list[dict[str, Any]]:
    """One record per file, in order: the top_k labels, or the reason it was unreadable.

    New files go through the same decoder and letterbox as the training data, so the
    model sees what it was trained on. Inference runs batch_size images at a time, so
    a long file list costs time, not memory.
    """
    if top_k < 1:
        raise KicError("top-k must be at least 1")
    if batch_size < 1:
        raise KicError("batch size must be at least 1")
    run = load_run(run_dir)
    classes: list[str] = run["classes"]
    model = None
    records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], NDArray[np.float32]]] = []

    def flush() -> None:
        nonlocal model
        if not pending:
            return
        if model is None:
            model = load_model(run_dir)
        probabilities = model.predict(np.stack([pixels for _, pixels in pending]), verbose=0)
        for (record, _), row in zip(pending, probabilities, strict=True):
            best = np.argsort(row)[::-1][:top_k]
            record["predictions"] = [
                {"label": classes[int(i)], "probability": round(float(row[int(i)]), 4)}
                for i in best
            ]
        pending.clear()

    for file in files:
        try:
            image = letterbox(load_rgb(file), run["image_size"])
        except KicError as error:
            records.append({"file": str(file), "error": str(error)})
            continue
        record: dict[str, Any] = {"file": str(file)}
        records.append(record)
        pending.append((record, np.asarray(image, dtype=np.float32)))
        if len(pending) == batch_size:
            flush()
    flush()
    return records
