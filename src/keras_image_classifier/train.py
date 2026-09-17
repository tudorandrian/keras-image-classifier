"""Train the network on a prepared, split data set and record how it was done."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from keras_image_classifier import KicError, __version__
from keras_image_classifier.dataset import load_manifest, load_split

MODEL_FILE = "model.keras"
RUN_FILE = "run.json"
HISTORY_FILE = "history.csv"


@dataclass(frozen=True)
class TrainConfig:
    data: str
    run_dir: str
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 0
    width: int = 16
    dropout: float = 0.2
    augment: bool = True
    patience: int = 4

    def validate(self) -> None:
        problems = []
        if not 1 <= self.epochs <= 1000:
            problems.append("epochs must be between 1 and 1000")
        if not 1 <= self.batch_size <= 1024:
            problems.append("batch size must be between 1 and 1024")
        if not 0 < self.learning_rate <= 1:
            problems.append("learning rate must be in (0, 1]")
        if not 4 <= self.width <= 256:
            problems.append("width must be between 4 and 256")
        if not 0 <= self.dropout < 1:
            problems.append("dropout must be in [0, 1)")
        if self.patience < 1:
            problems.append("patience must be at least 1")
        if problems:
            raise KicError("; ".join(problems))


def environment() -> dict[str, str]:
    """Versions that decide whether a run can be repeated."""
    import keras
    import numpy

    return {
        "keras_image_classifier": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "keras": keras.__version__,
        "backend": keras.backend.backend(),
        "numpy": numpy.__version__,
    }


def train(config: TrainConfig) -> dict[str, Any]:
    """Fit the model, keep the best epoch, write model.keras, history.csv and run.json."""
    config.validate()
    data = Path(config.data)
    run_dir = Path(config.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise KicError(f"{run_dir} is not empty; every run gets its own directory")
    manifest = load_manifest(data)
    train_paths, train_labels, classes = load_split(data, "train")
    val_paths, val_labels, _ = load_split(data, "val")

    import keras

    from keras_image_classifier.data import ImageBatches
    from keras_image_classifier.model import BATCH_NORM_WARMUP_STEPS, build_cnn

    keras.utils.set_random_seed(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_cnn(
        manifest.image_size,
        len(classes),
        width=config.width,
        dropout=config.dropout,
    )
    batches = ImageBatches(
        data,
        train_paths,
        train_labels,
        batch_size=config.batch_size,
        shuffle=True,
        augment=config.augment,
        seed=config.seed,
    )
    validation = ImageBatches(
        data, val_paths, val_labels, batch_size=config.batch_size, shuffle=False
    )
    # Cosine decay to zero over the planned epochs: the late, small steps are what
    # steadies a validation loss that swings from epoch to epoch at a constant rate.
    schedule = keras.optimizers.schedules.CosineDecay(
        config.learning_rate, decay_steps=config.epochs * len(batches)
    )
    model.compile(
        optimizer=keras.optimizers.Adam(schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    warmup_epochs = math.ceil(BATCH_NORM_WARMUP_STEPS / len(batches))
    started = time.perf_counter()
    history = model.fit(
        batches,
        validation_data=validation,
        epochs=config.epochs,
        verbose=2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(run_dir / MODEL_FILE), monitor="val_loss", save_best_only=True
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.patience, start_from_epoch=warmup_epochs
            ),
            keras.callbacks.CSVLogger(str(run_dir / HISTORY_FILE)),
        ],
    )
    seconds = time.perf_counter() - started
    losses = history.history["val_loss"]
    best = min(range(len(losses)), key=losses.__getitem__)
    summary = {
        "config": asdict(config),
        "classes": classes,
        "image_size": manifest.image_size,
        "parameters": int(model.count_params()),
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "epochs_run": len(losses),
        "best_epoch": best + 1,
        "best_val_loss": float(losses[best]),
        "best_val_accuracy": float(history.history["val_accuracy"][best]),
        "train_seconds": round(seconds, 1),
        "environment": environment(),
    }
    (run_dir / RUN_FILE).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_run(run_dir: Path) -> dict[str, Any]:
    try:
        loaded: dict[str, Any] = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KicError(f"{run_dir} has no {RUN_FILE}; is it a run directory?") from None
    if not (run_dir / MODEL_FILE).is_file():
        raise KicError(f"{run_dir} has no {MODEL_FILE}")
    return loaded


def load_model(run_dir: Path) -> Any:
    """Load with safe_mode, which refuses models that carry arbitrary Python code."""
    import keras

    return keras.saving.load_model(str(run_dir / MODEL_FILE), safe_mode=True)
