"""Measure a trained run on a held-out split and write a report a person can read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from keras_image_classifier import KicError
from keras_image_classifier.dataset import load_manifest, load_split, split_identity
from keras_image_classifier.metrics import IntMatrix, confusion_matrix, report
from keras_image_classifier.train import load_model, load_run

METRICS_FILE = "metrics.json"
REPORT_FILE = "report.md"
MATRIX_FILE = "confusion_matrix.png"


def evaluate(run_dir: Path, *, split: str = "test", batch_size: int = 128) -> dict[str, Any]:
    run = load_run(run_dir)
    data = Path(run["config"]["data"])
    manifest = load_manifest(data)
    if manifest.classes != run["classes"] or manifest.image_size != run["image_size"]:
        raise KicError(f"{data} no longer matches the data this run was trained on")
    paths, labels, classes = load_split(data, split)
    if split_identity(data) != run.get("split"):
        raise KicError(
            f"the split in {data} no longer matches the split this run was trained on; "
            "retrain, or restore the splits.json that produced this run"
        )

    from keras_image_classifier.data import ImageBatches

    model = load_model(run_dir)
    batches = ImageBatches(data, paths, labels, batch_size=batch_size, shuffle=False)
    probabilities = model.predict(batches, verbose=0)
    predicted = [int(i) for i in np.argmax(probabilities, axis=1)]
    matrix = confusion_matrix(labels, predicted, len(classes))
    result = {"split": split, **report(matrix, classes), "confusion_matrix": matrix.tolist()}
    (run_dir / METRICS_FILE).write_text(json.dumps(result, indent=1), encoding="utf-8")
    plot_matrix(matrix, classes, run_dir / MATRIX_FILE)
    (run_dir / REPORT_FILE).write_text(render_report(run, result), encoding="utf-8", newline="\n")
    return result


def plot_matrix(matrix: IntMatrix, classes: list[str], target: Path) -> None:
    """Row-normalised heat map; the object API keeps matplotlib free of global state."""
    from matplotlib.figure import Figure

    side = max(4.0, 0.6 * len(classes) + 2)
    figure = Figure(figsize=(side, side), dpi=120)
    axes = figure.subplots()
    rows = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, rows, out=np.zeros(matrix.shape), where=rows > 0)
    axes.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = list(range(len(classes)))
    axes.set_xticks(ticks, labels=classes, rotation=45, ha="right")
    axes.set_yticks(ticks, labels=classes)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    for row in ticks:
        for column in ticks:
            colour = "white" if shares[row, column] > 0.5 else "black"
            axes.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=colour,
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(target)


def render_report(run: dict[str, Any], result: dict[str, Any]) -> str:
    environment = run["environment"]
    lines = [
        "# Evaluation report",
        "",
        f"- Split: `{result['split']}`, {result['samples']} images, "
        f"{len(run['classes'])} classes, {run['image_size']} px",
        f"- Accuracy: **{result['accuracy']:.4f}** "
        f"(always answering the largest class would score {result['majority_baseline']:.4f})",
        f"- Macro F1: **{result['macro_f1']:.4f}**",
        f"- Model: {run['parameters']:,} parameters, best epoch {run['best_epoch']} of "
        f"{run['epochs_run']}, trained in {run['train_seconds']} s, seed {run['config']['seed']}",
        f"- Environment: Keras {environment['keras']} on {environment['backend']}, "
        f"Python {environment['python']}, {environment['platform']}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in result["per_class"].items():
        lines.append(
            f"| {name} | {entry['precision']:.3f} | {entry['recall']:.3f} | "
            f"{entry['f1']:.3f} | {entry['support']} |"
        )
    lines += ["", f"![Confusion matrix]({MATRIX_FILE})", ""]
    return "\n".join(lines)
