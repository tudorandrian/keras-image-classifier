"""One real training run on synthetic shapes, checked from every side.

The numbers are the point. A pipeline that shuffles labels, leaks the test split or
saves the wrong epoch still runs to the end; it does not reach 0.9 on three classes.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from keras_image_classifier import KicError
from keras_image_classifier.cli import main
from keras_image_classifier.data import ImageBatches
from keras_image_classifier.dataset import SPLITS, split_digest
from keras_image_classifier.evaluate import MATRIX_FILE, METRICS_FILE, REPORT_FILE, evaluate
from keras_image_classifier.predict import predict
from keras_image_classifier.train import (
    CACHE_BUDGET_BYTES,
    HISTORY_FILE,
    MODEL_FILE,
    RUN_FILE,
    TrainConfig,
    load_model,
    load_run,
    train,
)


def test_training_records_everything_needed_to_repeat_it(trained: dict[str, Path]) -> None:
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    assert run["classes"] == ["circle", "square", "triangle"]
    assert run["image_size"] == 32
    assert run["parameters"] == 98_547
    assert (run["train_samples"], run["val_samples"]) == (252, 54)
    assert run["config"]["seed"] == 0
    assert run["environment"]["backend"] == "jax"
    assert run["best_val_accuracy"] >= 0.9
    assert (trained["run"] / MODEL_FILE).stat().st_size > 50_000
    with (trained["run"] / HISTORY_FILE).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == run["epochs_run"]
    assert float(rows[run["best_epoch"] - 1]["val_loss"]) == pytest.approx(run["best_val_loss"])


def test_evaluation_beats_the_baseline_by_a_wide_margin(trained: dict[str, Path]) -> None:
    result = evaluate(trained["run"])
    assert result["split"] == "test"
    assert result["samples"] == 54
    assert result["majority_baseline"] == pytest.approx(1 / 3)
    assert result["accuracy"] >= 0.9
    assert result["macro_f1"] >= 0.9
    assert sum(map(sum, result["confusion_matrix"])) == 54
    assert json.loads((trained["run"] / METRICS_FILE).read_text()) == result
    report = (trained["run"] / REPORT_FILE).read_text()
    assert "| circle |" in report
    assert chr(0x2014) not in report  # house style: no em dash in generated text
    assert (trained["run"] / MATRIX_FILE).read_bytes().startswith(b"\x89PNG")


def test_prediction_names_the_right_shape_and_survives_bad_files(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    junk = tmp_path / "junk.png"
    junk.write_text("not an image")
    files = [
        trained["raw"] / "circle" / "circle_00003.png",
        junk,
        trained["raw"] / "triangle" / "triangle_00003.png",
    ]
    records = predict(trained["run"], files, top_k=2)
    assert [record["file"] for record in records] == [str(file) for file in files]
    assert records[1] == {
        "file": str(junk),
        "error": "not a readable image (UnidentifiedImageError)",
    }
    assert records[0]["predictions"][0]["label"] == "circle"
    assert records[2]["predictions"][0]["label"] == "triangle"
    assert len(records[0]["predictions"]) == 2
    assert (
        records[0]["predictions"][0]["probability"] >= records[0]["predictions"][1]["probability"]
    )


def test_a_run_directory_is_never_reused(trained: dict[str, Path]) -> None:
    with pytest.raises(KicError, match="every run gets its own directory"):
        train(TrainConfig(data=str(trained["prepared"]), run_dir=str(trained["run"])))


def test_evaluation_refuses_data_that_changed_after_training(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    other = tmp_path / "other"
    assert main(["prepare", str(trained["raw"]), str(other), "--image-size", "24"]) == 0
    assert main(["split", str(other)]) == 0
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    moved = tmp_path / "run"
    moved.mkdir()
    (moved / MODEL_FILE).write_bytes((trained["run"] / MODEL_FILE).read_bytes())
    run["config"]["data"] = str(other)
    (moved / RUN_FILE).write_text(json.dumps(run))
    with pytest.raises(KicError, match="no longer matches"):
        evaluate(moved)


def test_evaluation_refuses_a_split_that_changed_after_training(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    """Re-splitting must be caught even when the class list and image size still match.

    Otherwise a re-split test set can silently overlap the training set: the guard
    would pass, evaluate would print a clean score, and it would be wrong.
    """
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    other = tmp_path / "other"
    assert (
        main(["prepare", str(trained["raw"]), str(other), "--image-size", str(run["image_size"])])
        == 0
    )
    assert main(["split", str(other), "--seed", "99"]) == 0
    moved = tmp_path / "run"
    moved.mkdir()
    (moved / MODEL_FILE).write_bytes((trained["run"] / MODEL_FILE).read_bytes())
    run["config"]["data"] = str(other)
    (moved / RUN_FILE).write_text(json.dumps(run))
    with pytest.raises(KicError, match="split"):
        evaluate(moved)


def test_the_same_seed_gives_the_same_training_curve(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    results = [
        train(TrainConfig(data=str(trained["prepared"]), run_dir=str(tmp_path / name), epochs=2))
        for name in ("first", "second")
    ]
    assert results[0]["best_val_loss"] == pytest.approx(results[1]["best_val_loss"], rel=1e-4)


def test_augmentation_is_wired_to_the_train_split_only(
    trained: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Augmented validation data would silently corrupt model selection; pin it shut."""
    created: list[ImageBatches] = []

    class RecordingBatches(ImageBatches):
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
            super().__init__(
                root,
                paths,
                labels,
                batch_size=batch_size,
                shuffle=shuffle,
                augment=augment,
                seed=seed,
                **kwargs,
            )
            created.append(self)

    monkeypatch.setattr("keras_image_classifier.data.ImageBatches", RecordingBatches)
    train(TrainConfig(data=str(trained["prepared"]), run_dir=str(tmp_path / "aug-check"), epochs=1))
    assert len(created) == 2
    assert created[0].augment is True  # train batches, built first
    assert created[1].augment is False  # validation batches, never augmented


def test_a_model_file_with_embedded_code_is_refused(tmp_path: Path) -> None:
    import keras

    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Lambda(lambda x: x * 2)(inputs)
    keras.Model(inputs, outputs).save(tmp_path / MODEL_FILE)
    with pytest.raises(ValueError, match=r"(?i)lambda|unsafe|deserializ"):
        load_model(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 0, "epochs must be between 1 and 1000"),
        ("batch_size", 5000, "batch size must be between 1 and 1024"),
        ("learning_rate", 0.0, "learning rate must be in"),
        ("width", 2, "width must be between 4 and 256"),
        ("dropout", 1.0, "dropout must be in"),
        ("patience", 0, "patience must be at least 1"),
    ],
)
def test_training_arguments_are_checked(field: str, value: float, message: str) -> None:
    with pytest.raises(KicError, match=message):
        TrainConfig(data="x", run_dir="y", **{field: value}).validate()  # type: ignore[arg-type]


def test_every_bad_argument_is_reported_at_once() -> None:
    with pytest.raises(KicError) as caught:
        TrainConfig(data="x", run_dir="y", epochs=0, width=2).validate()
    assert "epochs" in str(caught.value)
    assert "width" in str(caught.value)


def test_the_command_line_reaches_evaluate_and_predict(
    trained: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["evaluate", str(trained["run"]), "--split", "val"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["split"] == "val"
    assert summary["accuracy"] >= 0.9
    square = trained["raw"] / "square" / "square_00001.png"
    assert main(["predict", str(trained["run"]), str(square), "--top-k", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["predictions"][0]["label"] == "square"


def test_predict_checks_top_k_and_the_model_file(trained: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(KicError, match="top-k must be at least 1"):
        predict(trained["run"], [], top_k=0)
    assert predict(trained["run"], []) == []
    (tmp_path / RUN_FILE).write_text("{}")
    with pytest.raises(KicError, match=f"has no {MODEL_FILE}"):
        predict(tmp_path, [])


def test_a_truncated_run_file_is_a_user_error_not_a_traceback(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    """run.json lives in a directory the user owns; a half-written one must exit 2."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / MODEL_FILE).write_bytes((trained["run"] / MODEL_FILE).read_bytes())
    whole = (trained["run"] / RUN_FILE).read_text()
    (broken / RUN_FILE).write_text(whole[: len(whole) // 2])
    with pytest.raises(KicError, match=f"{RUN_FILE} is not valid JSON"):
        load_run(broken)


@pytest.mark.parametrize("field", ["config", "classes", "image_size"])
def test_a_run_file_missing_a_field_evaluate_needs_is_refused(
    trained: dict[str, Path], tmp_path: Path, field: str
) -> None:
    broken = tmp_path / f"broken-{field}"
    broken.mkdir()
    (broken / MODEL_FILE).write_bytes((trained["run"] / MODEL_FILE).read_bytes())
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    del run[field]
    (broken / RUN_FILE).write_text(json.dumps(run))
    with pytest.raises(KicError, match=f"has no {field}; is it a run directory"):
        load_run(broken)


def test_evaluate_explains_a_run_started_from_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # run.json keeps the data path as typed, usually relative. From another directory it
    # points nowhere, and "run kic prepare first" would send the user to redo good work.
    run = tmp_path / "run"
    run.mkdir()
    (run / MODEL_FILE).write_bytes(b"")
    document = {
        "config": {"data": "data/shapes"},
        "classes": ["a", "b"],
        "image_size": 32,
        "split_digest": "irrelevant",
    }
    (run / RUN_FILE).write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(KicError, match="run kic evaluate from the directory"):
        evaluate(run)


def test_a_run_records_the_digest_of_the_images_it_saw(trained: dict[str, Path]) -> None:
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    assert run["split_digest"] == split_digest(trained["prepared"])
    assert run["cached_in_memory"] is True


def test_evaluation_refuses_a_test_list_refilled_from_train(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    """Same seed, same ratios, same class list: only the membership is wrong."""
    prepared = tmp_path / "prepared"
    shutil.copytree(trained["prepared"], prepared)
    document = json.loads((prepared / SPLITS).read_text())
    document["test"] = document["train"][: len(document["test"])]
    document["train"] = document["train"][len(document["test"]) :]
    (prepared / SPLITS).write_text(json.dumps(document))
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    moved = tmp_path / "run"
    moved.mkdir()
    (moved / MODEL_FILE).write_bytes((trained["run"] / MODEL_FILE).read_bytes())
    run["config"]["data"] = str(prepared)
    (moved / RUN_FILE).write_text(json.dumps(run))
    with pytest.raises(KicError, match="not the images this run was trained on"):
        evaluate(moved)


def test_evaluation_refuses_a_prepared_file_whose_pixels_changed(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    prepared = tmp_path / "prepared"
    shutil.copytree(trained["prepared"], prepared)
    victim = json.loads((prepared / SPLITS).read_text())["test"][0]
    with Image.open(prepared / victim) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    pixels[0, 0] = 255 - pixels[0, 0]
    Image.fromarray(pixels).save(prepared / victim)
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    moved = tmp_path / "run"
    moved.mkdir()
    (moved / MODEL_FILE).write_bytes((trained["run"] / MODEL_FILE).read_bytes())
    run["config"]["data"] = str(prepared)
    (moved / RUN_FILE).write_text(json.dumps(run))
    with pytest.raises(KicError, match="does not match the hash"):
        evaluate(moved)


def test_a_run_from_an_earlier_version_is_refused_with_a_reason(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    del run["split_digest"]
    moved = tmp_path / "run"
    moved.mkdir()
    (moved / MODEL_FILE).write_bytes((trained["run"] / MODEL_FILE).read_bytes())
    (moved / RUN_FILE).write_text(json.dumps(run))
    with pytest.raises(KicError, match=r"earlier version.*retrain"):
        evaluate(moved)


def test_predict_still_works_with_a_run_from_an_earlier_version(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    """kic predict needs no split, so a 1.0.x run.json without split_digest still works."""
    moved = tmp_path / "run"
    moved.mkdir()
    (moved / MODEL_FILE).write_bytes((trained["run"] / MODEL_FILE).read_bytes())
    run = json.loads((trained["run"] / RUN_FILE).read_text())
    del run["split_digest"]
    (moved / RUN_FILE).write_text(json.dumps(run))
    square = trained["raw"] / "square" / "square_00001.png"
    records = predict(moved, [square], top_k=1)
    assert records[0]["predictions"][0]["label"] == "square"


def test_the_cache_is_switched_off_above_the_budget(
    trained: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("keras_image_classifier.train.CACHE_BUDGET_BYTES", 1)
    run_dir = tmp_path / "run"
    summary = train(TrainConfig(data=str(trained["prepared"]), run_dir=str(run_dir), epochs=1))
    assert summary["cached_in_memory"] is False
    assert json.loads((run_dir / RUN_FILE).read_text())["cached_in_memory"] is False
    assert CACHE_BUDGET_BYTES == 2 * 1024**3
