"""One real training run on synthetic shapes, checked from every side.

The numbers are the point. A pipeline that shuffles labels, leaks the test split or
saves the wrong epoch still runs to the end; it does not reach 0.9 on three classes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from keras_image_classifier import KicError
from keras_image_classifier.cli import main
from keras_image_classifier.evaluate import MATRIX_FILE, METRICS_FILE, REPORT_FILE, evaluate
from keras_image_classifier.predict import predict
from keras_image_classifier.train import (
    HISTORY_FILE,
    MODEL_FILE,
    RUN_FILE,
    TrainConfig,
    load_model,
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


def test_the_same_seed_gives_the_same_training_curve(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    results = [
        train(TrainConfig(data=str(trained["prepared"]), run_dir=str(tmp_path / name), epochs=2))
        for name in ("first", "second")
    ]
    assert results[0]["best_val_loss"] == pytest.approx(results[1]["best_val_loss"], rel=1e-4)


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
