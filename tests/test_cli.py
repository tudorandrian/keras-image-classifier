from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from keras_image_classifier import __version__
from keras_image_classifier.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == f"kic {__version__}"


@pytest.mark.parametrize("argv", [[], ["unknown"], ["train"], ["split", "x", "--ratios", "1"]])
def test_usage_errors_exit_with_status_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2


def test_user_errors_print_one_line_and_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["prepare", str(tmp_path / "missing"), str(tmp_path / "out")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("kic: error: ")
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err


def test_every_step_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", str(tmp_path / "raw"), "--per-class", "4", "--size", "16"]) == 0
    assert json.loads(capsys.readouterr().out)["images"] == 12
    assert main(["prepare", str(tmp_path / "raw"), str(tmp_path / "p"), "--image-size", "16"]) == 0
    assert json.loads(capsys.readouterr().out)["classes"] == {
        "circle": 4,
        "square": 4,
        "triangle": 4,
    }
    assert main(["split", str(tmp_path / "p"), "--ratios", "0.5", "0.25", "0.25"]) == 0
    assert json.loads(capsys.readouterr().out) == {"train": 6, "val": 3, "test": 3}


def test_predict_and_evaluate_explain_a_wrong_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["evaluate", str(tmp_path)]) == 2
    assert main(["predict", str(tmp_path), "a.png"]) == 2
    assert capsys.readouterr().err.count("has no run.json") == 2


def test_info_reports_the_backend(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "jax"


def test_the_module_can_be_run_without_the_console_script() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "keras_image_classifier", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (done.returncode, done.stdout.strip()) == (0, f"kic {__version__}")


def test_fetch_eurosat_is_wired_to_the_downloader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("keras_image_classifier.fetch.fetch_eurosat", lambda destination: 27000)
    assert main(["fetch-eurosat", str(tmp_path / "eurosat")]) == 0
    assert json.loads(capsys.readouterr().out)["images"] == 27000


def test_train_prints_json_and_sends_progress_to_standard_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Keras prints one line per epoch. On standard output it would break every script that
    # parses the JSON, as `kic train ... | jq` or `kic train ... > result.json` do.
    assert main(["synth", str(tmp_path / "raw"), "--per-class", "8", "--size", "32"]) == 0
    assert main(["prepare", str(tmp_path / "raw"), str(tmp_path / "p"), "--image-size", "32"]) == 0
    assert main(["split", str(tmp_path / "p")]) == 0
    capsys.readouterr()
    assert main(["train", str(tmp_path / "p"), str(tmp_path / "run"), "--epochs", "1"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["epochs_run"] == 1
    assert "val_loss" in captured.err  # the per-epoch progress line


@pytest.mark.parametrize("backend", ["tensorflow", "torch"])
def test_a_backend_that_is_not_installed_is_explained(
    backend: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def import_fails(args: object) -> object:
        raise ModuleNotFoundError(f"No module named '{backend}'", name=backend)

    monkeypatch.setenv("KERAS_BACKEND", backend)
    monkeypatch.setattr("keras_image_classifier.cli.run", import_fails)
    assert main(["info"]) == 2
    captured = capsys.readouterr()
    assert captured.err.count("\n") == 1
    assert f"KERAS_BACKEND is '{backend}'" in captured.err
    assert "unset KERAS_BACKEND" in captured.err


def test_an_unrelated_missing_module_is_still_a_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    def import_fails(args: object) -> object:
        raise ModuleNotFoundError("No module named 'yaml'", name="yaml")

    monkeypatch.setattr("keras_image_classifier.cli.run", import_fails)
    with pytest.raises(ModuleNotFoundError):
        main(["info"])


def test_a_foreign_backend_left_in_the_environment_is_not_a_traceback() -> None:
    # The real thing, end to end: Keras reads KERAS_BACKEND once, at import, so this has
    # to run in a fresh interpreter. A user coming from TensorFlow often has it set.
    done = subprocess.run(
        [sys.executable, "-m", "keras_image_classifier", "info"],
        capture_output=True,
        text=True,
        env={**os.environ, "KERAS_BACKEND": "tensorflow"},
        check=False,
    )
    assert done.returncode == 2
    assert done.stderr.startswith("kic: error: KERAS_BACKEND is 'tensorflow'")
    assert "Traceback" not in done.stderr
