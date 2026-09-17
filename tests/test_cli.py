from __future__ import annotations

import json
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
