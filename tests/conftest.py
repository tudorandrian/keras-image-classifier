from __future__ import annotations

from pathlib import Path

import pytest

from keras_image_classifier.synth import generate


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """Twenty 32 px images for each of the three shapes."""
    target = tmp_path / "raw"
    generate(target, per_class=20, size=32, seed=1)
    return target


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One real training run, driven through the command line, shared by the slow tests."""
    from keras_image_classifier.cli import main

    root = tmp_path_factory.mktemp("pipeline")
    paths = {"raw": root / "raw", "prepared": root / "prepared", "run": root / "run"}
    assert main(["synth", str(paths["raw"]), "--per-class", "120", "--size", "32"]) == 0
    assert main(["prepare", str(paths["raw"]), str(paths["prepared"]), "--image-size", "32"]) == 0
    assert main(["split", str(paths["prepared"])]) == 0
    assert main(["train", str(paths["prepared"]), str(paths["run"]), "--epochs", "20"]) == 0
    return paths
