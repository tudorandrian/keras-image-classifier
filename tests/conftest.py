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
