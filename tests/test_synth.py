from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from keras_image_classifier import KicError
from keras_image_classifier.synth import SHAPES, draw_shape, generate


def digest(directory: Path) -> str:
    combined = hashlib.sha256()
    for file in sorted(directory.rglob("*.png")):
        combined.update(file.read_bytes())
    return combined.hexdigest()


def test_generate_writes_the_requested_number_per_class(tmp_path: Path) -> None:
    assert generate(tmp_path, per_class=5, size=16) == 15
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(SHAPES)
    assert len(list((tmp_path / "circle").iterdir())) == 5


def test_the_same_seed_gives_the_same_bytes(tmp_path: Path) -> None:
    generate(tmp_path / "a", per_class=4, size=16, seed=3)
    generate(tmp_path / "b", per_class=4, size=16, seed=3)
    generate(tmp_path / "c", per_class=4, size=16, seed=4)
    assert digest(tmp_path / "a") == digest(tmp_path / "b") != digest(tmp_path / "c")


def test_shapes_differ_only_by_shape() -> None:
    drawn = {shape: np.asarray(draw_shape(shape, 48, np.random.default_rng(0))) for shape in SHAPES}
    bright = {shape: int((pixels.max(axis=2) >= 120).sum()) for shape, pixels in drawn.items()}
    assert bright["square"] > bright["circle"] > bright["triangle"] > 0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"per_class": 2}, "at least 3"),
        ({"size": 8}, "between 16 and 512"),
    ],
)
def test_arguments_are_checked(tmp_path: Path, options: dict[str, int], message: str) -> None:
    with pytest.raises(KicError, match=message):
        generate(tmp_path, **options)


def test_unknown_shape_is_refused() -> None:
    with pytest.raises(KicError, match="unknown shape 'star'"):
        draw_shape("star", 32, np.random.default_rng(0))


def test_generate_never_writes_into_a_used_directory(tmp_path: Path) -> None:
    # 20 then 3 per class used to leave 60 files on disk while reporting 9, so the next
    # `kic prepare` read a set that matched neither command.
    generate(tmp_path / "raw", per_class=20, size=16)
    with pytest.raises(KicError, match="is not empty"):
        generate(tmp_path / "raw", per_class=3, size=16)
    assert sum(1 for _ in (tmp_path / "raw").rglob("*.png")) == 60
