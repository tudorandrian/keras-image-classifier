from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from keras_image_classifier.data import ImageBatches
from keras_image_classifier.dataset import load_split, prepare, split
from keras_image_classifier.model import build_cnn


@pytest.fixture
def train_split(raw: Path, tmp_path: Path) -> tuple[Path, list[str], list[int]]:
    prepared = tmp_path / "prepared"
    prepare(raw, prepared, image_size=24)
    split(prepared)
    paths, labels, _ = load_split(prepared, "train")
    return prepared, paths, labels


def test_batches_cover_every_sample_once(train_split: tuple[Path, list[str], list[int]]) -> None:
    root, paths, labels = train_split
    batches = ImageBatches(root, paths, labels, batch_size=16, shuffle=False)
    assert len(batches) == 3  # 42 samples: 16 + 16 + 10
    pixels, targets = batches[2]
    assert pixels.shape == (10, 24, 24, 3)
    assert pixels.dtype == np.float32
    assert pixels.max() > 1.0  # raw 0-255: the model rescales, the loader does not
    assert targets.dtype == np.int32
    assert np.concatenate([batches[i][1] for i in range(3)]).tolist() == labels


def test_shuffling_is_seeded_and_changes_every_epoch(
    train_split: tuple[Path, list[str], list[int]],
) -> None:
    root, paths, labels = train_split
    first = ImageBatches(root, paths, labels, batch_size=42, shuffle=True, seed=7)
    second = ImageBatches(root, paths, labels, batch_size=42, shuffle=True, seed=7)
    assert first[0][1].tolist() == second[0][1].tolist() != labels
    before = first[0][1].tolist()
    first.on_epoch_end()
    assert first[0][1].tolist() != before
    assert sorted(first[0][1].tolist()) == sorted(labels)


def test_augmentation_only_mirrors_left_to_right(
    train_split: tuple[Path, list[str], list[int]],
) -> None:
    root, paths, labels = train_split
    plain = ImageBatches(root, paths, labels, batch_size=42, shuffle=False)[0][0]
    augmented = ImageBatches(root, paths, labels, batch_size=42, shuffle=False, augment=True)[0][0]
    same = [np.array_equal(a, p) for a, p in zip(augmented, plain, strict=True)]
    mirrored = [np.array_equal(a, p[:, ::-1]) for a, p in zip(augmented, plain, strict=True)]
    assert all(s or m for s, m in zip(same, mirrored, strict=True))
    assert 5 < sum(mirrored) < 37


def test_decoded_images_are_cached(train_split: tuple[Path, list[str], list[int]]) -> None:
    root, paths, labels = train_split
    batches = ImageBatches(root, paths, labels, batch_size=16, shuffle=False)
    _ = batches[0]
    assert sorted(batches.cache) == list(range(16))


def test_model_outputs_a_probability_per_class() -> None:
    model = build_cnn(24, 5)
    output = np.asarray(model(np.zeros((2, 24, 24, 3), dtype=np.float32)))
    assert output.shape == (2, 5)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


def test_parameter_count_does_not_depend_on_the_image_size() -> None:
    assert build_cnn(32, 3).count_params() == build_cnn(128, 3).count_params() == 98_547
    assert not any("flatten" in layer.name for layer in build_cnn(32, 3).layers)


def test_the_model_takes_raw_pixels() -> None:
    names = [layer.__class__.__name__ for layer in build_cnn(32, 3).layers]
    assert names[1] == "Rescaling"
