from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from keras_image_classifier import KicError
from keras_image_classifier.data import ImageBatches
from keras_image_classifier.dataset import load_manifest, load_split, prepare, split
from keras_image_classifier.model import build_cnn


@pytest.fixture
def train_split(raw: Path, tmp_path: Path) -> tuple[Path, list[str], list[int]]:
    prepared = tmp_path / "prepared"
    prepare(raw, prepared, image_size=24)
    split(prepared)
    samples, classes = load_split(prepared, "train")
    paths = [s.path for s in samples]
    labels = [classes.index(s.label) for s in samples]
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


def test_a_prepared_file_whose_pixels_changed_is_refused_before_training(
    train_split: tuple[Path, list[str], list[int]],
) -> None:
    root, paths, labels = train_split
    digests = {s.path: s.sha256 for s in load_manifest(root).samples}
    with Image.open(root / paths[3]) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    pixels[0, 0] = 255 - pixels[0, 0]
    Image.fromarray(pixels).save(root / paths[3])  # same size, same format, one pixel off
    batches = ImageBatches(
        root, paths, labels, batch_size=16, shuffle=False, digests=[digests[p] for p in paths]
    )
    with pytest.raises(KicError, match=f"{re.escape(paths[3])}.*does not match"):
        batches.verify()


def test_verify_fills_the_cache_and_passes_intact_data(
    train_split: tuple[Path, list[str], list[int]],
) -> None:
    root, paths, labels = train_split
    digests = {s.path: s.sha256 for s in load_manifest(root).samples}
    batches = ImageBatches(
        root, paths, labels, batch_size=16, shuffle=False, digests=[digests[p] for p in paths]
    )
    batches.verify()
    assert len(batches.cache) == len(paths)


def test_the_cache_can_be_switched_off(train_split: tuple[Path, list[str], list[int]]) -> None:
    root, paths, labels = train_split
    batches = ImageBatches(root, paths, labels, batch_size=16, shuffle=False, cache=False)
    batches.verify()
    assert batches.cache == {}
    assert batches[0][0].shape == (16, 24, 24, 3)
    assert batches.cache == {}


def test_a_jpeg_saved_under_the_expected_png_name_is_refused(
    train_split: tuple[Path, list[str], list[int]],
) -> None:
    root, paths, labels = train_split
    with Image.open(root / paths[0]) as image:
        rgb = image.convert("RGB")
    rgb.save(root / paths[0], format="JPEG")  # same name, wrong container
    batches = ImageBatches(root, paths, labels, batch_size=16, shuffle=False)
    with pytest.raises(KicError, match=f"{re.escape(paths[0])}.*cannot be read"):
        batches.verify()


def test_a_decompression_bomb_is_refused(
    train_split: tuple[Path, list[str], list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)
    root, paths, labels = train_split
    batches = ImageBatches(root, paths, labels, batch_size=16, shuffle=False)
    with pytest.raises(KicError, match=f"{re.escape(paths[0])}.*cannot be read"):
        batches.verify()


def test_a_prepared_file_that_disappeared_is_a_user_error(
    train_split: tuple[Path, list[str], list[int]],
) -> None:
    root, paths, labels = train_split
    (root / paths[0]).unlink()
    batches = ImageBatches(root, paths, labels, batch_size=16, shuffle=False)
    with pytest.raises(KicError, match=f"{re.escape(paths[0])}.*cannot be read"):
        batches.verify()


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
