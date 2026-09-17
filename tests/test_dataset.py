from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from keras_image_classifier import KicError
from keras_image_classifier.dataset import (
    MANIFEST,
    SPLITS,
    load_manifest,
    load_split,
    prepare,
    scan_classes,
    split,
)


def test_scan_needs_a_directory_with_two_classes(tmp_path: Path) -> None:
    with pytest.raises(KicError, match="is not a directory"):
        scan_classes(tmp_path / "missing")
    (tmp_path / "only").mkdir()
    with pytest.raises(KicError, match="at least two class directories, found 1"):
        scan_classes(tmp_path)


def test_scan_skips_hidden_directories_and_loose_files(raw: Path) -> None:
    (raw / ".cache").mkdir()
    (raw / "notes.txt").write_text("not a class")
    assert list(scan_classes(raw)) == ["circle", "square", "triangle"]


@pytest.mark.parametrize("name", ["semi;colon", "-leading-dash", "x" * 65])
def test_scan_refuses_unsafe_class_names(raw: Path, name: str) -> None:
    (raw / name).mkdir()
    with pytest.raises(KicError, match="is not allowed"):
        scan_classes(raw)


def test_prepare_writes_hash_named_png_files_and_a_manifest(raw: Path, tmp_path: Path) -> None:
    manifest = prepare(raw, tmp_path / "out", image_size=24)
    assert manifest.counts() == {"circle": 20, "square": 20, "triangle": 20}
    sample = manifest.samples[0]
    assert sample.path == f"{sample.label}/{sample.sha256[:16]}.png"
    with Image.open(tmp_path / "out" / sample.path) as image:
        assert image.size == (24, 24)
    assert load_manifest(tmp_path / "out") == manifest


def test_prepare_records_why_a_file_was_skipped(raw: Path, tmp_path: Path) -> None:
    (raw / "circle" / "readme.txt").write_text("hello")
    manifest = prepare(raw, tmp_path / "out", image_size=24)
    assert manifest.skipped == [
        {"file": "circle/readme.txt", "reason": "not a readable image (UnidentifiedImageError)"}
    ]
    assert manifest.counts()["circle"] == 20


def test_prepare_keeps_one_copy_of_a_duplicate(raw: Path, tmp_path: Path) -> None:
    shutil.copy(raw / "circle" / "circle_00000.png", raw / "circle" / "again.png")
    manifest = prepare(raw, tmp_path / "out", image_size=24)
    assert manifest.duplicates == 1
    assert manifest.counts()["circle"] == 20


def test_prepare_drops_a_picture_that_carries_two_labels(raw: Path, tmp_path: Path) -> None:
    shutil.copy(raw / "circle" / "circle_00000.png", raw / "square" / "really_a_circle.png")
    manifest = prepare(raw, tmp_path / "out", image_size=24)
    assert manifest.conflicts == 1
    assert manifest.counts() == {"circle": 19, "square": 20, "triangle": 20}
    on_disk = {
        p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*.png")
    }
    assert on_disk == {s.path for s in manifest.samples}


def test_prepare_never_writes_into_a_used_directory(raw: Path, tmp_path: Path) -> None:
    prepare(raw, tmp_path / "out", image_size=24)
    with pytest.raises(KicError, match="is not empty"):
        prepare(raw, tmp_path / "out", image_size=24)


@pytest.mark.parametrize("size", [0, 15, 1025])
def test_prepare_bounds_the_image_size(raw: Path, tmp_path: Path, size: int) -> None:
    with pytest.raises(KicError, match="between 16 and 1024"):
        prepare(raw, tmp_path / "out", image_size=size)


def test_split_is_stratified_disjoint_and_complete(raw: Path, tmp_path: Path) -> None:
    manifest = prepare(raw, tmp_path / "out", image_size=24)
    parts = split(tmp_path / "out")
    assert {name: len(paths) for name, paths in parts.items()} == {"train": 42, "val": 9, "test": 9}
    for name in ("val", "test"):
        assert sorted(path.split("/")[0] for path in parts[name]) == (
            ["circle"] * 3 + ["square"] * 3 + ["triangle"] * 3
        )
    everything = parts["train"] + parts["val"] + parts["test"]
    assert len(set(everything)) == len(everything) == len(manifest.samples)


def test_split_depends_on_content_and_seed_only(raw: Path, tmp_path: Path) -> None:
    renamed = tmp_path / "renamed"
    shutil.copytree(raw, renamed)
    for number, file in enumerate(sorted(renamed.rglob("*.png"), reverse=True)):
        file.rename(file.with_name(f"z{number:03d}.png"))
    prepare(raw, tmp_path / "a", image_size=24)
    prepare(renamed, tmp_path / "b", image_size=24)
    assert split(tmp_path / "a", seed=5) == split(tmp_path / "b", seed=5)
    assert split(tmp_path / "a", seed=5) != split(tmp_path / "a", seed=6)


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (1.0, 0.0, 0.0), (0.9, 0.2, -0.1)])
def test_split_checks_the_ratios(raw: Path, tmp_path: Path, ratios: tuple[float, ...]) -> None:
    prepare(raw, tmp_path / "out", image_size=24)
    with pytest.raises(KicError, match="add up to 1"):
        split(tmp_path / "out", ratios=ratios)  # type: ignore[arg-type]


def test_split_needs_three_images_per_class(raw: Path, tmp_path: Path) -> None:
    for file in sorted((raw / "square").iterdir())[2:]:
        file.unlink()
    prepare(raw, tmp_path / "out", image_size=24)
    with pytest.raises(KicError, match="class 'square' has 2 usable images"):
        split(tmp_path / "out")


def test_load_split_returns_labels_that_match_the_paths(raw: Path, tmp_path: Path) -> None:
    prepare(raw, tmp_path / "out", image_size=24)
    split(tmp_path / "out")
    paths, labels, classes = load_split(tmp_path / "out", "val")
    assert classes == ["circle", "square", "triangle"]
    assert [classes[label] for label in labels] == [path.split("/")[0] for path in paths]
    document = json.loads((tmp_path / "out" / SPLITS).read_text())
    assert document["seed"] == 0
    assert document["ratios"] == [0.7, 0.15, 0.15]


def test_missing_files_are_explained(raw: Path, tmp_path: Path) -> None:
    with pytest.raises(KicError, match=f"has no {MANIFEST}"):
        split(tmp_path)
    prepare(raw, tmp_path / "out", image_size=24)
    with pytest.raises(KicError, match=f"has no {SPLITS}"):
        load_split(tmp_path / "out", "train")
    split(tmp_path / "out")
    with pytest.raises(KicError, match="unknown split 'holdout'"):
        load_split(tmp_path / "out", "holdout")


def test_split_refuses_ratios_that_leave_no_training_data(raw: Path, tmp_path: Path) -> None:
    for file in sorted((raw / "square").iterdir())[4:]:
        file.unlink()
    prepare(raw, tmp_path / "out", image_size=24)
    with pytest.raises(KicError, match="class 'square' is too small"):
        split(tmp_path / "out", ratios=(0.1, 0.45, 0.45))  # 4 images: 2 val + 2 test, 0 train
