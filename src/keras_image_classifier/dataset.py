"""Turn a folder-per-class directory into a clean, deduplicated, split dataset.

Nothing here imports Keras, so preparing data is fast and testable on its own.

    raw/<class>/<anything>.jpg  --prepare-->  prepared/<class>/<hash16>.png + manifest.json
    manifest.json               --split---->  splits.json (lists of paths, no copies)
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from keras_image_classifier import KicError
from keras_image_classifier.images import letterbox, load_rgb, pixel_hash

MANIFEST = "manifest.json"
SPLITS = "splits.json"
SPLIT_NAMES = ("train", "val", "test")
CLASS_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}")


@dataclass(frozen=True)
class Sample:
    path: str  # POSIX path relative to the prepared directory
    label: str
    sha256: str


@dataclass
class Manifest:
    image_size: int
    classes: list[str]
    samples: list[Sample]
    skipped: list[dict[str, str]] = field(default_factory=list)
    duplicates: int = 0
    conflicts: int = 0
    version: int = 1

    def counts(self) -> dict[str, int]:
        return {name: sum(s.label == name for s in self.samples) for name in self.classes}


def scan_classes(source: Path) -> dict[str, list[Path]]:
    """Map each class directory under source to its files, both in sorted order."""
    if not source.is_dir():
        raise KicError(f"{source} is not a directory")
    found: dict[str, list[Path]] = {}
    for directory in sorted(p for p in source.iterdir() if p.is_dir() and not p.is_symlink()):
        if directory.name.startswith("."):
            continue
        if not CLASS_NAME.fullmatch(directory.name):
            raise KicError(f"class directory name {directory.name!r} is not allowed")
        files = sorted(p for p in directory.rglob("*") if p.is_file() and not p.is_symlink())
        found[directory.name] = files
    if len(found) < 2:
        raise KicError(f"{source} needs at least two class directories, found {len(found)}")
    return found


def prepare(source: Path, destination: Path, *, image_size: int) -> Manifest:
    """Decode, letterbox and deduplicate every image; write PNG files and a manifest.

    A picture that appears twice under one label is kept once. A picture that appears
    under two labels cannot be trusted with either, so every copy is dropped.
    """
    if not 16 <= image_size <= 1024:  # four poolings need at least 16 px
        raise KicError("image size must be between 16 and 1024")
    if destination.exists() and any(destination.iterdir()):
        raise KicError(f"{destination} is not empty; choose a new directory or delete it")
    classes = scan_classes(source)
    manifest = Manifest(image_size=image_size, classes=list(classes), samples=[])
    label_of: dict[str, str] = {}
    conflicted: set[str] = set()
    for label, files in classes.items():
        (destination / label).mkdir(parents=True, exist_ok=True)
        for file in files:
            try:
                image = letterbox(load_rgb(file), image_size)
            except KicError as error:
                relative = file.relative_to(source).as_posix()
                manifest.skipped.append({"file": relative, "reason": str(error)})
                continue
            digest = pixel_hash(image)
            if digest in label_of:
                if label_of[digest] == label:
                    manifest.duplicates += 1
                else:
                    conflicted.add(digest)
                continue
            label_of[digest] = label
            relative_path = f"{label}/{digest[:16]}.png"
            image.save(destination / relative_path, format="PNG")
            manifest.samples.append(Sample(relative_path, label, digest))
    for sample in manifest.samples:
        if sample.sha256 in conflicted:
            (destination / sample.path).unlink()
    manifest.samples = [s for s in manifest.samples if s.sha256 not in conflicted]
    manifest.conflicts = len(conflicted)
    (destination / MANIFEST).write_text(json.dumps(asdict(manifest), indent=1), encoding="utf-8")
    return manifest


def load_manifest(prepared: Path) -> Manifest:
    try:
        raw = json.loads((prepared / MANIFEST).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KicError(f"{prepared} has no {MANIFEST}; run 'kic prepare' first") from None
    raw["samples"] = [Sample(**sample) for sample in raw["samples"]]
    return Manifest(**raw)


def split(
    prepared: Path, *, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0
) -> dict[str, list[str]]:
    """Stratified, seeded split written to splits.json; returns {split name: paths}.

    The order depends only on the content hashes and the seed, not on file system
    order, so the same data gives the same split on every machine.
    """
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise KicError("ratios must be three positive numbers that add up to 1")
    manifest = load_manifest(prepared)
    result: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for label in manifest.classes:
        members = sorted((s for s in manifest.samples if s.label == label), key=lambda s: s.sha256)
        if len(members) < 3:
            raise KicError(f"class {label!r} has {len(members)} usable images; at least 3 needed")
        # A reproducible shuffle, not a secret: the standard generator is the right tool.
        random.Random(f"{seed}:{label}").shuffle(members)  # noqa: S311  # nosec B311
        n_val = max(1, round(len(members) * ratios[1]))
        n_test = max(1, round(len(members) * ratios[2]))
        n_train = len(members) - n_val - n_test
        if n_train < 1:
            raise KicError(f"class {label!r} is too small for ratios {ratios}")
        result["train"] += [s.path for s in members[:n_train]]
        result["val"] += [s.path for s in members[n_train : n_train + n_val]]
        result["test"] += [s.path for s in members[n_train + n_val :]]
    document = {"version": 1, "seed": seed, "ratios": list(ratios), **result}
    (prepared / SPLITS).write_text(json.dumps(document, indent=1), encoding="utf-8")
    return result


def load_split(prepared: Path, name: str) -> tuple[list[str], list[int], list[str]]:
    """Return (paths, integer labels, class names) for one split."""
    manifest = load_manifest(prepared)
    try:
        document = json.loads((prepared / SPLITS).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KicError(f"{prepared} has no {SPLITS}; run 'kic split' first") from None
    if name not in SPLIT_NAMES:
        raise KicError(f"unknown split {name!r}; choose from {', '.join(SPLIT_NAMES)}")
    index = {label: i for i, label in enumerate(manifest.classes)}
    paths: list[str] = document[name]
    labels = [index[path.split("/", 1)[0]] for path in paths]
    return paths, labels, manifest.classes
