"""Turn a folder-per-class directory into a clean, deduplicated, split dataset.

Nothing here imports Keras, so preparing data is fast and testable on its own.

    raw/<class>/<anything>.jpg  --prepare-->  prepared/<class>/<hash16>.png + manifest.json
    manifest.json               --split---->  splits.json (lists of paths, no copies)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from keras_image_classifier import KicError
from keras_image_classifier.images import letterbox, load_rgb, pixel_hash

MANIFEST = "manifest.json"
SPLITS = "splits.json"
SPLIT_NAMES = ("train", "val", "test")
CLASS_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}")
HEX64 = re.compile(r"[0-9a-f]{64}")


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


def read_json(path: Path, missing: str) -> dict[str, Any]:
    """Load one JSON artefact as an object, or raise KicError saying what is wrong with it.

    Artefacts are written by this tool but live in a directory the user owns, so they
    can be truncated by a full disk, hand-edited or half-copied. That is a problem the
    user can fix, which means KicError and exit 2, not a traceback: `cli.py` promises
    that anything else is a bug in the tool.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise KicError(missing) from None
    try:
        document = json.loads(text)
    except ValueError as error:  # json.JSONDecodeError is a ValueError
        raise KicError(f"{path} is not valid JSON ({error}); regenerate it") from None
    if not isinstance(document, dict):
        raise KicError(f"{path} should hold a JSON object, found {type(document).__name__}")
    return document


def load_manifest(prepared: Path) -> Manifest:
    """Read manifest.json and check that it is one `prepare` could have written.

    The manifest decides where the loader opens files. A hand-edited path such as
    `circle/../../x.png` keeps a valid label and leaves the prepared directory, so
    every sample must be exactly `<label>/<sha256[:16]>.png` with a label from the
    class list, and every class name must pass the same rule as a raw directory.
    """
    raw = read_json(prepared / MANIFEST, f"{prepared} has no {MANIFEST}; run 'kic prepare' first")
    try:
        raw["samples"] = [Sample(**sample) for sample in raw["samples"]]
        manifest = Manifest(**raw)
    except (KeyError, TypeError) as error:
        raise KicError(
            f"{prepared / MANIFEST} is not a usable manifest ({error!r}); run 'kic prepare' again"
        ) from None
    for name in manifest.classes:
        if not isinstance(name, str) or not CLASS_NAME.fullmatch(name):
            raise KicError(f"{prepared / MANIFEST}: class name {name!r} is not allowed")
    seen_hashes: set[str] = set()
    for sample in manifest.samples:
        message = (
            f"{prepared / MANIFEST} lists {sample.path!r}, which 'kic prepare' would not "
            "have written; run 'kic prepare' again"
        )
        # The hex check must happen before any slicing of sha256, so a non-string value
        # (or one too short to slice meaningfully) fails cleanly instead of raising TypeError.
        if not isinstance(sample.sha256, str) or not HEX64.fullmatch(sample.sha256):
            raise KicError(message)
        expected = f"{sample.label}/{sample.sha256[:16]}.png"
        if sample.label not in manifest.classes or sample.path != expected:
            raise KicError(message)
        # `prepare` keeps one copy of a duplicate; a manifest that lists the same content
        # hash twice was hand-edited (or copied) after the fact, and would let one picture
        # count twice in a split or straddle train and test under two different labels.
        if sample.sha256 in seen_hashes:
            raise KicError(
                f"{prepared / MANIFEST} lists the same picture twice ({sample.sha256}); "
                "run 'kic prepare' again"
            )
        seen_hashes.add(sample.sha256)
    return manifest


def split_identity(prepared: Path) -> dict[str, Any]:
    """The version, seed and ratios that say which split a splits.json is.

    `train` records this triple and `evaluate` compares it, so both read it the same way
    and a hand-edited file fails with one message instead of two different KeyErrors.
    """
    document = read_json(prepared / SPLITS, f"{prepared} has no {SPLITS}; run 'kic split' first")
    try:
        return {key: document[key] for key in ("version", "seed", "ratios")}
    except KeyError as error:
        raise KicError(f"{prepared / SPLITS} has no {error} field; run 'kic split' again") from None


def load_splits(prepared: Path) -> tuple[Manifest, dict[str, list[Sample]]]:
    """Read splits.json as three lists of manifest samples, or say what is wrong with it.

    splits.json is a plain list of paths the user can edit. Two things must hold before
    a run can trust it: every entry names a sample the manifest knows, and no entry
    appears twice, in one split or across two. The second rule is what keeps a test
    score honest; `split_identity` alone cannot see a test list refilled from train.
    """
    manifest = load_manifest(prepared)
    document = read_json(prepared / SPLITS, f"{prepared} has no {SPLITS}; run 'kic split' first")
    by_path = {sample.path: sample for sample in manifest.samples}
    seen: dict[str, str] = {}
    splits: dict[str, list[Sample]] = {}
    for name in SPLIT_NAMES:
        entries = document.get(name)
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise KicError(
                f"{prepared / SPLITS} does not list usable paths for the {name!r} split; "
                "run 'kic split' again"
            )
        members = []
        for entry in entries:
            if entry not in by_path:
                raise KicError(
                    f"{prepared / SPLITS} lists {entry!r} in the {name!r} split, which is not "
                    f"in {MANIFEST}; run 'kic split' again"
                )
            if entry in seen:
                raise KicError(
                    f"{prepared / SPLITS} lists {entry!r} twice ({seen[entry]} and {name}); "
                    "run 'kic split' again"
                )
            seen[entry] = name
            members.append(by_path[entry])
        splits[name] = members
    return manifest, splits


def split_digest(prepared: Path) -> str:
    """SHA-256 over each split's [path, content hash] pairs, in order.

    `train` records it and `evaluate` compares it, so a run is tied to the exact
    images and labels it saw, not only to the seed and ratios that were meant to
    produce them. The path is included, not only the content hash, because the label
    is the first path component: relabelling a sample without touching its pixels
    must still change the digest, or a run could be scored against labels it never saw.
    """
    _, splits = load_splits(prepared)
    text = json.dumps(
        {name: [[s.path, s.sha256] for s in splits[name]] for name in SPLIT_NAMES},
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()


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


def load_split(prepared: Path, name: str) -> tuple[list[Sample], list[str]]:
    """Return (samples of one split, class names), after `load_splits` has checked the file."""
    if name not in SPLIT_NAMES:
        raise KicError(f"unknown split {name!r}; choose from {', '.join(SPLIT_NAMES)}")
    manifest, splits = load_splits(prepared)
    return splits[name], manifest.classes
