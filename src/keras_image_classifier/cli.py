"""The kic command. Every step is a subcommand with flags; nothing asks questions.

Exit status: 0 success, 2 a problem the user can fix (printed without a traceback).
Anything else is a bug and keeps its traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from keras_image_classifier import KicError, __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kic", description="Reproducible image classification with Keras 3."
    )
    parser.add_argument("--version", action="version", version=f"kic {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    synth = commands.add_parser("synth", help="generate a synthetic shapes data set (offline)")
    synth.add_argument("destination", type=Path)
    synth.add_argument("--per-class", type=int, default=200)
    synth.add_argument("--size", type=int, default=48)
    synth.add_argument("--seed", type=int, default=0)

    fetch = commands.add_parser("fetch-eurosat", help="download EuroSAT RGB (95 MB, MIT licence)")
    fetch.add_argument("destination", type=Path)

    prepare = commands.add_parser("prepare", help="decode, letterbox and deduplicate raw/<class>/")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("--image-size", type=int, default=64)

    split = commands.add_parser("split", help="stratified, seeded train/val/test split")
    split.add_argument("prepared", type=Path)
    split.add_argument(
        "--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST")
    )
    split.add_argument("--seed", type=int, default=0)

    train = commands.add_parser("train", help="train the CNN and record the run")
    train.add_argument("prepared", type=Path)
    train.add_argument("run_dir", type=Path)
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--width", type=int, default=16)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--patience", type=int, default=4)
    train.add_argument("--no-augment", action="store_true")

    evaluate = commands.add_parser("evaluate", help="metrics, confusion matrix and report")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--split", default="test", choices=("train", "val", "test"))

    predict = commands.add_parser("predict", help="classify image files, print JSON")
    predict.add_argument("run_dir", type=Path)
    predict.add_argument("files", type=Path, nargs="+")
    predict.add_argument("--top-k", type=int, default=3)

    commands.add_parser("info", help="print versions and the active Keras backend")
    return parser


def run(args: argparse.Namespace) -> object:
    if args.command == "synth":
        from keras_image_classifier.synth import generate

        total = generate(args.destination, per_class=args.per_class, size=args.size, seed=args.seed)
        return {"images": total, "destination": str(args.destination)}
    if args.command == "fetch-eurosat":
        from keras_image_classifier.fetch import fetch_eurosat

        return {"images": fetch_eurosat(args.destination), "destination": str(args.destination)}
    if args.command == "prepare":
        from keras_image_classifier.dataset import prepare

        manifest = prepare(args.source, args.destination, image_size=args.image_size)
        return {
            "classes": manifest.counts(),
            "skipped": len(manifest.skipped),
            "duplicates": manifest.duplicates,
            "conflicts": manifest.conflicts,
        }
    if args.command == "split":
        from keras_image_classifier.dataset import split

        result = split(args.prepared, ratios=tuple(args.ratios), seed=args.seed)
        return {name: len(paths) for name, paths in result.items()}
    if args.command == "train":
        from keras_image_classifier.train import TrainConfig, train

        config = TrainConfig(
            data=str(args.prepared),
            run_dir=str(args.run_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            width=args.width,
            dropout=args.dropout,
            augment=not args.no_augment,
            patience=args.patience,
        )
        summary = train(config)
        keys = ("epochs_run", "best_epoch", "best_val_loss", "best_val_accuracy", "train_seconds")
        return {key: summary[key] for key in keys}
    if args.command == "evaluate":
        from keras_image_classifier.evaluate import evaluate

        result = evaluate(args.run_dir, split=args.split)
        return {
            key: result[key]
            for key in ("split", "samples", "accuracy", "macro_f1", "majority_baseline")
        }
    if args.command == "predict":
        from keras_image_classifier.predict import predict

        return predict(args.run_dir, args.files, top_k=args.top_k)
    from keras_image_classifier.train import environment

    return environment()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except KicError as error:
        print(f"kic: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1))
    return 0
