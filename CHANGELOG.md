# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-18

A rewrite of the 2024 coursework. The idea is unchanged; every file is new.

### Fixed (defects of the coursework version)

- Labels were assigned round-robin by file position (`classes[i % class_count]` in
  `split_dataset_single_class.py`), not by content, so the model was trained on noise. Labels now
  come from the class directory.
- Training passed `class_names=[selected_class]` to `image_dataset_from_directory`, so the
  network had a one-unit softmax that always outputs 1.0 and the reported accuracy meant nothing.
  In the same script `prediction.argmax() == 1` could never be true with one output, so
  "run model" never saved an image.
- Only a `test` split was ever created (`split_dataset_test_only`), while training required
  `train` and `validation`.
- `requirements.txt` was UTF-16 with a byte-order mark, which pip cannot read, and pinned 57
  packages from one Windows machine, including `tensorflow_intel==2.18.0` and `dlib==19.24.6`,
  which need a compiler elsewhere.
- Archives were extracted without a filter or any limit on files or bytes; MD5 was used for
  deduplication; images were opened with no format check and no size limit; every error was
  caught by a bare `except Exception`, printed and forgotten.
- `.gitignore` was 488 lines, 485 of them individual image file paths instead of patterns.

### Changed

- Five interactive scripts became the `kic` command with subcommands, flags, JSON output and
  exit codes. Comments, messages and documentation are in English.
- TensorFlow 2.18 became Keras 3 on the JAX backend; Anaconda and `requirements.txt` became
  `pyproject.toml` and `uv.lock`.
- CelebA, LFW, FER-2013 and photographs of people became EuroSAT (MIT licence, no people) and a
  synthetic shapes generator.
- The 11,169,347-parameter network with a Flatten layer became a 99,450-parameter network with
  global average pooling.

### Added

- Content-hash deduplication, conflict detection, seeded stratified split without file copies.
- Run records (`run.json`, `history.csv`), evaluation report with baseline, per-class metrics
  and confusion matrix.
- 100 tests plus one opt-in network test, including a real training run, mypy strict, ruff, bandit, pip-audit, gitleaks,
  CI on Linux and Windows for Python 3.12 and 3.13.
- README, architecture and testing notes, a EuroSAT benchmark under `docs/results/`,
  SECURITY.md, CITATION.cff, MIT licence.

### Removed

- `.idea/`, `test_environment.py` (replaced by `kic info`), `core_main_file.py` and the four
  scripts it launched.

## [0.1.0-coursework] - 2024-11-24

The coursework as submitted, kept under the tag `v0.1.0-coursework`.

[1.0.0]: https://github.com/tudorandrian/keras-image-classifier/releases/tag/v1.0.0
[0.1.0-coursework]: https://github.com/tudorandrian/keras-image-classifier/tree/v0.1.0-coursework
