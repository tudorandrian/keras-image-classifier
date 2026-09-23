# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-09-23

An integrity release, from a review of the 1.0.2 code: a saved run is now tied to the exact
images it was trained and evaluated on. The EuroSAT numbers below were measured again with this
version; see the README.

### Compatibility

The CLI, prepared data sets and `splits.json` are unchanged. A run directory written by 1.0.x
can still be used by `kic predict`, but must be retrained to be evaluated, because it carries no
`split_digest`. `kic synth` now refuses a non-empty destination directory.

### Changed

- `run.json` records `split_digest`, the SHA-256 over each split's paths and content hashes, in
  order, and `cached_in_memory`. `kic evaluate` compares the digest and refuses a run directory
  written by 1.0.x, which has none; retrain to evaluate such a run.
- `kic synth` refuses a non-empty destination, as `prepare` and `train` already did. Running it
  twice into the same directory used to keep the old files and report only the new count.
- GitHub Actions are pinned to commit SHAs instead of tags.

### Fixed

- `splits.json` was trusted as written: an entry such as `circle/../../x.png` kept a valid
  label and opened a file outside the prepared directory, and a test list refilled from training
  paths passed the seed-and-ratios check and was scored as held out. Every entry must now be a
  manifest path and appear exactly once across the three splits.
- `manifest.json` paths are checked to be the ones `kic prepare` writes, and every prepared file
  is compared with its recorded pixel hash before training or scoring. A manifest that lists the
  same content hash twice, which `kic prepare` never writes but a hand-edited file could, is now
  refused as well.
- The loader opened a prepared file with no format check, so a file replaced under its expected
  name after preparation could still be decoded as long as some format matched. It now opens
  each file as PNG only, since `kic prepare` never writes anything else, and a replacement large
  enough to trip Pillow's decompression-bomb ceiling is refused the same way any other unreadable
  file is, instead of raising a warning or an uncaught error.
- `kic predict` stacked every decoded input before inference; it now runs batches of 128.
- The decoded-image cache is turned off above 2 GiB of pixels for train plus val.

### Benchmark

- EuroSAT, re-measured with 1.1.0: identical accuracy 0.9491 and macro F1 0.9473, the same
  per-class figures and confusion matrix as 1.0.0, training 1,261.6 s (21 min) on the same
  laptop.

## [1.0.2] - 2026-09-18

Fixes found by installing the project from a fresh clone and following the README the way a new
user would. The pipeline's results do not change; the EuroSAT numbers in the README are from the
same run.

### Fixed

- With `KERAS_BACKEND` set to `tensorflow` or `torch` in the shell, a common leftover from
  earlier TensorFlow work, every `kic` command, `kic info` included, failed with a 52-line
  `ModuleNotFoundError` traceback and exit status 1. It now prints one line naming the variable
  and the missing package and exits with status 2, as the README promises for a problem the user
  can fix. `docs/architecture.md` said "set `KERAS_BACKEND` to try one" without saying that the
  backend's package must be installed first; it now says so.
- `kic train` printed the Keras per-epoch progress on standard output, ahead of its JSON, so its
  output could not be parsed, contrary to "every command prints JSON" in the README. The
  progress now goes to standard error. `test_every_step_prints_json` checked `synth`, `prepare`
  and `split` but not `train`, which is how this was missed; a new test covers it.
- `kic evaluate`, run from any directory other than the one training ran in, reported "has no
  manifest.json; run 'kic prepare' first" and sent the user to redo work that was never missing:
  `run.json` keeps the data path as typed, usually relative. It now says that the path does not
  resolve from here and to run it from the directory you trained in.
- The README said the quick start takes "under a minute", and 52 seconds, and
  `docs/testing.md` called its timings upper bounds. Measured again on the same laptop: 58
  seconds, and about 90 on the first run after `uv sync`. Both documents now give those numbers.
- `pyproject.toml` classified the project as "4 - Beta" while the README calls it stable. It is
  now "5 - Production/Stable".

### Added

- Python 3.14. The locked dependencies install on it unchanged and the whole test suite passes
  there; CI now tests Python 3.12, 3.13 and 3.14 on Linux and Windows. The README's route
  without uv, which failed on 3.14 with "requires a different Python", now names the supported
  versions.
- `CONTRIBUTING.md`, a bug-report form that asks for the output of `kic info`, and a
  pull-request checklist, so the rules CI enforces (no lowered thresholds, no ignored warnings,
  no em dash) are written down before a contributor meets them as a failed check.
- A README paragraph on `KERAS_BACKEND` for use from your own Python code: Keras on its own
  defaults to TensorFlow, which this project does not install.

## [1.0.1] - 2026-09-18

Corrections found by a review of the 1.0.0 branch. No behaviour of the pipeline changes; the
EuroSAT numbers in the README are from the same run and are unchanged.

### Fixed

- **A claim in the README was false and has been corrected.** The History section said "No image
  file has ever been committed to this repository, in any branch or tag". That is refuted by one
  command, `git log --all --diff-filter=A --name-only -- '*.png'`, which returns
  `docs/results/eurosat/confusion_matrix.png`, added by the 1.0.0 documentation commit and
  rendered in the README itself. The sentence now says what was actually meant and is true: no
  photograph and no data-set image has ever been committed, and the confusion matrix is the only
  image in the repository.
- A malformed, truncated or hand-edited `manifest.json`, `splits.json` or `run.json` produced a
  raw `JSONDecodeError` or `KeyError` traceback and exit 1, which contradicted the contract
  stated in `cli.py` and in the README: `KicError` and exit 2 for a problem the user can fix,
  anything else is a bug in the tool. These artefacts are now read through
  `dataset.read_json`, and `dataset.split_identity` and `train.load_run` name the file and the
  missing field. 17 tests cover the new paths.
- `SECURITY.md` attributed Pillow's `MAX_IMAGE_PIXELS` ceiling to "Pillow 11" while the lock
  ships Pillow 12.3.0. The number was right in both; the label is gone.
- `docs/architecture.md` quoted wheel sizes for jaxlib 0.11.2 while `uv.lock` pins 0.11.1. It now
  quotes the pinned version: 68.5 MB for Windows and 87.9 MB for Linux x86_64.
- The opt-in network test called `urllib.request.urlopen` instead of `fetch._OPENER.open`, so it
  did not exercise the module's own https-only redirect handler. A regression from the fetch
  hardening work.
- A comment in `data.py` claimed the equivalent Keras layer "cost about 40 % of a training
  step". No such measurement exists in this repository, and `docs/architecture.md` refuses to
  quote numbers for variants that were not re-run. The figure is removed; the design reason
  stays.

### Added

- `README.md` explains how to recover from an interrupted `kic train`: the run directory is
  created before fitting, so delete it or choose a new one.
- `.gitignore` lists `.superpowers/` instead of relying on a tool-generated ignore file inside
  that directory.

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
- 100 tests plus one opt-in network test, including a real training run, mypy strict, ruff,
  bandit, pip-audit, gitleaks, CI on Linux and Windows for Python 3.12 and 3.13.
- README, architecture and testing notes, a EuroSAT benchmark under `docs/results/`,
  SECURITY.md, CITATION.cff, MIT licence.

### Removed

- `.idea/`, `test_environment.py` (replaced by `kic info`), `core_main_file.py` and the four
  scripts it launched.

## [0.1.0-coursework] - 2024-11-24

The coursework as submitted, kept under the tag `v0.1.0-coursework`.

[1.1.0]: https://github.com/tudorandrian/keras-image-classifier/releases/tag/v1.1.0
[1.0.2]: https://github.com/tudorandrian/keras-image-classifier/releases/tag/v1.0.2
[1.0.1]: https://github.com/tudorandrian/keras-image-classifier/releases/tag/v1.0.1
[1.0.0]: https://github.com/tudorandrian/keras-image-classifier/releases/tag/v1.0.0
[0.1.0-coursework]: https://github.com/tudorandrian/keras-image-classifier/tree/v0.1.0-coursework
