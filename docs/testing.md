# Testing

A training pipeline fails quietly. Shuffled labels, a test image that leaked into training, the
wrong epoch saved: each of these still runs to the end and still prints an accuracy. So the
tests here assert numbers, and one of them trains a real model.

## Levels

| Level | Where | Runs in CI | Network |
| --- | --- | --- | --- |
| Static analysis | ruff, ruff format, mypy strict, bandit, pip-audit, `scripts/check_text.py` on ubuntu-24.04; gitleaks in its own job | every push to `main`, every pull request, weekly | advisories only |
| Unit, quantitative | `test_images.py`, `test_dataset.py`, `test_metrics.py`, `test_synth.py`, `test_fetch.py`, `test_data_and_model.py` | ubuntu-24.04 and windows-2025, Python 3.12, 3.13 and 3.14 | no |
| End to end | `test_pipeline.py`, `test_cli.py`: synth, prepare, split, train, evaluate, predict through the command line | same matrix | no |
| Upstream data | `-m network`: the EuroSAT archive on Zenodo still has the pinned size | weekly only | yes |
| Benchmark | the EuroSAT run in the README | by hand, before a release | yes |

There is no browser level because there is no web interface; the rendered README and
`docs/results/eurosat/report.md` are checked on GitHub once, in the release task.

## What is asserted

Data preparation

- The format comes from the file content: a GIF named `.jpg` is refused; empty, truncated and
  non-image files are skipped with a recorded reason; a 300 x 300 image is refused under a
  1000-pixel limit before it is decoded; EXIF rotation is applied.
- Letterboxing a 100 x 50 white image into 64 px gives exactly 16 black rows above and below.
- The pixel hash is equal for the same picture saved as PNG and BMP, and changes with one pixel.
- A duplicate is kept once; a picture under two labels is dropped from both, and the files on
  disk match the manifest exactly.
- The split is stratified (3 of 20 per class in validation and in test), disjoint and complete;
  renaming every source file does not change it; another seed does.
- `splits.json` entries must be manifest paths and appear once; `circle/../../x.png`, a path in
  two splits, or a path twice in one split is refused naming the entry and the split.
- `manifest.json` samples must be exactly `<label>/<sha256[:16]>.png` with an allowed class
  name; anything else is refused as something `kic prepare` would not have written. A manifest
  that lists the same content hash twice is refused the same way.
- `kic synth` refuses a non-empty directory.

Untrusted archives and downloads

- Members named `../../x.jpg`, `/abs/x.jpg`, `C:/x.jpg` or with backslashes write nothing
  outside the destination.
- A 20 MB member of zeros that compresses far smaller is stopped at the byte limit, which counts
  bytes actually written rather than trusting the zip header.
- Plain http is refused, and so is a redirect that leaves https; a wrong checksum or an
  oversized response deletes the partial file.

Metrics

- A 10-sample, 3-class example worked by hand: accuracy 0.7, baseline 0.4, per-class precision,
  recall and F1. A class that is never predicted scores 0 without a warning.

Model and loader

- The parameter count is 98,547 for three classes at 32 px and at 128 px, and there is no
  Flatten layer. Outputs sum to 1. The first layer rescales, so the loader must deliver 0-255.
- Shuffling is seeded and changes each epoch; augmentation produces only the image or its
  mirror, and is wired to the training split only; decoded images are cached.
- The loader compares every decoded image with its manifest hash; a one-pixel edit is refused
  before training or scoring starts, and a deleted file is a user error, not a traceback. It
  opens each file as PNG only, since `kic prepare` never writes anything else, so a file replaced
  by another format under the same name is refused the same way.
- `kic predict` on seven files with a batch size of three calls the model with 3, 3 and 1 images.
- Above `CACHE_BUDGET_BYTES` (2 GiB of decoded uint8 pixels), `train` turns the cache off for
  train plus val and records `"cached_in_memory": false`; `evaluate` applies the same budget to
  the split it scores.

End to end, on 360 synthetic images

- Validation and test accuracy of at least 0.9 against a majority baseline of 1/3. This is the
  test that caught the batch-normalisation problem described in `architecture.md`; the fixture
  runs 20 epochs of 4 steps, which is what puts it past the warm-up.
- `run.json` holds the configuration, class list, versions, best epoch and timings, and agrees
  with `history.csv`.
- The same seed gives the same validation loss to four significant figures.
- A run directory is never overwritten; evaluation refuses data that no longer matches the run;
  an unreadable file among the inputs of `predict` yields an error record and the rest are
  still classified.
- A `.keras` file containing a Lambda layer is refused by `load_model`.
- Every command, `kic train` included, prints one JSON document on standard output and nothing
  else; the per-epoch progress of `kic train` goes to standard error, so `kic train ... | jq`
  works.
- A `KERAS_BACKEND` naming a backend whose package is not installed stops `kic` with one line
  and status 2. This is checked in a fresh interpreter, because Keras reads the variable only
  once, at import.
- `kic evaluate` run from a directory where the run's relative data path does not resolve says
  exactly that, instead of asking for a `kic prepare` that is not needed.
- User errors exit with status 2 and one line on standard error, never a traceback. That
  includes a damaged artefact: a truncated or hand-edited `manifest.json`, `splits.json` or
  `run.json` is a problem the user can fix, so it is a `KicError` naming the file and the
  missing or unparsable part, not a `JSONDecodeError` or `KeyError` traceback.

What `evaluate` checks before scoring, in order: the prepared set's class list and image size
against `run.json`; the split's version, seed and ratios; then `split_digest`, the SHA-256 over
each split's paths and content hashes, in order, against the value `train` recorded, so a
relabelled sample is caught as well as a changed one; then every file of the split against the
hash `kic prepare` wrote into `manifest.json`. A test list
refilled from training images, a prepared set rebuilt from other raw data under the same seed,
and a PNG edited after preparation are each refused with one line. What is still not covered:
near-duplicates across splits (see Known limits), and a `manifest.json` rewritten together with
the files it describes, which is a new data set rather than a damaged one.

## Reference measurements

Measured on 2026-09-23: Windows 10, Intel Core i7-7700HQ (2017, 4 cores and 8 threads), 16 GB of
memory, no GPU, Python 3.13.15, Keras 3.15.1 on JAX 0.11.1. The machine was doing other light
work during part of the EuroSAT prepare step, so treat that time as an upper bound.

| Measure | Value |
| --- | --- |
| Environment from `uv sync` | 64 packages, 609 MB including the development tools |
| Test suite, `uv run pytest --cov` | 154 tests and 1 deselected network test, 180 s measured on a busy machine (other processes at 63 % CPU load beforehand; treat as an upper bound), 100 % line and branch coverage |
| Quick start on synthetic shapes: 600 images, 48 px, 15 epochs | about 50 s for all six commands (48 s measured), about 90 s on the first run after a fresh `uv sync`; test accuracy 1.000, baseline 0.333 |
| Batch-norm warm-up on the same data (7 steps per epoch) | validation accuracy exactly 0.3333 through step 28, 0.3444 at step 35, 0.9333 at step 42, 1.0000 at step 49, while training accuracy is 1.0000 throughout |
| EuroSAT prepare: decode, letterbox, hash and write 27,000 images | 2 min 40 s; 0 skipped, 0 duplicates, 0 conflicts |
| EuroSAT training: 20 epochs, 18,900 images, 99,450 parameters | 1,261.6 s (21 min), 296 steps of 64 images per epoch, 213 ms per step |
| EuroSAT test split, 4,050 images | accuracy 0.9491, macro F1 0.9473, baseline 0.1111 |
| EuroSAT weakest and strongest class by F1 | River 0.907, SeaLake 0.992 |

The `run.json` committed under `docs/results/eurosat/` was produced at the final 1.1.0 commit
and records 1,818.0 s for that training run, not the 1,261.6 s quoted above: other processes
loaded the machine during it. Its accuracy, per-class figures and confusion matrix are identical
to the idle run this table quotes, so the table keeps the idle number as the representative one.

Repeat the EuroSAT rows after any change to `model.py`, `train.py` or `data.py`, and update the
table if a value moves by more than a quarter. The 1.1.0 re-measurement reproduced every
per-class figure, the confusion matrix and the best validation loss exactly from the 1.0.0 run,
which confirms across two releases that JAX on CPU is deterministic on this one machine (see
Known limits).

## Known limits

- Near-duplicates (the same scene re-encoded at another quality or crop) have different pixel
  hashes and can still straddle the split. `kic prepare` found no exact duplicate in EuroSAT;
  near-duplicates were not looked for. A scraped data set is far more exposed.
- The EuroSAT split is this project's own seeded 70/15/15 split. Published results use other
  splits, so compare with care.
- One seed, one run: no confidence interval is reported. `--seed` makes repeating it cheap.
- JAX on CPU is deterministic on one machine. Across machines and versions, expect the third
  decimal to move.
- Only the JAX backend is tested. The code uses nothing but the Keras API, so the PyTorch and
  TensorFlow backends should work, but nothing here proves it, and no timing comparison between
  backends has been made for this release.
- The alternative model shapes that were tried while `model.py` was being written (fewer blocks,
  a constant learning rate) were not re-run for 1.0.0, so this document quotes no numbers for
  them. Only the shipped configuration is measured.
- The decoded-image cache holds a whole split in memory: 27,000 images of 64 px are 332 MB as
  uint8, the same count at 224 px would be 4.1 GB. Above `CACHE_BUDGET_BYTES` (2 GiB) the cache
  is turned off, for train plus val in `kic train` and for the scored split in `kic evaluate`,
  and each decodes its images again every epoch or every call instead.
- `images.load_rgb` takes a `max_pixels` argument, but Pillow's own decompression-bomb ceiling
  (`Image.MAX_IMAGE_PIXELS`, 89,478,485 here) is applied first, while the header is read. So
  `max_pixels` can tighten the limit below the project default of 50,000,000 and cannot raise it
  past Pillow's ceiling.
