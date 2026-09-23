# Architecture

## Data flow

```
raw/<class>/*.jpg            any folder-per-class directory, or kic synth / kic fetch-eurosat
   |  kic prepare            decode safely, letterbox, hash pixels, drop duplicates and conflicts
prepared/<class>/<hash>.png  + manifest.json (what was kept, what was skipped and why)
   |  kic split              stratified, seeded, by content hash
prepared/splits.json         lists of paths; no file is copied, so no file can be in two splits
   |  kic train              small CNN, best epoch kept, early stopping after a warm-up
runs/<name>/                 model.keras, history.csv, run.json (config, versions, timings)
   |  kic evaluate           held-out split
runs/<name>/                 metrics.json, confusion_matrix.png, report.md
   |  kic predict            new files through the same decoder as the training data
JSON on standard output
```

## Modules

| Module | Responsibility | Imports Keras |
| --- | --- | --- |
| `images.py` | decode one untrusted file: format allow-list, pixel limit, EXIF rotation, letterbox, pixel hash | no |
| `dataset.py` | scan classes, prepare, manifest, split | no |
| `synth.py` | synthetic shapes data set for tests and the offline quick start | no |
| `fetch.py` | EuroSAT download with checksum, archive extraction that cannot leave its directory | no |
| `metrics.py` | confusion matrix, precision, recall, F1, majority baseline, in NumPy | no |
| `data.py` | `ImageBatches`, a `keras.utils.PyDataset` with cache, seeded shuffle and flips | yes |
| `model.py` | `build_cnn` | yes |
| `train.py` | `TrainConfig`, `train`, `load_run`, `load_model` | inside functions |
| `evaluate.py` | metrics, confusion-matrix plot, Markdown report | through `train.py` |
| `predict.py` | classify files | through `train.py` |
| `cli.py` | `kic` subcommands, JSON output, exit codes | no |

Everything up to `metrics.py` runs and is tested without a deep-learning library, and `kic --help`,
`kic prepare` and `kic split` start quickly because Keras is imported only by the commands that
need it.

## Decisions

**uv instead of Anaconda.** The coursework carried a `requirements.txt` exported from one Windows
machine in UTF-16 with a byte-order mark, which pip cannot read, pinning 57 packages, among them
`tensorflow_intel==2.18.0` and `dlib==19.24.6`, which need a compiler on most other machines.
`pyproject.toml` now states five direct dependencies with version ranges, `uv.lock` pins the full
tree of 64 packages with hashes for Windows, Linux and macOS, and `uv sync` installs Python
itself. A plain `pip install .` in a virtual environment works as well, without the lock.

**Keras 3 on JAX instead of TensorFlow.** Keras 3 runs on JAX, PyTorch or TensorFlow. The code
uses only the Keras API, and JAX is the default because its wheels are far smaller: on PyPI the
`jaxlib` 0.11.1 wheel that `uv.lock` pins is 68.5 MB for Windows and 87.9 MB for Linux x86_64
(cp313), against 351.2 MB and 572.9 MB for the current TensorFlow 2.21.0 wheels. The whole
environment here, development tools included, is 609 MB. The one TensorFlow-only convenience the coursework used,
`image_dataset_from_directory`, is replaced by `data.py`, a 75-line `PyDataset`. Other backends
are not tested here. To try one, install its package into the environment first and then
set `KERAS_BACKEND`; the variable alone, naming a package that is not installed, makes `kic`
stop with a one-line explanation instead of a traceback.

**Satellite images instead of faces.** The coursework drew on CelebA, LFW and FER-2013 and on
photographs of people collected from the web. Face data sets carry research-only licences and
are personal data. EuroSAT is MIT-licensed, has a DOI, contains no people, and at 64 px trains
on a laptop CPU. The pipeline itself accepts any folder-per-class directory.

**A split is a list, not a directory tree.** Copying files into `train/`, `validation/` and
`test/` invites the two classic errors: a file in two places, and a split that changes when the
file system lists names in another order. Here a picture is identified by the SHA-256 of its
decoded pixels, duplicates are removed before splitting, a picture found under two labels is
dropped, and the split depends on the hashes and the seed alone. Because the list is a plain
file, `train` records a digest of its members' content hashes and `evaluate` compares it, and
both check every prepared file against the manifest before they start.

**Global average pooling.** The coursework network flattened a 26 x 26 x 128 feature map into a
128-unit Dense layer: 11,075,712 of its 11,169,347 parameters sat in that one layer. The
replacement has four convolution blocks and global average pooling, so the head is a few hundred
weights whatever the input size: 98,547 parameters for three classes, 99,450 for ten.

**Batch normalisation needs a warm-up before early stopping.** Batch normalisation uses batch
statistics while training and moving averages at inference. The moving variance starts at 1.0
while the real value after the first convolution is far smaller, so until the average has caught
up the model answers at chance on anything evaluated in inference mode, including its own
training images. Measured here on the synthetic shapes set, 420 training images in batches of
64, so 7 optimiser steps per epoch:

| Epoch | Steps so far | Training accuracy | Validation accuracy |
| ---: | ---: | ---: | ---: |
| 4 | 28 | 1.0000 | 0.3333 |
| 5 | 35 | 1.0000 | 0.3444 |
| 6 | 42 | 1.0000 | 0.9333 |
| 7 | 49 | 1.0000 | 1.0000 |

Validation is pinned to exactly 1/3, the majority baseline for three classes, while the model
has already fit the training data perfectly. It recovers at about 45 optimiser steps. Two things
follow. `model.py` sets the batch-norm momentum to 0.9 instead of the Keras default of 0.99, so
the averages converge in tens of steps rather than hundreds; and `train.py` converts
`BATCH_NORM_WARMUP_STEPS = 50` into `warmup_epochs = ceil(50 / steps_per_epoch)` and passes it to
`EarlyStopping(start_from_epoch=...)`, so early stopping cannot end a run during the blind
period. Without both, the end-to-end test that demands 0.9 accuracy fails.

**Four convolution blocks and a cosine-decay learning rate.** `train.py` schedules the learning
rate with `CosineDecay` to zero over `epochs * steps_per_epoch`: the late, small steps are what
steady a validation loss that otherwise swings from epoch to epoch. Three-block and constant-rate
variants were tried while the model was being developed; those runs were not repeated for this
release, so no numbers for them are quoted here. What is measured is the shipped configuration,
in `docs/results/eurosat/`.

**Flips in the loader, not in the model.** Mirroring the NumPy batch in `data.py` is seeded by
(seed, epoch, batch), so the same run sees the same flips again, and it keeps the saved model
free of training-only layers, which is one less way to serve a model that behaves differently
from the one that was validated. A test pins augmentation to the training split only.

**Interactive prompts removed.** Each of the five coursework scripts asked its questions through
`input()`, which cannot be tested, scripted or run in CI. Each step is now a subcommand with
flags that prints JSON and returns an exit status.

**MIT licence.** The five direct dependencies are permissive: jax and keras are Apache-2.0,
numpy is BSD-3-Clause, pillow is MIT-CMU and matplotlib uses its own PSF-style licence. EuroSAT is
MIT. Nothing there constrains the choice, and MIT is what the author uses for code elsewhere. A later stage that adds pretrained weights would need to check the licence of
those weights, not of this code.
