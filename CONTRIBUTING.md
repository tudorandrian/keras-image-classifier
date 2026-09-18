# Contributing

Contributions are welcome. This is a small project with a precise idea of what "finished" means,
so this page is short and specific: how to set up, what a pull request has to pass, and the
conventions behind those checks.

## Before you start

For anything larger than a typo, open an issue first and describe the change, so that nobody
spends an evening on something that will not be merged. Security problems are the exception:
report them privately, as [SECURITY.md](SECURITY.md) describes, never in a public issue.

## Set up

```bash
git clone https://github.com/tudorandrian/keras-image-classifier.git
cd keras-image-classifier
uv sync
```

[uv](https://docs.astral.sh/uv/getting-started/installation/) installs Python and the locked
dependencies, development tools included. No GPU is needed.

## What a pull request has to pass

CI runs these on Linux and Windows for Python 3.12, 3.13 and 3.14. Run them before you push:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run bandit -q -r src
uv run python scripts/check_text.py
uv run pytest --cov
```

## Conventions

- **Tests assert numbers.** `tests/test_pipeline.py` trains a real model and demands 0.9
  accuracy on three classes. If a change makes it fail, find out why; do not lower a threshold.
  A short training run sits at the majority baseline until batch normalisation has warmed up,
  as the README explains, and that is the usual cause.
- **Warnings are errors.** `pyproject.toml` turns every warning into a test failure, with one
  documented exception for an upstream Keras issue. Fix the cause of a new warning instead of
  adding an ignore.
- **New code comes with tests.** Coverage is 100 % of lines and branches today; the enforced
  floor is 90 %.
- **Two kinds of failure.** A problem the user can fix raises `KicError`, which the command
  line prints as one line with exit status 2. Anything else is a bug in the tool and keeps its
  traceback.
- **Nothing generated goes into git**: no data sets, no images, no model files, no run
  directories. `.gitignore` covers `data/`, `runs/` and `*.keras`.
- **No images of people**, and no data set whose licence or content makes it personal data.
- **English** in code, comments, messages and documentation.
- **No em dash** (U+2014) in any tracked file. It is a house style rule, and
  `scripts/check_text.py` fails the build on one. Use a hyphen.
- **The documentation states only what was measured.** If a change moves a number that the
  README or `docs/testing.md` quotes, measure it again and update the text in the same pull
  request.

## Commits and pull requests

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`.
- One logical change per pull request.
- Pull requests are merged with a merge commit, and the history of `main` is never rewritten.
  Squash and rebase merges are switched off, so your commits land as you wrote them: keep each
  one meaningful.
- Describe user-visible changes in the pull request; they are recorded in
  [CHANGELOG.md](CHANGELOG.md) when a release is cut.

## Licence

By contributing, you agree that your contribution is licensed under the [MIT licence](LICENSE)
of this project.
