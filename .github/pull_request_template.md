## What and why

<!-- One logical change. Link the issue if there is one. -->

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy` passes
- [ ] `uv run bandit -q -r src` and `uv run python scripts/check_text.py` pass
- [ ] `uv run pytest --cov` passes, with no threshold lowered and no warning ignored
- [ ] Any number quoted in the README or `docs/` that this change affects was measured again
