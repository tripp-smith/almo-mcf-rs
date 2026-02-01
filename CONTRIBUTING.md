# Contributing to almo-mcf

Thanks for helping improve the almost-linear minimum-cost flow solver! This guide covers how to set up a
local development environment, run tests, and prepare changes for review.

## Development setup

### Python + Rust toolchain
- Install Python 3.10+.
- Install Rust (stable) with `rustfmt` and `clippy` components.
- Install `uv` for fast dependency management.

### Install the project
```bash
uv pip install -e . --system
```

### Build the wheel locally (PyPI packaging smoke test)
```bash
maturin build --release --out dist
pip install dist/*.whl
pytest -q tests/test_ipm.py tests/test_property_randomized.py
```

## Running tests

Run the full test suite:
```bash
pytest
```

Run a faster subset over the tests directory:
```bash
pytest -q tests/
```

Check Rust formatting (from the `rust/` directory):
```bash
cargo fmt --check
```

Run a scalability smoke test on a large graph (m >= 1e5 edges):
```bash
RUN_SCALABILITY=1 pytest -q tests/test_scalability.py
```

## Code style
- Keep changes focused and well-tested.
- Avoid large refactors without clear motivation.
- Add or update tests when changing behavior.

## Submitting a PR
- Ensure CI passes.
- Update `CHANGELOG.md` when adding user-facing changes.
- Include reproduction steps for bug fixes where possible.
