#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install maturin

maturin build --release --out dist
python -m pip install dist/*.whl

pytest -q tests/test_ipm.py tests/test_property_randomized.py
