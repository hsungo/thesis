#!/usr/bin/env bash
# Create the Python virtual environment on the Linux GPU server (RTX 4090, sm_89).
#
#   bash scripts/setup_venv.sh                 # .venv, CUDA wheel picked from the driver
#   CUDA=cu126 VENV=~/venvs/eesd bash scripts/setup_venv.sh
#   source .venv/bin/activate
#
# Env vars:  PYTHON (default python3, needs >= 3.10), VENV (default .venv),
#            CUDA   (cu128 | cu126 | cu124 ...; default: newest wheel the driver supports)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
VENV=${VENV:-.venv}

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'

if [[ -z "${CUDA:-}" ]]; then
  drv=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  if [[ -z "$drv" ]]; then echo "nvidia-smi not found: set CUDA=cuXXX explicitly"; exit 1; fi
  if   (( drv >= 128 )); then CUDA=cu128
  elif (( drv >= 126 )); then CUDA=cu126
  elif (( drv >= 124 )); then CUDA=cu124
  else CUDA=cu121; fi
  echo "[setup] driver supports CUDA ${drv:0:2}.${drv:2} -> using PyTorch wheel $CUDA"
fi

"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA}"
pip install -e ".[dev]"

echo "[setup] running unit tests (CPU, ~15 s)"
python -m pytest
echo "[setup] environment check"
python scripts/check_env.py
echo
echo "Done.  Activate with:  source $VENV/bin/activate"
echo "Optional: python scripts/prefetch_data.py   (download all datasets once)"
