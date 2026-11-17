#!/usr/bin/env bash
# Project-local dependencies only. No sudo, OS packages, browser downloads or global config changes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CPU=0
for arg in "$@"; do
  case "$arg" in
    --cpu) CPU=1 ;;
    --help) echo 'Usage: bash scripts/setup.sh [--cpu]  (PYTHON=python3.11 selects the Linux interpreter)'; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
command -v node >/dev/null && command -v npm >/dev/null || { echo 'Install Node.js 22.13+ and npm first.' >&2; exit 1; }
node -e 'const [a,b]=process.versions.node.split(".").map(Number); if(a<22 || (a===22 && b<13)) process.exit(1)' || { echo 'Node.js 22.13+ is required.' >&2; exit 1; }
cd -- "$ROOT"
npm ci --ignore-scripts
if (( CPU )); then
  # Keep Linux/WSL separate from the Windows .venv when sharing a checkout.
  PYTHON="${PYTHON:-python3}"
  "$PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ is required"'
  if [[ -e .venv-linux && ! -x .venv-linux/bin/python ]]; then
    echo '.venv-linux exists but is not a usable Linux venv; choose a clean checkout. Nothing was deleted.' >&2; exit 1
  fi
  if [[ ! -d .venv-linux ]]; then "$PYTHON" -m venv .venv-linux; fi
  .venv-linux/bin/python -m pip install --only-binary=:all: -r requirements/cpu.in
  .venv-linux/bin/python -m pip install --only-binary=:all: -r requirements/torch-cpu.in
  .venv-linux/bin/python -m pip check
  PYTHON="$ROOT/.venv-linux/bin/python" bash scripts/check-environment.sh --with-frameworks --require-frameworks
  mkdir -p build/environment
  .venv-linux/bin/python -m pip freeze > build/environment/linux-freeze.txt
fi
