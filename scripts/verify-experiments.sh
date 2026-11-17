#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv-linux/bin/python}"
[[ -x "$PYTHON" ]] || command -v "$PYTHON" >/dev/null || { echo 'Run bash scripts/setup.sh --cpu first.' >&2; exit 1; }
exec "$PYTHON" -B "$ROOT/scripts/verify-experiments.py" "$@"
