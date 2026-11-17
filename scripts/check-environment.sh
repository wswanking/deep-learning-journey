#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv-linux/bin/python" ]]; then PYTHON="$ROOT/.venv-linux/bin/python"; else PYTHON=python3; fi
fi
exec "$PYTHON" -B "$ROOT/scripts/check-environment.py" "$@"
