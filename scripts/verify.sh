#!/usr/bin/env bash
# Runs the existing source tests and Linux publication acceptance without installing anything.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -gt 0 ]]; then
  if [[ "$1" == --help && "$#" == 1 ]]; then echo 'Usage: bash scripts/verify.sh (optional BROWSER_EXECUTABLE and PYTHON environment variables)'; exit 0; fi
  echo 'Unknown arguments; see --help.' >&2; exit 2
fi
cd -- "$ROOT"
npm test
npm run check:markdown
bash scripts/check-environment.sh
"${PYTHON:-python3}" -B scripts/cpu-smoke.py
bash scripts/build.sh --format all --sample
node scripts/check-publication.mjs --name typesetting-sample
bash scripts/build.sh --format all
node scripts/check-publication.mjs --name deep-learning-journey
