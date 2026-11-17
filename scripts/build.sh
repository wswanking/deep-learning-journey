#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/build.sh [--format html|pdf|all] [--chapter 00] [--sample] [--browser /path/to/chromium]'
  exit 0
fi
command -v node >/dev/null || { echo 'Node.js 22.13+ is required.' >&2; exit 1; }
node -e 'const [a,b]=process.versions.node.split(".").map(Number); if(a<22 || (a===22 && b<13)) process.exit(1)' || { echo 'Node.js 22.13+ is required.' >&2; exit 1; }
[[ -d "$ROOT/node_modules/markdown-it" ]] || { echo 'Run bash scripts/setup.sh first.' >&2; exit 1; }
exec node "$ROOT/scripts/build.mjs" "$@"
