"""Inventory and syntax-check repository-owned scripts, without executing training."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = {'.git', 'node_modules', 'build', '.venv', '.venv-linux', '__pycache__'}
EXTENSIONS = {'.py', '.mjs', '.js', '.sh', '.ps1', '.cmd', '.bat'}


def inventory(root):
    files = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        files.extend(Path(directory) / name for name in sorted(names) if Path(name).suffix in EXTENSIONS)
    return sorted(files)


def classify(path):
    name = path.name
    if path.suffix in {'.ps1', '.cmd', '.bat'}:
        return 'windows-native'
    if name == 'run.py':
        return 'chapter-entry'
    if name.startswith('generate-'):
        return 'figure-generator'
    if 'tests' in path.parts:
        return 'test'
    if 'examples' in path.parts and name != 'render-report.mjs':
        return 'chapter-helper'
    return 'tool-or-shared-module'


def audit(root):
    rows = []
    for path in inventory(root):
        text = path.read_text(encoding='utf-8-sig')
        row = {'path': path.relative_to(root).as_posix(), 'kind': classify(path), 'syntax': 'passed'}
        try:
            if path.suffix == '.py':
                ast.parse(text, filename=str(path))
            elif path.suffix in {'.mjs', '.js', '.sh'}:
                command = ['bash', '-n', str(path)] if path.suffix == '.sh' else ['node', '--check', str(path)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError(result.stderr)
            else:
                row['syntax'] = 'not_run_native_windows'
            if path.suffix in {'.mjs', '.js'} and 'chromium.launch(' in text and path.name != 'browser.mjs' and 'tests' not in path.relative_to(root).parts:
                raise RuntimeError('Browser entry bypasses the shared cross-platform browser launcher')
            if path.suffix == '.sh' and '\r' in path.read_bytes().decode('utf-8'):
                raise RuntimeError('Bash entry must use LF line endings')
        except (SyntaxError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            row['syntax'] = 'failed'
            row['error'] = str(error)
        rows.append(row)
    return {'status': 'failed' if any(r['syntax'] == 'failed' for r in rows) else 'passed',
            'scope': 'repository scripts; excludes .git, dependencies, virtualenvs and generated build output',
            'total': len(rows), 'files': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(ROOT)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload, end='')
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
