"""Run every chapter and figure generator in an isolated source copy; never overwrite source SVGs."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for each command, not a reduced training schedule.')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if not (ROOT / 'node_modules').is_dir():
        parser.error('Run scripts/setup.sh first, and use the CPU virtualenv to invoke this script.')
    # A fresh directory preserves previous evidence. Only Git-tracked source is copied.
    base = ROOT / 'build/full-script-audit'
    base.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='run-', dir=base))
    work = destination / 'source'
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in filter(None, tracked):
        source = ROOT / name
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(ROOT / 'node_modules', work / 'node_modules')
    logs = destination / 'logs'
    logs.mkdir()
    report = {'status': 'running', 'workspace': str(work), 'steps': [],
              'scope': 'default CPU experiment schedules, all standalone generators, capstone rebuild and replay; no GPU or live GitHub web check'}
    report_file = destination / 'results.json'
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def run(label, command, expected=(0,)):
        started = time.monotonic()
        logfile = logs / (label + '.log')
        row = {'name': label, 'command': command, 'expected_exit_codes': list(expected)}
        print('RUN ' + label, flush=True)
        with logfile.open('w', encoding='utf-8') as output:
            try:
                result = subprocess.run(command, cwd=work, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout)
                row.update(returncode=result.returncode, status='passed' if result.returncode in expected else 'failed')
            except subprocess.TimeoutExpired:
                row.update(status='failed', error='timeout')
        row['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report['steps'].append(row)
        report_file.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if row['status'] != 'passed':
            print(logfile.read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)
            raise RuntimeError(label + ' failed; see ' + str(logfile))

    try:
        run('script-inventory', [sys.executable, '-B', 'scripts/audit-scripts.py', '--output', 'build/qa/script-inventory.json'])
        for script in sorted((work / 'scripts').glob('generate-*.mjs')):
            run(script.stem, ['node', str(script.relative_to(work))])
        chapters = sorted((work / 'examples').glob('*/run.py'))
        if len(chapters) != 30:
            raise RuntimeError('Expected all 30 experiment entry points (chapter 00 uses cpu-smoke.py)')
        run('chapter-00', [sys.executable, '-B', 'scripts/cpu-smoke.py', '--with-frameworks'])
        for script in chapters:
            text = script.read_text(encoding='utf-8')
            command = [sys.executable, '-B', str(script.relative_to(work))]
            if '--write-figures' in text:
                command += ['--write-figures']
            if '--figure-dir' in text:
                command += ['--figure-dir', str(Path('docs/assets') / script.parent.name)]
            run('chapter-' + script.parent.name, command)
        run('padding-positive-negative-regression', [sys.executable, '-B', 'tests/padding.test.py'])
        # Standalone helpers get an actual CLI execution; other modules are exercised by run.py.
        for script in sorted((work / 'examples').glob('*/figures.py')):
            module = ast.parse(script.read_text(encoding='utf-8'))
            if any(isinstance(node, ast.If) and '__name__' in ast.unparse(node.test) for node in module.body):
                run('figures-' + script.parent.name, [sys.executable, '-B', str(script.relative_to(work))])
        run('capstone-replay', [sys.executable, '-B', 'examples/30-capstone/run.py', '--mode', 'replay'])
        # The project intentionally retains a quality_not_met example. Its strict gate should fail,
        # not be confused with failed engineering checks.
        result = json.loads((work / 'build/experiments/30-capstone/replay-results.json').read_text())
        quality_ok = result['quality_review']['ready_for_stated_application']
        run('capstone-quality-gate', [sys.executable, '-B', 'examples/30-capstone/run.py', '--mode', 'replay', '--require-quality'], expected=(0,) if quality_ok else (2,))
        run('rebuild-regenerated-book', ['node', 'scripts/build.mjs', '--format', 'all'])
        run('qa-regenerated-book', ['node', 'scripts/check-publication.mjs'])
        report['status'] = 'passed'
    except (RuntimeError, OSError, KeyError) as error:
        report.update(status='failed', error=str(error))
    finally:
        report_file.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        # Copy compact evidence out; keep full isolated workspace locally, without uploading models.
        evidence = destination / 'evidence'
        evidence.mkdir(exist_ok=True)
        for source in (work / 'build').rglob('*.json'):
            target = evidence / source.relative_to(work / 'build')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        print(json.dumps({'status': report['status'], 'report': str(report_file), 'steps': len(report['steps'])}), flush=True)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
