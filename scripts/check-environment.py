"""Read-only Linux/WSL resource inventory; CPU smoke tests are opt-in. No network probes."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def probe(command, timeout=10):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "stdout": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "error": type(error).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-frameworks", action="store_true")
    parser.add_argument("--require-frameworks", action="store_true",
                        help="Fail unless NumPy and CPU PyTorch smoke tests pass (implies --with-frameworks).")
    parser.add_argument("--output", type=Path, help="Create a new JSON report; refuses to overwrite.")
    args = parser.parse_args()
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key + "_bytes"] = int(value.split()[0]) * 1024
    packages = {}
    for name in ["pip", "numpy", "matplotlib", "torch"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(ROOT)
    report = {
        "schema_version": 1, "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.system(), "release": platform.release(), "architecture": platform.machine(),
        "is_wsl": "microsoft" in platform.release().lower(), "logical_cpus": os.cpu_count(),
        "memory": memory, "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "resource_note": "Point-in-time snapshot; /proc memory may describe the host, not container limits. Recheck before training.",
        "python": {"executable": sys.executable, "version": platform.python_version(),
                   "is_virtualenv": sys.prefix != sys.base_prefix, "packages": packages},
        "tools": {name: shutil.which(name) for name in ["node", "npm", "git", "chromium", "chromium-browser", "google-chrome", "nvidia-smi", "nvcc"]},
        "chinese_fonts": probe(["fc-list", ":lang=zh", "family"]),
        "nvidia": probe(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"]),
        "cpu_smoke": {"status": "not_run"}, "network_probed": False,
    }
    failed = False
    if args.with_frameworks or args.require_frameworks:
        result = probe([sys.executable, "-B", str(ROOT / "scripts/cpu-smoke.py"), "--with-frameworks"], timeout=120)
        try:
            report["cpu_smoke"] = json.loads(result.get("stdout", ""))
        except json.JSONDecodeError:
            report["cpu_smoke"] = result
        failed = result["status"] != "passed"
        if args.require_frameworks:
            failed |= any(report["cpu_smoke"].get(name, {}).get("status") != "passed"
                          for name in ["baseline", "numpy", "torch"])
    report["status"] = "failed" if failed else "passed"
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        # Exclusive create protects previous evidence and user files.
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload)
    print(payload, end="")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
