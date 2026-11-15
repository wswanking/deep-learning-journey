"""Chapter 04: finite-precision arithmetic, stable probabilities, and gradient descent."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time
STARTED = time.perf_counter()
import numpy as np
import torch
from figures import write_figures

ROOT = Path(__file__).resolve().parents[2]


def peak_working_set_bytes():
    if sys.platform != "win32":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in ["PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
    kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    c = Counters(); c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(c.PeakWorkingSetSize)


def serializable(value):
    """Keep expected NaN/Inf evidence as explicit strings, not invalid JSON numbers."""
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def stable_probabilities(z):
    """Rows are examples; the last axis is the class axis."""
    if z.ndim != 2 or z.shape[1] == 0 or not np.all(np.isfinite(z)):
        raise ValueError("Expected a non-empty 2D array of finite logits.")
    maximum = z.max(axis=1, keepdims=True)
    shifted = z - maximum
    exponentials = np.exp(shifted)
    denominator = exponentials.sum(axis=1, keepdims=True)
    log_denominator = np.log(denominator)
    return exponentials / denominator, maximum + log_denominator, shifted - log_denominator


def descent(rate, steps):
    point = np.array([2.0, 1.0], dtype=np.float64)
    points, values, max_gradient_error = [], [], 0.0
    for k in range(steps + 1):
        points.append(point.tolist())
        values.append(float(0.5 * (point[0] ** 2 + 4 * point[1] ** 2)))
        if k == steps:
            break
        analytic = np.array([point[0], 4 * point[1]], dtype=np.float64)
        # Each point is an independent differentiation example; no graph history is retained.
        x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        value = 0.5 * (x[0] ** 2 + 4 * x[1] ** 2)
        automatic, = torch.autograd.grad(value, x)
        error = float(np.max(np.abs(automatic.numpy() - analytic)))
        max_gradient_error = max(max_gradient_error, error)
        np.testing.assert_allclose(automatic.numpy(), analytic, rtol=1e-13, atol=1e-13)
        point = point - rate * analytic
    # The closed form independently checks every recorded iterate.
    for k, p in enumerate(points):
        expected = [2 * (1 - rate) ** k, (1 - 4 * rate) ** k]
        np.testing.assert_allclose(p, expected, rtol=1e-12, atol=1e-12)
    return {"learning_rate": rate, "points": points, "values": values,
            "final_value": values[-1], "max_autograd_error": max_gradient_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/experiments/04-numerical-optimization/results.json")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    compute_started = time.perf_counter()
    precision = []
    for dtype in [np.float32, np.float64]:
        info = np.finfo(dtype)
        one = dtype(1)
        neighboring = np.nextafter(one, dtype(np.inf))
        a, b, c = dtype(1e8), dtype(-1e8), dtype(1)
        small = dtype(1e-8)
        naive_cancellation = np.sqrt(one + small) - one
        rationalized = small / (np.sqrt(one + small) + one)
        precision.append({"dtype": str(np.dtype(dtype)), "bits": info.bits,
            "eps": float(info.eps), "smallest_normal": float(info.smallest_normal),
            "smallest_subnormal": float(info.smallest_subnormal), "max": float(info.max),
            "neighbor_above_one": float(neighboring), "neighbor_gap": float(neighboring - one),
            "left_association": float((a + b) + c), "right_association": float(a + (b + c)),
            "sqrt_naive": float(naive_cancellation), "sqrt_rationalized": float(rationalized)})
    assert precision[0]["left_association"] == 1 and precision[0]["right_association"] == 0
    assert precision[1]["left_association"] == precision[1]["right_association"] == 1
    assert precision[0]["sqrt_naive"] == 0 and precision[0]["sqrt_rationalized"] > 0

    logits = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]], dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        raw_exp = np.exp(logits)
        naive = raw_exp / raw_exp.sum(axis=1, keepdims=True)
    probabilities, lse, log_probabilities = stable_probabilities(logits)
    shifted_probabilities, _, _ = stable_probabilities(logits + 123.0)
    tensor = torch.tensor(logits, dtype=torch.float64)
    torch_probabilities = torch.softmax(tensor, dim=1)
    torch_lse = torch.logsumexp(tensor, dim=1, keepdim=True)
    torch_log = torch.log_softmax(tensor, dim=1)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(probabilities, shifted_probabilities, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(probabilities, torch_probabilities.numpy(), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(lse, torch_lse.numpy(), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(log_probabilities, torch_log.numpy(), rtol=1e-13, atol=1e-13)
    assert not np.any(np.isfinite(naive)) and np.all(np.isfinite(probabilities))
    softmax_cases = [{"logits": logits[i].tolist(), "raw_exponentials": raw_exp[i].tolist(),
        "naive_probability": naive[i].tolist(), "stable_probability": probabilities[i].tolist(),
        "stable_log_probability": log_probabilities[i].tolist(), "logsumexp": float(lse[i, 0])}
        for i in range(2)]

    extreme = np.array([[0.0, -1000.0]], dtype=np.float64)
    extreme_p, extreme_lse, extreme_log = stable_probabilities(extreme)
    with np.errstate(divide="ignore"):
        log_after_probability = np.log(extreme_p)
    assert np.isneginf(log_after_probability[0, 1]) and extreme_log[0, 1] == -1000.0
    boundary_example = {"logits": extreme.tolist(), "probability": extreme_p.tolist(),
        "log_after_softmax": log_after_probability.tolist(), "direct_log_softmax": extreme_log.tolist(),
        "target_second_class_loss": float(-extreme_log[0, 1])}

    steps = 40
    cases = [descent(rate, steps) for rate in [0.03, 0.2, 0.45, 0.55]]
    assert cases[0]["final_value"] > 0.1
    assert cases[1]["final_value"] < 1e-6 and cases[2]["final_value"] < 1e-6
    assert cases[3]["values"][1] < cases[3]["values"][0] and cases[3]["final_value"] > 1e6
    report = {"chapter": "04", "passed": True,
        "run_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "os": platform.platform(), "device": "cpu", "torch_threads": torch.get_num_threads(),
            "randomness": "Fixed analytic functions; no random sampling or external data."},
        "configuration": {"steps": steps, "initial_point": [2, 1], "objective": "0.5*(theta1**2+4*theta2**2)",
            "learning_rates": [0.03, 0.2, 0.45, 0.55], "optimization_dtype": "float64",
            "class_axis": 1, "tolerances": {"normalization_and_shift_rtol": 1e-14, "normalization_and_shift_atol": 1e-14, "framework_rtol": 1e-13, "framework_atol": 1e-13, "iterate_rtol": 1e-12, "iterate_atol": 1e-12}, "data_split": "Not applicable: analytic optimization, no fitted data model."},
        "floating_point": precision, "softmax_cases": softmax_cases, "log_probability_boundary": boundary_example,
        "gradient_descent": cases,
        "resources": {"calculation_seconds": time.perf_counter() - compute_started,
            "including_imports_seconds": time.perf_counter() - STARTED,
            "peak_working_set_bytes": peak_working_set_bytes(),
            "memory_scope": "Windows process peak resident memory, including library imports; SVG generation excluded from timing."}}
    if not args.skip_figures:
        write_figures(report, ROOT / "docs/assets/04-numerical-optimization")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for case in softmax_cases:
        print("softmax", case["logits"], "naive_finite=False stable=", case["stable_probability"])
    for case in cases:
        print(f"eta={case['learning_rate']}: J0={case['values'][0]:.6g} J1={case['values'][1]:.6g} J40={case['final_value']:.9g}")
    print("PASS; report=" + str(args.output.resolve()))


if __name__ == "__main__":
    main()
