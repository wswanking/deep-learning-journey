"""Chapter 02: analytic derivatives, central differences, and CPU autograd.

Run from the repository root:
    .venv/Scripts/python.exe -B examples/02-calculus/run.py
The report is generated under build/ and is not a hand-maintained source file.
"""
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


def peak_working_set_bytes() -> int | None:
    """Return process peak RSS on Windows/Linux; unavailable platforms return None."""
    if sys.platform.startswith("linux"):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if sys.platform != "win32":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def central_gradient(function, x: np.ndarray, h: float) -> np.ndarray:
    """Each coordinate perturbation keeps all other coordinates fixed."""
    result = np.empty_like(x, dtype=np.float64)
    for j in range(x.size):
        direction = np.zeros_like(x)
        direction[j] = h
        result[j] = (function(x + direction) - function(x - direction)) / (2 * h)
    return result


def central_jacobian(function, x: np.ndarray, h: float) -> np.ndarray:
    """Output coordinates are rows; input coordinates are columns."""
    result = np.empty((function(x).size, x.size), dtype=np.float64)
    for j in range(x.size):
        direction = np.zeros_like(x)
        direction[j] = h
        result[:, j] = (function(x + direction) - function(x - direction)) / (2 * h)
    return result


def check_gradient(name, np_function, torch_function, analytic, point, h):
    x_np = np.asarray(point, dtype=np.float64)
    expected = np.asarray(analytic(x_np), dtype=np.float64)
    finite = central_gradient(np_function, x_np, h)
    x_torch = torch.tensor(point, dtype=torch.float64, device="cpu", requires_grad=True)
    value = torch_function(x_torch)
    assert value.ndim == 0, "A scalar output is needed for this gradient comparison."
    automatic, = torch.autograd.grad(value, x_torch)
    actual = automatic.detach().numpy()
    assert expected.shape == actual.shape == finite.shape == x_np.shape
    np.testing.assert_allclose(finite, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    return {
        "name": name, "point": x_np.tolist(), "value": float(value.detach()),
        "gradient_shape": list(actual.shape), "analytic": expected.tolist(),
        "central_difference": finite.tolist(), "autograd": actual.tolist(),
        "central_max_abs_error": float(np.max(np.abs(finite - expected))),
        "autograd_max_abs_error": float(np.max(np.abs(actual - expected))),
    }


def vector_map_np(x):
    return np.array([x[0] ** 2 + x[1], x[0] * x[1], 2 * x[0] - x[1]], dtype=np.float64)


def vector_map_torch(x):
    return torch.stack((x[0] ** 2 + x[1], x[0] * x[1], 2 * x[0] - x[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("build/experiments/02-calculus/results.json"))
    args = parser.parse_args()
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("--epsilon must be finite and positive")
    torch.set_num_threads(1)
    h = args.epsilon
    compute_started = time.perf_counter()

    cases = [
        check_gradient(
            "quadratic", lambda x: x[0] ** 2 + 2 * x[1] ** 2,
            lambda x: x[0] ** 2 + 2 * x[1] ** 2,
            lambda x: [2 * x[0], 4 * x[1]], [1.0, 1.0], h,
        ),
        check_gradient(
            "branching_chain", lambda x: (x[0] * x[1] + x[0]) ** 2,
            lambda x: (x[0] * x[1] + x[0]) ** 2,
            lambda x: [2 * (x[0] * x[1] + x[0]) * (x[1] + 1),
                       2 * (x[0] * x[1] + x[0]) * x[0]],
            [1.0, 2.0], h,
        ),
        check_gradient(
            "non_polynomial", lambda x: np.sin(x[0]) * np.exp(x[1]) + x[0] ** 3 * x[1],
            lambda x: torch.sin(x[0]) * torch.exp(x[1]) + x[0] ** 3 * x[1],
            lambda x: [np.cos(x[0]) * np.exp(x[1]) + 3 * x[0] ** 2 * x[1],
                       np.sin(x[0]) * np.exp(x[1]) + x[0] ** 3],
            [0.7, -0.2], h,
        ),
    ]

    scalar_finite = ((2.0 + h) ** 2 - (2.0 - h) ** 2) / (2 * h)
    scalar_x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    scalar_auto, = torch.autograd.grad(scalar_x ** 2, scalar_x)
    np.testing.assert_allclose([scalar_finite, scalar_auto.item()], [4.0, 4.0], rtol=1e-8, atol=1e-8)

    direction = np.array([3.0 / 5.0, 4.0 / 5.0], dtype=np.float64)
    point = np.array([1.0, 1.0], dtype=np.float64)
    quadratic = lambda x: x[0] ** 2 + 2 * x[1] ** 2
    directional = float((quadratic(point + h * direction) - quadratic(point - h * direction)) / (2 * h))
    expected_directional = float(np.dot([2.0, 4.0], direction))
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
    np.testing.assert_allclose(directional, expected_directional, rtol=1e-8, atol=1e-8)

    point_j = np.array([2.0, 3.0], dtype=np.float64)
    expected_j = np.array([[4.0, 1.0], [3.0, 2.0], [2.0, -1.0]], dtype=np.float64)
    finite_j = central_jacobian(vector_map_np, point_j, h)
    x_j = torch.tensor(point_j, dtype=torch.float64, requires_grad=True)
    automatic_j = torch.autograd.functional.jacobian(vector_map_torch, x_j)
    assert tuple(automatic_j.shape) == (3, 2)
    np.testing.assert_allclose(finite_j, expected_j, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(automatic_j.numpy(), expected_j, rtol=1e-12, atol=1e-12)
    weights = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    final_value = torch.dot(weights, vector_map_torch(x_j))
    input_gradient, = torch.autograd.grad(final_value, x_j)
    expected_input_gradient = expected_j.T @ weights.numpy()
    np.testing.assert_allclose(input_gradient.numpy(), [8.0, 6.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(expected_input_gradient, input_gradient.numpy())

    error_sweep = []
    for step in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12]:
        estimate = (math.sin(0.7 + step) - math.sin(0.7 - step)) / (2 * step)
        error_sweep.append({"h": step, "estimate": estimate, "absolute_error": abs(estimate - math.cos(0.7))})

    boundary_x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    boundary_auto, = torch.autograd.grad(torch.abs(boundary_x), boundary_x)
    left = (abs(0.0) - abs(-h)) / h
    right = (abs(h) - abs(0.0)) / h
    center = (abs(h) - abs(-h)) / (2 * h)
    assert left == -1.0 and right == 1.0 and center == 0.0
    assert boundary_auto.item() == 0.0

    report = {
        "chapter": "02", "passed": True,
        "run_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "environment": {
            "os": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "device": "cpu", "dtype": "float64", "torch_threads": torch.get_num_threads(),
            "randomness": "No random numbers or external datasets are used.",
        },
        "configuration": {"epsilon": h, "finite_rtol": 1e-6, "finite_atol": 1e-7,
                          "autograd_rtol": 1e-12, "autograd_atol": 1e-12},
        "scalar_derivative": {"point": 2.0, "analytic": 4.0,
                              "central_difference": scalar_finite, "autograd": scalar_auto.item()},
        "gradient_cases": cases,
        "directional_derivative": {"direction": direction.tolist(), "analytic": expected_directional,
                                   "central_difference": directional},
        "jacobian": {"point": point_j.tolist(), "value": vector_map_np(point_j).tolist(),
                     "shape": list(automatic_j.shape), "analytic": expected_j.tolist(),
                     "central_difference": finite_j.tolist(), "autograd": automatic_j.tolist(),
                     "scalar_weights": weights.tolist(), "scalar_value": final_value.item(),
                     "chain_input_gradient": input_gradient.tolist()},
        "sin_error_sweep": error_sweep,
        "nondifferentiable_boundary": {
            "function": "abs(s)", "point": 0.0, "left_quotient": left,
            "right_quotient": right, "central_difference": center,
            "autograd_convention": boundary_auto.item(), "classical_derivative_exists": False,
            "interpretation": "A numerical return value at a kink does not establish differentiability.",
        },
        "resources": {"calculation_seconds": time.perf_counter() - compute_started,
                      "including_imports_seconds": time.perf_counter() - STARTED,
                      "peak_working_set_bytes": peak_working_set_bytes(),
                      "memory_scope": "Process peak RSS including imports; Windows PeakWorkingSetSize / Linux ru_maxrss; unavailable platforms return null."},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(f"{case['name']}: analytic={case['analytic']} autograd={case['autograd']} "
              f"central_error={case['central_max_abs_error']:.3e}")
    print(f"jacobian shape={list(automatic_j.shape)}; scalar chain gradient={input_gradient.tolist()}")
    print(f"abs(0): left={left}, right={right}, central={center}, autograd={boundary_auto.item()}; derivative does not exist")
    print(f"PASS; report={args.output.resolve()}")


if __name__ == "__main__":
    main()
