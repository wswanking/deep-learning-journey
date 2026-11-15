"""Small deterministic CPU checks; no third-party packages are required."""
import argparse
import importlib.util
import json
import math
import platform
import sys
import time


def baseline():
    started = time.perf_counter()
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    product = [
        [sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)]
        for i in range(2)
    ]
    if product != [[19.0, 22.0], [43.0, 50.0]]:
        raise RuntimeError("Matrix multiplication mismatch")
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ys = [2.0 * x + 1.0 for x in xs]

    def loss(w, bias):
        return sum((w * x + bias - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

    def gradient(w, bias):
        errors = [w * x + bias - y for x, y in zip(xs, ys)]
        return (
            2.0 * sum(error * x for error, x in zip(errors, xs)) / len(xs),
            2.0 * sum(errors) / len(xs),
        )

    w, bias = -0.25, 0.5
    eps = 1e-6
    analytical = gradient(w, bias)
    numerical = (
        (loss(w + eps, bias) - loss(w - eps, bias)) / (2.0 * eps),
        (loss(w, bias + eps) - loss(w, bias - eps)) / (2.0 * eps),
    )
    max_gradient_error = max(abs(x - y) for x, y in zip(analytical, numerical))
    if max_gradient_error > 1e-6:
        raise RuntimeError("Finite-difference gradient check failed")

    w, bias = 0.0, 0.0
    initial_loss = loss(w, bias)
    for _ in range(200):
        dw, db = gradient(w, bias)
        w -= 0.1 * dw
        bias -= 0.1 * db
    final_loss = loss(w, bias)
    if not (math.isfinite(final_loss) and final_loss < 1e-12):
        raise RuntimeError("CPU gradient descent did not converge")
    return {
        "status": "passed",
        "backend": "Python standard library, CPU",
        "matrix_product": product,
        "max_gradient_error": max_gradient_error,
        "training": {
            "samples": 5,
            "steps": 200,
            "learning_rate": 0.1,
            "initial_mse": initial_loss,
            "final_mse": final_loss,
            "weight": w,
            "bias": bias,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def optional_numpy():
    if importlib.util.find_spec("numpy") is None:
        return {"status": "not_installed"}
    import numpy as np
    result = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
    if not np.array_equal(result, [[19, 22], [43, 50]]):
        raise RuntimeError("NumPy CPU multiplication failed")
    return {"status": "passed", "version": np.__version__, "matrix_product": result.tolist()}


def optional_torch():
    if importlib.util.find_spec("torch") is None:
        return {"status": "not_installed", "cuda_available": None}
    started = time.perf_counter()
    import torch
    torch.set_num_threads(2)
    product = torch.tensor([[1, 2], [3, 4]]) @ torch.tensor([[5, 6], [7, 8]])
    if product.tolist() != [[19, 22], [43, 50]]:
        raise RuntimeError("PyTorch CPU matrix multiplication failed")
    training_started = time.perf_counter()
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64, device="cpu")
    y = 2.0 * x + 1.0
    w = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    for _ in range(200):
        mse = ((w * x + bias - y) ** 2).mean()
        mse.backward()
        with torch.no_grad():
            w -= 0.1 * w.grad
            bias -= 0.1 * bias.grad
        w.grad = None
        bias.grad = None
    final_mse = ((w * x + bias - y) ** 2).mean().item()
    if not math.isfinite(final_mse) or final_mse >= 1e-12:
        raise RuntimeError("PyTorch CPU autograd training failed")
    return {
        "status": "passed",
        "version": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "threads": torch.get_num_threads(),
        "matrix_product": product.tolist(),
        "samples": 5,
        "steps": 200,
        "learning_rate": 0.1,
        "initial_mse": 9.0,
        "training_elapsed_seconds": time.perf_counter() - training_started,
        "elapsed_seconds_including_import": time.perf_counter() - started,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "final_mse": final_mse,
        "weight": w.item(),
        "bias": bias.item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-frameworks", action="store_true",
                        help="Also import and validate NumPy/PyTorch if installed.")
    args = parser.parse_args()
    report = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "baseline": baseline(),
    }
    if args.with_frameworks:
        for name, probe in (("numpy", optional_numpy), ("torch", optional_torch)):
            try:
                report[name] = probe()
            except Exception as exc:
                report[name] = {"status": "failed", "error": str(exc)}
    failed = any(isinstance(value, dict) and value.get("status") == "failed"
                 for value in report.values())
    report["status"] = "failed" if failed else "passed"
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
