"""CPU float64 experiments for Chapter 01; no downloads or training required."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import sys
import time
import tracemalloc

STARTED = time.perf_counter()
tracemalloc.start()
try:
    import numpy as np
    import torch
except ImportError as error:
    raise SystemExit(
        f"Missing chapter dependency: {error.name}. "
        "Use the project .venv and the installation instructions in requirements/README.md."
    ) from error
IMPORT_SECONDS = time.perf_counter() - STARTED


def plain(value):
    """Convert numeric library values to JSON values without losing shape metadata."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def close_numpy(actual, expected):
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(f"NumPy shape {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def close_torch(actual, expected):
    expected = torch.as_tensor(expected, dtype=torch.float64, device="cpu")
    if actual.shape != expected.shape:
        raise AssertionError(f"PyTorch shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def mm_reference(left, right):
    """Literal row-times-column sums independent of NumPy and PyTorch."""
    return [
        [sum(left[i][k] * right[k][j] for k in range(len(right)))
         for j in range(len(right[0]))]
        for i in range(len(left))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("build/experiments/01-linear-algebra/results.json"),
        help="JSON report path, relative to the current working directory",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    checks = []

    def record(name, **values):
        checks.append({"name": name, "passed": True, **plain(values)})
        print(f"PASS {name}")

    kwargs = {"dtype": torch.float64, "device": "cpu"}
    n = lambda value: np.asarray(value, dtype=np.float64)
    t = lambda value: torch.tensor(value, **kwargs)

    # 1. Shape, number of axes and number of entries are separate quantities.
    shapes = [(), (3,), (2, 3), (2, 3, 4)]
    for shape in shapes:
        an, at = np.zeros(shape, dtype=np.float64), torch.zeros(shape, **kwargs)
        assert an.shape == tuple(at.shape) == shape
        assert an.ndim == at.ndim == len(shape)
        assert an.size == at.numel() == math.prod(shape)
    record("objects_and_axes", shapes=shapes, counts=[math.prod(s) for s in shapes])

    # 2. A one-dimensional array has no explicit row/column axis.
    xn, xt = n([2, 1, 0]), t([2, 1, 0])
    assert xn.T.shape == (3,)
    assert xn[:, None].shape == tuple(xt[:, None].shape) == (3, 1)
    assert xn[None, :].shape == tuple(xt[None, :].shape) == (1, 3)
    record("vector_orientation", vector_shape=xn.shape,
           column_shape=xn[:, None].shape, row_shape=xn[None, :].shape)

    # 3. Multiplication is not an inner product unless it includes a sum.
    un, vn, ut, vt = n([1, 2, 3]), n([4, -1, 2]), t([1, 2, 3]), t([4, -1, 2])
    close_numpy(un * vn, [4, -2, 6])
    close_torch(ut * vt, [4, -2, 6])
    close_numpy(un @ vn, 8)
    close_torch(ut @ vt, 8)
    assert (un @ vn).shape == tuple((ut @ vt).shape) == ()
    close_numpy((un[None, :] @ vn[:, None]), [[8]])
    close_torch((ut[None, :] @ vt[:, None]), [[8]])
    record("elementwise_and_inner_product", elementwise=un * vn, inner=un @ vn,
           inner_shape=(un @ vn).shape, explicit_matrix_shape=(1, 1))

    # 4. A small matrix product, also compared to literal standard-library sums.
    a, b = [[1, 2, 3], [4, 5, 6]], [[1, 2], [0, 1], [-1, 0]]
    ab = mm_reference(a, b)
    assert ab == [[-2, 4], [-2, 13]]
    close_numpy(n(a) @ n(b), ab)
    close_torch(t(a) @ t(b), ab)
    record("matrix_product", left_shape=(2, 3), right_shape=(3, 2),
           output_shape=(2, 2), result=ab, reference="literal Python sums")

    # 5. A mathematical column example; explicit dimensions are kept.
    wn, wt = n([[1, 2, -1], [0, -1, 3]]), t([[1, 2, -1], [0, -1, 3]])
    bn, bt = n([0.5, -1]), t([0.5, -1])
    close_numpy(wn @ xn[:, None] + bn[:, None], [[4.5], [-2]])
    close_torch(wt @ xt[:, None] + bt[:, None], [[4.5], [-2]])
    record("single_column_affine", input_shape=(3, 1), weight_shape=(2, 3),
           output_shape=(2, 1), result=[[4.5], [-2]])

    # 6. Row batches use the transpose of the same W.
    Xn, Xt = n([[2, 1, 0], [0, -1, 2]]), t([[2, 1, 0], [0, -1, 2]])
    Zn, Zt = Xn @ wn.T + bn, Xt @ wt.T + bt
    expected_z = [[4.5, -2], [-3.5, 6]]
    close_numpy(Zn, expected_z)
    close_torch(Zt, expected_z)
    for i in range(2):
        close_numpy(Zn[i], wn @ Xn[i] + bn)
        close_torch(Zt[i], wt @ Xt[i] + bt)
    record("row_batch_affine", input_shape=Xn.shape, weight_shape=wn.shape,
           output_shape=Zn.shape, result=Zn)

    # 7. The framework module stores W as (out_features, in_features).
    layer = torch.nn.Linear(3, 2, bias=True, device="cpu", dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(wt)
        layer.bias.copy_(bt)
        layer_result = layer(Xt)
    close_torch(layer_result, expected_z)
    record("nn_linear_correspondence", weight_shape=tuple(layer.weight.shape),
           bias_shape=tuple(layer.bias.shape), result=layer_result)

    # 8. Broadcasting a bias equals adding 1_B b^T in matrix notation.
    explicit_n = Xn @ wn.T + np.ones((2, 1), dtype=np.float64) @ bn[None, :]
    explicit_t = Xt @ wt.T + torch.ones((2, 1), **kwargs) @ bt[None, :]
    close_numpy(explicit_n, expected_z)
    close_torch(explicit_t, expected_z)
    record("bias_broadcast_equals_explicit_matrix", result=explicit_n)

    # 9. An incompatible shape must fail in both libraries.
    failures = {}
    for name, operation, exception in [
        ("numpy", lambda: np.ones((2, 3)) + np.ones((2,)), ValueError),
        ("pytorch", lambda: torch.ones((2, 3)) + torch.ones((2,)), RuntimeError),
    ]:
        try:
            operation()
        except exception as error:
            failures[name] = type(error).__name__
        else:
            raise AssertionError(f"{name} unexpectedly accepted invalid broadcasting")
    record("incompatible_broadcast_is_rejected", observed_errors=failures)

    # 10. Legal shapes can nevertheless compare the wrong samples.
    pn, yn, pt, yt = n([[1], [2], [3]]), n([1, 2, 3]), t([[1], [2], [3]]), t([1, 2, 3])
    wrong_n, wrong_t = pn - yn, pt - yt
    right_n, right_t = pn - yn[:, None], pt - yt[:, None]
    expected_wrong = [[0, -1, -2], [1, 0, -1], [2, 1, 0]]
    close_numpy(wrong_n, expected_wrong)
    close_torch(wrong_t, expected_wrong)
    close_numpy(right_n, [[0], [0], [0]])
    close_torch(right_t, [[0], [0], [0]])
    close_numpy(np.mean(wrong_n ** 2), 4 / 3)
    close_torch(torch.mean(wrong_t ** 2), 4 / 3)
    close_numpy(np.mean(right_n ** 2), 0)
    close_torch(torch.mean(right_t ** 2), 0)
    record("silent_broadcast_bug", wrong_shape=wrong_n.shape, right_shape=right_n.shape,
           wrong_errors=wrong_n, wrong_mean_square=np.mean(wrong_n ** 2),
           right_mean_square=np.mean(right_n ** 2))

    # 11. Reduction axes and keepdim determine which value is subtracted per row.
    close_numpy(Xn.mean(axis=1, keepdims=True), [[1], [1 / 3]])
    close_torch(Xt.mean(dim=1, keepdim=True), [[1], [1 / 3]])
    centered_n = Xn - Xn.mean(axis=1, keepdims=True)
    centered_t = Xt - Xt.mean(dim=1, keepdim=True)
    close_numpy(centered_n.mean(axis=1), [0, 0])
    close_torch(centered_t.mean(dim=1), [0, 0])
    record("reduction_and_keepdim", row_mean_shape=(2, 1), centered=centered_n,
           centered_row_means=centered_n.mean(axis=1))

    # 12. Equal element counts do not make reshape equal to transpose.
    close_numpy(n(a).reshape(3, 2), [[1, 2], [3, 4], [5, 6]])
    close_torch(t(a).reshape(3, 2), [[1, 2], [3, 4], [5, 6]])
    close_numpy(n(a).T, [[1, 4], [2, 5], [3, 6]])
    close_torch(t(a).T, [[1, 4], [2, 5], [3, 6]])
    record("reshape_is_not_transpose", reshaped=n(a).reshape(3, 2), transposed=n(a).T)

    # 13. Last two axes form matrices; preceding axes are batch axes.
    batch_n = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    batch_t = torch.arange(24, **kwargs).reshape(2, 3, 4)
    rhs_n, rhs_t = np.ones((1, 4, 5), dtype=np.float64), torch.ones((1, 4, 5), **kwargs)
    expected_batch = np.repeat(batch_n.sum(axis=-1, keepdims=True), 5, axis=-1)
    close_numpy(batch_n @ rhs_n, expected_batch)
    close_torch(batch_t @ rhs_t, expected_batch.tolist())
    assert batch_n.swapaxes(-1, -2).shape == tuple(batch_t.transpose(-1, -2).shape) == (2, 4, 3)
    record("batched_matrix_product", left_shape=batch_n.shape, right_shape=rhs_n.shape,
           result_shape=(batch_n @ rhs_n).shape, result=batch_n @ rhs_n,
           matrix_transpose_shape=(2, 4, 3))

    # 14. Different vector norms encode different notions of size.
    vector_n, vector_t = n([-3, 4]), t([-3, 4])
    for order, expected in [(1, 7), (2, 5), (math.inf, 4)]:
        close_numpy(np.linalg.norm(vector_n, ord=order), expected)
        close_torch(torch.linalg.vector_norm(vector_t, ord=order), expected)
    record("vector_norms", l1=7, l2=5, linfinity=4)

    # 15. Frobenius norm differs from the induced matrix 2-norm.
    dn, dt = n([[3, 0], [0, 4]]), t([[3, 0], [0, 4]])
    close_numpy(np.linalg.norm(dn, ord="fro"), 5)
    close_torch(torch.linalg.matrix_norm(dt, ord="fro"), 5)
    close_numpy(np.linalg.norm(dn, ord=2), 4)
    close_torch(torch.linalg.matrix_norm(dt, ord=2), 4)
    close_numpy(np.linalg.norm(dn, axis=1), [3, 4])
    close_torch(torch.linalg.vector_norm(dt, dim=1), [3, 4])
    record("frobenius_and_per_row_norms", frobenius=5, induced_matrix_2_norm=4,
           per_row_norms=[3, 4], per_row_shape=(2,))

    # 16. Geometric coordinates match the figure's labeled arrows.
    gn, gt = n([[2, 1], [0, 1]]), t([[2, 1], [0, 1]])
    close_numpy(gn @ n([1, 0]), [2, 0])
    close_torch(gt @ t([1, 0]), [2, 0])
    close_numpy(gn @ n([0, 1]), [1, 1])
    close_torch(gt @ t([0, 1]), [1, 1])
    close_numpy(gn @ n([1, 2]), [4, 2])
    close_torch(gt @ t([1, 2]), [4, 2])
    record("linear_transform_coordinates", basis1=[2, 0], basis2=[1, 1],
           input=[1, 2], output=[4, 2])

    # 17. Check additivity and homogeneity for transparent fixed examples.
    p, q, alpha = [1, 2], [-1, 3], -2.0
    close_numpy(gn @ (n(p) + n(q)), gn @ n(p) + gn @ n(q))
    close_torch(gt @ (t(p) + t(q)), (gt @ t(p) + gt @ t(q)).tolist())
    close_numpy(gn @ (alpha * n(p)), alpha * (gn @ n(p)))
    close_torch(gt @ (alpha * t(p)), (alpha * (gt @ t(p))).tolist())
    record("linearity_fixed_examples", p=p, q=q, alpha=alpha)

    # 18. A nonzero bias maps zero to nonzero: affine, not linear.
    bias_n, bias_t = n([1, -1]), t([1, -1])
    affine_zero_n, affine_zero_t = gn @ n([0, 0]) + bias_n, gt @ t([0, 0]) + bias_t
    close_numpy(affine_zero_n, [1, -1])
    close_torch(affine_zero_t, [1, -1])
    record("affine_bias_counterexample", affine_zero=affine_zero_n, linear_zero=[0, 0])

    current, peak = tracemalloc.get_traced_memory()
    report = {
        "chapter": "01",
        "status": "passed",
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__,
            "pytorch": torch.__version__, "device": "cpu", "dtype": "float64",
            "torch_num_threads": torch.get_num_threads(), "seed": 0,
            "data": "Fixed arrays written in the chapter; no external dataset",
            "training": "Not applicable: arithmetic validation, no training",
        },
        "tolerances": {"rtol": 1e-10, "atol": 1e-12},
        "resources": {
            "import_seconds": round(IMPORT_SECONDS, 6),
            "elapsed_seconds_including_imports": round(time.perf_counter() - STARTED, 6),
            "python_traced_current_bytes": current,
            "python_traced_peak_bytes": peak,
            "memory_scope": "tracemalloc Python allocations only; excludes full native tensor memory and process RSS",
            "largest_explicit_teaching_result_elements": 30,
        },
        "summary": {"passed": len(checks), "failed": 0},
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(checks)} checks passed; report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
