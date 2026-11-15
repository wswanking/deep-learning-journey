"""Chapter 03: finite probability tables, Monte Carlo moments, and information measures.

Synthetic data only. No model training, external download, or GPU is required.
Run from the repository root with its .venv interpreter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np


def probability_vector(values):
    """Validate a finite PMF; do not silently normalize malformed inputs."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("A probability vector must be nonempty and one-dimensional")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.isclose(result.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Probabilities must sum to one")
    return result


def information(p_values, q_values):
    """Return discrete H(p), H(p,q), KL(p||q), in nats.

    Zero p entries contribute zero, even when q is also zero. A positive p
    paired with zero q gives infinite cross-entropy and KL, never an epsilon
    replacement. Logs are subtracted rather than forming a possibly unsafe ratio.
    """
    p = probability_vector(p_values)
    q = probability_vector(q_values)
    if p.shape != q.shape:
        raise ValueError("The two distributions must share the same state space")
    mask = p > 0
    log_p = np.log(p[mask])
    entropy = float(-np.dot(p[mask], log_p))
    if np.any(q[mask] == 0):
        return {"entropy_nats": entropy, "cross_entropy_nats": math.inf,
                "kl_nats": math.inf}
    log_q = np.log(q[mask])
    return {
        "entropy_nats": entropy,
        "cross_entropy_nats": float(-np.dot(p[mask], log_q)),
        "kl_nats": float(np.dot(p[mask], log_p - log_q)),
    }


def bernoulli_log_likelihood(theta, observations):
    """Log likelihood of an ordered sequence of independent Bernoulli trials."""
    data = np.asarray(observations)
    if data.ndim != 1 or data.size == 0 or not np.all((data == 0) | (data == 1)):
        raise ValueError("Observations must be a nonempty sequence of zeros and ones")
    if not math.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    successes = int(data.sum())
    failures = data.size - successes
    if theta == 0.0:
        return 0.0 if successes == 0 else -math.inf
    if theta == 1.0:
        return 0.0 if failures == 0 else -math.inf
    return float(successes * math.log(theta) + failures * math.log1p(-theta))


def json_safe(value):
    """Serialize mathematical infinity explicitly; JSON itself has no infinity."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("Unexpected NaN in experiment output")
        return "+inf" if value > 0 else "-inf"
    return value


def run(seed):
    started = time.perf_counter()
    checks = []

    def require(condition, name):
        if not bool(condition):
            raise RuntimeError(f"Validation failed: {name}")
        checks.append(name)

    joint = np.array([[0.10, 0.20], [0.30, 0.40]], dtype=np.float64)
    marginal_x = joint.sum(axis=1)
    marginal_y = joint.sum(axis=0)
    conditional_x_given_y1 = joint[:, 1] / marginal_y[1]
    y1_given_x1 = joint[1, 1] / marginal_x[1]
    bayes_x1_given_y1 = y1_given_x1 * marginal_x[1] / marginal_y[1]
    require(np.allclose(marginal_x, [0.3, 0.7], atol=1e-14, rtol=0),
            "joint_table_marginal_x")
    require(np.allclose(marginal_y, [0.4, 0.6], atol=1e-14, rtol=0),
            "joint_table_marginal_y")
    require(abs(conditional_x_given_y1.sum() - 1.0) < 1e-14,
            "conditional_distribution_normalizes")
    require(abs(bayes_x1_given_y1 - 2.0 / 3.0) < 1e-14,
            "bayes_matches_direct_conditioning")

    values = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    probabilities = probability_vector([0.25, 0.50, 0.25])
    expected_mean = float(values @ probabilities)
    expected_second = float((values**2) @ probabilities)
    expected_variance = expected_second - expected_mean**2
    generator = np.random.Generator(np.random.PCG64(seed))
    sample_sizes = [100, 1000, 10000]
    draws = generator.choice(values, size=max(sample_sizes), p=probabilities)
    estimates = []
    for count in sample_sizes:
        sample = draws[:count]
        estimates.append({
            "samples": count,
            "mean": float(sample.mean()),
            "mean_absolute_error": float(abs(sample.mean() - expected_mean)),
            "variance_ddof0": float(sample.var(ddof=0)),
            "variance_ddof1": float(sample.var(ddof=1)),
        })
    require(abs(expected_mean - 1.0) < 1e-14 and abs(expected_variance - 0.5) < 1e-14,
            "exact_moments_match_hand_calculation")
    require(abs(estimates[-1]["mean"] - expected_mean) < 0.03,
            "fixed_seed_10000_sample_mean_within_0_03")
    require(abs(estimates[-1]["variance_ddof0"] - expected_variance) < 0.03,
            "fixed_seed_10000_sample_variance_within_0_03")

    observations = [1, 0, 1, 1]
    likelihoods = []
    for theta in [0.0, 0.5, 0.75, 1.0]:
        log_value = bernoulli_log_likelihood(theta, observations)
        likelihoods.append({"theta": theta, "log_likelihood": log_value,
                            "likelihood": math.exp(log_value)})
    require(abs(likelihoods[1]["likelihood"] - 0.0625) < 1e-14,
            "ordered_sequence_likelihood_theta_half")
    require(abs(likelihoods[2]["likelihood"] - 0.10546875) < 1e-14,
            "ordered_sequence_likelihood_theta_three_quarters")
    require(likelihoods[0]["likelihood"] == 0.0 and likelihoods[-1]["likelihood"] == 0.0,
            "bernoulli_likelihood_endpoint_limits")

    p = [0.25, 0.75]
    candidates = [
        ("matched", [0.25, 0.75]),
        ("uniform", [0.50, 0.50]),
        ("reversed", [0.75, 0.25]),
    ]
    comparisons = []
    for name, q in candidates:
        result = information(p, q)
        error = abs(result["cross_entropy_nats"]
                    - result["entropy_nats"] - result["kl_nats"])
        require(error < 1e-12, f"cross_entropy_identity_{name}")
        require(result["kl_nats"] >= -1e-12, f"kl_nonnegative_{name}")
        comparisons.append({"name": name, "q": q, **result,
                            "identity_absolute_error": error})
    reverse_kl = information([0.5, 0.5], p)["kl_nats"]
    forward_kl = comparisons[1]["kl_nats"]
    require(abs(forward_kl - reverse_kl) > 1e-3, "kl_is_not_symmetric")
    zero_supported = information([1.0, 0.0], [1.0, 0.0])
    missing_support = information([0.25, 0.75], [1.0, 0.0])
    tiny_probability = information([0.0, 1.0], [1.0, 1e-300])
    require(zero_supported["cross_entropy_nats"] == 0.0
            and zero_supported["kl_nats"] == 0.0,
            "zero_p_terms_are_skipped")
    require(math.isinf(missing_support["cross_entropy_nats"])
            and math.isinf(missing_support["kl_nats"]),
            "positive_p_and_zero_q_produces_infinity")
    require(math.isfinite(tiny_probability["cross_entropy_nats"])
            and abs(tiny_probability["cross_entropy_nats"] - 300 * math.log(10)) < 1e-10,
            "tiny_positive_probability_is_not_clipped")
    try:
        probability_vector([0.2, 0.2])
    except ValueError:
        checks.append("unnormalized_probabilities_are_rejected")
    else:
        raise RuntimeError("Invalid probabilities were silently accepted")

    return {
        "schema_version": 1,
        "chapter": "03",
        "collected_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "python_executable": sys.executable, "platform": platform.platform(),
            "device": "CPU", "dtype": "float64",
        },
        "configuration": {
            "seed": seed, "generator": "numpy.Generator(PCG64)",
            "sample_sizes": sample_sizes, "data_source": "synthetic finite distributions",
            "train_validation_test_split": "not_applicable: numerical simulation, no fitted model",
            "model_parameters": "not_applicable", "optimizer": "not_applicable",
            "learning_rate": "not_applicable", "training_epochs": "not_applicable",
            "checkpoint": "not_applicable: deterministic rerun from seed",
            "logarithm_base": "e", "information_unit": "nat",
        },
        "joint_table": {
            "rows_x_columns_y": joint.tolist(),
            "marginal_x": marginal_x.tolist(), "marginal_y": marginal_y.tolist(),
            "p_x_given_y1": conditional_x_given_y1.tolist(),
            "p_y1_given_x1": float(y1_given_x1),
            "bayes_p_x1_given_y1": float(bayes_x1_given_y1),
            "independent": bool(np.allclose(joint, np.outer(marginal_x, marginal_y))),
        },
        "moments": {
            "values": values.tolist(), "probabilities": probabilities.tolist(),
            "exact_mean": expected_mean, "exact_second_moment": expected_second,
            "exact_variance": expected_variance, "estimates": estimates,
        },
        "ordered_bernoulli_data": {
            "observations": observations, "successes": sum(observations),
            "empirical_success_probability": sum(observations) / len(observations),
            "comparisons": likelihoods,
        },
        "information": {
            "p": p, "comparisons": comparisons,
            "kl_uniform_to_p_nats": reverse_kl,
            "zero_p_zero_q": zero_supported, "missing_support": missing_support,
            "tiny_positive_q": tiny_probability,
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "sample_array_bytes": draws.nbytes,
            "sample_array_bytes_scope": "NumPy sample payload only; excludes interpreter and library memory",
            "process_peak_memory": "not_measured",
        },
        "checks": {"status": "passed", "count": len(checks), "names": checks},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--output", type=Path,
                        default=Path("build/experiments/03-probability.json"))
    arguments = parser.parse_args()
    report = json_safe(run(arguments.seed))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
