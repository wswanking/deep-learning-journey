"""Chapter 06: from-scratch perceptron on OR and XOR truth tables, CPU/NumPy.

Prediction chooses +1 at score zero. Training updates every nonpositive signed
margin, including zero-margin positive samples that prediction happens to accept.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import platform
import sys
import time
from xml.sax.saxutils import escape

import numpy as np


def predict(features, weights, bias):
    scores = np.asarray(features, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + bias
    return np.where(scores >= 0.0, 1, -1)


def train(features, labels, learning_rate=1.0, max_epochs=30):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("Expected features (N,d) and labels (N,)")
    if not np.all(np.isfinite(x)) or not np.all(np.isin(y, [-1.0, 1.0])):
        raise ValueError("Finite inputs and labels -1/+1 are required")
    if learning_rate <= 0 or not np.isfinite(learning_rate) or max_epochs < 1:
        raise ValueError("Positive learning rate and epoch budget are required")
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    history, updates = [], []
    converged = False
    for epoch in range(1, max_epochs + 1):
        epoch_updates = 0
        for index in range(x.shape[0]):
            score = float(x[index] @ weights + bias)
            signed_margin = float(y[index] * score)
            if signed_margin <= 0.0:
                before = {"weights": weights.tolist(), "bias": bias}
                weights += learning_rate * y[index] * x[index]
                bias += learning_rate * y[index]
                epoch_updates += 1
                updates.append({
                    "number": len(updates) + 1, "epoch": epoch, "sample_index": index,
                    "x": x[index].tolist(), "y": int(y[index]), "score_before": score,
                    "signed_margin_before": signed_margin, "before": before,
                    "after": {"weights": weights.tolist(), "bias": bias},
                })
        scores = x @ weights + bias
        history.append({
            "epoch": epoch, "updates": epoch_updates, "total_updates": len(updates),
            "training_accuracy": float(np.mean(predict(x, weights, bias) == y)),
            "nonpositive_signed_margins": int(np.sum(y * scores <= 0.0)),
            "minimum_signed_margin": float(np.min(y * scores)),
            "weights": weights.tolist(), "bias": bias,
        })
        if epoch_updates == 0:
            converged = True
            break
    return {"weights": weights.tolist(), "bias": bias,
            "epochs": len(history), "total_updates": len(updates),
            "converged": converged, "predictions": predict(x, weights, bias).tolist(),
            "scores": (x @ weights + bias).tolist(), "history": history, "updates": updates}


def svg_text(x, y, value, size=22, attrs=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" {attrs}>{escape(str(value))}</text>'


def svg_line(x1, y1, x2, y2, stroke="#8a9eb4", width=2, attrs=""):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" {attrs}/>'


def svg_rect(x, y, w, h, fill, attrs=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" {attrs}/>'


def save_svg(folder, name, height, title, description, body):
    document = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" '
        f'viewBox="0 0 900 {height}" role="img" aria-labelledby="c06-{name}-title c06-{name}-desc">'
        f'<title id="c06-{name}-title">{escape(title)}</title>'
        f'<desc id="c06-{name}-desc">{escape(description)}</desc>'
        + svg_rect(0, 0, 900, height, "#f7f9fc", 'rx="16"')
        + '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">'
        + body + '</g></svg>\n'
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.svg").write_text(document, encoding="utf-8")


def write_figures(report, folder):
    blue, orange, ink = "#246cc0", "#bd611a", "#172c44"
    body = svg_text(30, 42, "一个感知机：加权求和，加偏置，再做阈值判断", 28, 'font-weight="700"')
    body += svg_text(30, 78, "输入是数值向量；输出是 −1 / +1 标签，分数本身不是概率。", 21)
    for yy, label, weight in [(155, "x₁", "w₁"), (260, "x₂", "w₂"), (365, "x₃", "w₃")]:
        body += svg_rect(45, yy - 30, 115, 60, "#e6effb", 'rx="10" stroke="#5f8dbe" stroke-width="2"')
        body += svg_text(102, yy + 8, label, 26, 'text-anchor="middle"')
        body += svg_line(160, yy, 375, 260, blue, 3)
        body += svg_text(230, yy + (260 - yy) * .35 - 10, weight, 23)
    body += '<circle cx="410" cy="260" r="44" fill="#fff" stroke="#5f8dbe" stroke-width="2"/>'
    body += svg_text(410, 269, "Σ + b", 25, 'text-anchor="middle"')
    body += svg_text(410, 337, "分数 z", 22, 'text-anchor="middle"')
    body += svg_line(455, 260, 565, 260, blue, 3)
    body += svg_rect(565, 215, 165, 90, "#fff", 'rx="12" stroke="#5f8dbe" stroke-width="2"')
    body += svg_text(647, 251, "阈值激活", 23, 'text-anchor="middle"')
    body += svg_text(647, 284, "g(z)", 23, 'text-anchor="middle"')
    body += svg_line(730, 260, 790, 260, blue, 3)
    body += svg_text(835, 252, "ŷ", 28, 'text-anchor="middle"')
    body += svg_text(835, 285, "±1", 22, 'text-anchor="middle"')
    body += svg_text(30, 439, "d=3示意：3个权重 + 1个偏置；g(z)=+1（z≥0），否则−1。", 22)
    save_svg(folder, "perceptron-unit", 475, "图06-1：感知机的仿射计算与阈值激活", "输入d个数值分量，各自乘权重后求和，加标量偏置得到z。阈值函数在z大于等于0时输出正一，否则负一。", body)

    # Plot the actual learned OR boundary. XOR terminal weights are zero, so it
    # truly has no dividing line: show its all-positive region instead of inventing one.
    body = svg_text(30, 42, "同一训练规则：OR可分，XOR终态仍然出错", 28, 'font-weight="700"')
    body += svg_text(30, 77, "点的形状表示真实标签；背景表示最终模型预测，参数来自实际运行。", 21)
    data = report["data"]["features"]
    for offset, key, target in [(65, "or", report["data"]["or_labels"]), (505, "xor", report["data"]["xor_labels"])]:
        result = report[key]
        w, bias = result["weights"], result["bias"]
        title = "OR：w=(2,2)，b=−1" if key == "or" else "XOR：w=(0,0)，b=0"
        body += svg_text(offset - 20, 123, title, 23, 'font-weight="700"')
        left, top, size, lo, hi = offset, 165, 300, -.25, 1.25
        sx = lambda v: left + (v - lo) / (hi - lo) * size
        sy = lambda v: top + size - (v - lo) / (hi - lo) * size
        body += svg_rect(left, top, size, size, "#e5effc", 'stroke="#8a9eb4" stroke-width="1"')
        if key == "or":
            # x1+x2<0.5 within the square gives a right triangle.
            negative = [(lo, lo), (.5 - lo, lo), (lo, .5 - lo)]
            points = " ".join(f"{sx(a):.2f},{sy(b):.2f}" for a, b in negative)
            body += f'<polygon points="{points}" fill="#fae9dc"/>'
            body += svg_line(sx(lo), sy(.5 - lo), sx(.5 - lo), sy(lo), ink, 3)
        for tick in [0, 1]:
            body += svg_line(sx(tick), top, sx(tick), top + size, "#d0dbe7", 1)
            body += svg_line(left, sy(tick), left + size, sy(tick), "#d0dbe7", 1)
            body += svg_text(sx(tick), top + size + 28, str(tick), 21, 'text-anchor="middle"')
            body += svg_text(left - 12, sy(tick) + 7, str(tick), 21, 'text-anchor="end"')
        for index, ((a, b), label) in enumerate(zip(data, target)):
            px, py = sx(a), sy(b)
            if label == 1:
                body += f'<circle cx="{px}" cy="{py}" r="10" fill="{blue}" stroke="#fff" stroke-width="2"/>'
            else:
                body += svg_rect(px - 10, py - 10, 20, 20, orange, 'stroke="#fff" stroke-width="2"')
            if result["predictions"][index] != label:
                body += f'<circle cx="{px}" cy="{py}" r="17" fill="none" stroke="{orange}" stroke-width="2" stroke-dasharray="4 3"/>'
                body += svg_text(px + 23, py - 14, "错", 21)
        body += svg_text(left + size - 2, top + size + 61, "x₁", 23, 'text-anchor="end"')
        body += svg_text(left - 5, top - 13, "x₂", 23)
        footer = "准确率100%；边界 x₁+x₂=0.5" if key == "or" else "准确率50%；全部预测+1，无分割线"
        body += svg_text(offset - 20, 566, footer, 21)
    body += svg_text(30, 607, "● 真值+1；■ 真值−1；虚线外圈标出误分类。训练分数零时按+1预测。", 21)
    save_svg(folder, "or-xor-boundaries", 640, "图06-3：实际训练后的OR决策边界和XOR退化终态", "OR终态权重为2和2，偏置负1，直线x1加x2等于0.5正确分开全部四点。XOR终态权重和偏置全零，按零分数预测正类的约定全部预测正类，两个负类错误，没有有效分割线。", body)

    body = svg_text(30, 42, "XOR的限制来自单条直线能表达什么", 28, 'font-weight="700"')
    body += svg_text(30, 78, "两类对角线共享中点(0.5,0.5)；右侧用本章阈值约定给出矛盾。", 21)
    left, top, size = 80, 140, 300
    sx = lambda v: left + v * size
    sy = lambda v: top + size - v * size
    body += svg_rect(left, top, size, size, "#fff", 'stroke="#8a9eb4" stroke-width="2"')
    body += svg_line(sx(0), sy(0), sx(1), sy(1), orange, 3, 'stroke-dasharray="8 5"')
    body += svg_line(sx(0), sy(1), sx(1), sy(0), blue, 3, 'stroke-dasharray="8 5"')
    for a, b, label in [(0, 0, -1), (1, 1, -1), (0, 1, 1), (1, 0, 1)]:
        px, py = sx(a), sy(b)
        if label == 1:
            body += f'<circle cx="{px}" cy="{py}" r="10" fill="{blue}"/>'
        else:
            body += svg_rect(px - 10, py - 10, 20, 20, orange)
        body += svg_text(px + (20 if a == 0 else -20), py + (-17 if b == 0 else 28), f"({a},{b})", 21, f'text-anchor="{"start" if a == 0 else "end"}"')
    body += f'<circle cx="{sx(.5)}" cy="{sy(.5)}" r="7" fill="{ink}"/>'
    body += svg_text(230, 475, "两类对角线的中点相同", 21, 'text-anchor="middle"')
    body += svg_text(455, 160, "负类要求：", 24, 'font-weight="700"')
    body += svg_text(455, 200, "b < 0；w₁+w₂+b < 0", 23)
    body += svg_text(455, 245, "相加：w₁+w₂+2b < 0", 23, f'fill="{orange}"')
    body += svg_text(455, 305, "正类要求：", 24, 'font-weight="700"')
    body += svg_text(455, 345, "w₁+b ≥ 0；w₂+b ≥ 0", 23)
    body += svg_text(455, 390, "相加：w₁+w₂+2b ≥ 0", 23, f'fill="{blue}"')
    body += svg_text(455, 455, "同一个量无法同时 <0 和 ≥0。", 23, 'font-weight="700"')
    save_svg(folder, "xor-impossibility", 510, "图06-2：XOR无法被单一仿射阈值分类器表示", "负类在零零和一一，正类在零一和一零。正类条件相加要求w1加w2加2b大于等于零，负类条件相加要求同一个量小于零，形成矛盾。", body)

    body = svg_text(30, 42, "每轮更新数：可分时停止，XOR持续出现违例", 28, 'font-weight="700"')
    body += svg_text(30, 78, "固定数据顺序与学习率1；更新数包含落在边界上的样本。", 21)
    left, top, width, height, xmax, ymax = 90, 125, 730, 270, 30, 4.5
    sx = lambda epoch: left + (epoch - 1) / (xmax - 1) * width
    sy = lambda count: top + height - count / ymax * height
    for tick in range(5):
        body += svg_line(left, sy(tick), left + width, sy(tick), "#d0dbe7", 1)
        body += svg_text(left - 15, sy(tick) + 7, str(tick), 21, 'text-anchor="end"')
    body += svg_line(left, top, left, top + height, ink)
    body += svg_line(left, top + height, left + width, top + height, ink)
    for tick in [1, 5, 10, 20, 30]:
        body += svg_text(sx(tick), top + height + 32, str(tick), 21, 'text-anchor="middle"')
    body += svg_text(30, 115, "更新数", 21)
    body += svg_text(820, 458, "训练轮次", 22, 'text-anchor="end"')
    for key, color, dash in [("or", blue, ""), ("xor", orange, 'stroke-dasharray="9 5"')]:
        points = [(sx(row["epoch"]), sy(row["updates"])) for row in report[key]["history"]]
        path_data = " ".join(f'{"M" if i == 0 else "L"} {x:.3f} {y:.3f}' for i, (x, y) in enumerate(points))
        body += f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="3" {dash}/>'
        for x, y in points:
            body += f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>'
    body += svg_text(30, 502, "蓝实线 OR：共9次更新，第6轮无更新后停止。", 22)
    body += svg_text(30, 542, "橙虚线 XOR：30轮共120次更新，达到预算；每轮参数回到零。", 22)
    body += svg_text(30, 582, "OR停止后未继续运行，不将后面的轮次伪装为实测零更新。", 21)
    save_svg(folder, "training-updates", 615, "图06-4：两种任务每轮实际更新次数", "OR六轮更新次数为三、一、二、二、一、零，总共九次。XOR在三十轮中每轮更新四次，达到预算仍未收敛。OR曲线只画到实际运行的第六轮。", body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/experiments/06-perceptron"))
    parser.add_argument("--write-figures", action="store_true", help="Rebuild chapter SVG sources from this exact run")
    args = parser.parse_args()
    started = time.perf_counter()
    features = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float64)
    or_labels = np.array([-1, 1, 1, 1], dtype=np.int64)
    xor_labels = np.array([-1, 1, 1, -1], dtype=np.int64)
    result_or = train(features, or_labels)
    result_xor = train(features, xor_labels)
    checks = []

    def require(condition, label):
        if not bool(condition):
            raise RuntimeError(f"Validation failed: {label}")
        checks.append(label)

    require(result_or["converged"] and result_or["epochs"] == 6, "or_stops_after_zero_update_pass")
    require(result_or["weights"] == [2.0, 2.0] and result_or["bias"] == -1.0, "or_parameters_match_fixed_order")
    require(result_or["predictions"] == or_labels.tolist(), "or_truth_table_all_correct")
    require(result_or["total_updates"] == 9, "or_uses_nine_updates")
    require(result_or["history"][-1]["minimum_signed_margin"] > 0, "or_strict_positive_margin")
    require(result_or["updates"][0]["after"] == {"weights": [0.0, 0.0], "bias": -1.0}, "first_hand_update_changes_bias")
    require(result_or["updates"][1]["after"] == {"weights": [0.0, 1.0], "bias": 0.0}, "second_hand_update_matches")
    require(result_or["updates"][2]["after"] == {"weights": [1.0, 1.0], "bias": 1.0}, "positive_zero_margin_is_updated")
    require(not result_xor["converged"] and result_xor["epochs"] == 30, "xor_reaches_epoch_budget")
    require(result_xor["total_updates"] == 120, "xor_has_120_updates")
    require(all(row["updates"] == 4 and row["weights"] == [0.0, 0.0] and row["bias"] == 0.0 for row in result_xor["history"]), "xor_cycle_is_observed_each_pass")
    require(result_xor["history"][-1]["training_accuracy"] == 0.5, "xor_final_accuracy_is_half")
    require(np.array_equal(predict(features, [0., 0.], 0.), np.ones(4)), "score_zero_prediction_is_positive")
    require(np.array_equal(predict(features, [2., 2.], -1.), predict(features, [4., 4.], -2.)), "positive_rescaling_preserves_predictions")
    augmented = np.column_stack([features, np.ones(len(features))])
    separator = np.array([1.0, 1.0, -0.5])
    unit_separator = separator / np.linalg.norm(separator)
    gamma = float(np.min(or_labels * (augmented @ unit_separator)))
    radius_squared = float(np.max(np.sum(augmented**2, axis=1)))
    update_bound = radius_squared / gamma**2
    require(gamma > 0 and result_or["total_updates"] <= update_bound + 1e-12, "or_satisfies_convergence_update_bound")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {"chapter": "06", "weights": result_or["weights"], "bias": result_or["bias"],
                  "prediction_rule": "+1 if score >= 0, otherwise -1", "purpose": "inference parameters; no optimizer state"}
    checkpoint_path = args.output_dir / "or-model.json"
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    restored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    require(np.array_equal(predict(features, restored["weights"], restored["bias"]), or_labels), "saved_parameters_restore_predictions")
    report = {
        "schema_version": 1, "chapter": "06",
        "collected_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "python_executable": sys.executable, "platform": platform.platform(),
                        "device": "CPU", "dtype": "float64"},
        "configuration": {"learning_rate": 1.0, "max_epochs": 30, "seed": None,
                          "randomness": "none: fixed truth tables, zero initialization, fixed cyclic order",
                          "initial_weights": [0.0, 0.0], "initial_bias": 0.0,
                          "sample_order": [0, 1, 2, 3], "parameters": 3,
                          "update_condition": "y * score <= 0", "prediction_at_zero": 1,
                          "split": "full four-point truth table; training diagnostics, no statistical test set",
                          "optimizer": "online perceptron update; no autograd"},
        "data": {"source": "synthetic OR/XOR truth tables, no download", "features": features.tolist(),
                 "or_labels": or_labels.tolist(), "xor_labels": xor_labels.tolist()},
        "or": result_or, "xor": result_xor,
        "or_convergence_bound": {"augmented_unit_separator": unit_separator.tolist(),
                                 "gamma": gamma, "radius_squared": radius_squared,
                                 "maximum_updates_bound": update_bound},
        "checkpoint": {"path": str(checkpoint_path), "restored_predictions_match": True,
                       "scope": "inference weights only; training state is not resumed"},
        "resources": {"elapsed_seconds": time.perf_counter() - started,
                      "input_and_label_payload_bytes": features.nbytes + or_labels.nbytes + xor_labels.nbytes,
                      "memory_scope": "three input/label arrays only; process peak memory not measured"},
        "checks": {"status": "passed", "count": len(checks), "names": checks},
    }
    if args.write_figures:
        root = Path(__file__).resolve().parents[2]
        write_figures(report, root / "docs/assets/06-perceptron")
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (args.output_dir / "results.json").write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps({"collected_at": report["collected_at"], "environment": report["environment"],
                      "or": {key: result_or[key] for key in ["weights", "bias", "epochs", "total_updates", "converged", "predictions"]},
                      "xor": {key: result_xor[key] for key in ["weights", "bias", "epochs", "total_updates", "converged", "predictions"]},
                      "or_convergence_bound": report["or_convergence_bound"], "resources": report["resources"],
                      "checks": report["checks"], "output": str(args.output_dir / "results.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
