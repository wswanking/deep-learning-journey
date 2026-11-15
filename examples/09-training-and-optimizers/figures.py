"""Pure SVG figures derived from chapter 09 numerical outputs. No bitmap assets."""
from __future__ import annotations
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

BLUE, ORANGE, GREEN, PURPLE = "#246cc0", "#bd611a", "#39805b", "#8954a3"
INK, GRID = "#172c44", "#d0dbe7"


def text(x, y, value, size=22, extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(str(value))}</text>'


def line(x1, y1, x2, y2, color=GRID, width=1.5, extra=""):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {extra}/>'


def rect(x, y, w, h, fill="#fff", extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" {extra}/>'


def path(points, color, width=3, extra=""):
    if not points:
        return ""
    data = " ".join(f'{"M" if i == 0 else "L"} {x:.3f} {y:.3f}' for i, (x, y) in enumerate(points))
    return f'<path d="{data}" fill="none" stroke="{color}" stroke-width="{width}" {extra}/>'


def save(folder, name, height, title, desc, body):
    definition = f'<defs><marker id="c09-{name}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{BLUE}"/></marker></defs>'
    source = (f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}" role="img" aria-labelledby="c09-{name}-title c09-{name}-desc">'
              f'<title id="c09-{name}-title">{escape(title)}</title><desc id="c09-{name}-desc">{escape(desc)}</desc>'
              + definition + rect(0, 0, 900, height, "#f7f9fc", 'rx="16"')
              + '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172c44">' + body + '</g></svg>\n')
    (folder / f"{name}.svg").write_text(source, encoding="utf-8")


def curve_plot(histories, left, top, width, height, xmax=60, labels=True):
    sx = lambda value: left + value / xmax * width
    sy = lambda value: top + height - (math.log10(value) + 3) / 3 * height
    body = ""
    for tick in [1.0, 0.1, 0.01, 0.001]:
        body += line(left, sy(tick), left + width, sy(tick))
        if labels: body += text(left - 12, sy(tick) + 7, f"{tick:g}", 20, 'text-anchor="end"')
    body += line(left, top, left, top + height, INK, 2) + line(left, top + height, left + width, top + height, INK, 2)
    for tick in [0, 30, 60]:
        body += text(sx(tick), top + height + 30, str(tick), 21, 'text-anchor="middle"')
    for history, color, dash in histories:
        body += path([(sx(row["passes"]), sy(row["train_mse"])) for row in history], color, 3, dash)
    return body


def write_figures(report, folder):
    folder.mkdir(parents=True, exist_ok=True)
    body = text(30, 42, "训练循环：每一步负责什么", 28, 'font-weight="700"')
    body += text(30, 78, "反向传播计算梯度；优化器使用梯度和自身状态更新参数。", 21)
    xs = [30, 245, 460, 675]
    labels = [("取mini-batch", "设置本步 ηₖ"), ("清除旧梯度", "zero_grad"), ("前向与损失", "得到标量损失"), ("反向求导", "backward")]
    lower = [("下一批数据", "更新计数 k"), ("评估与记录", "固定验证集"), ("参数与状态更新", "optimizer.step"), ("可选梯度裁剪", "限制梯度范数")]
    for yy, cells in [(115, labels), (290, lower)]:
        for x, (title, subtitle) in zip(xs, cells):
            body += rect(x, yy, 195, 92, "#fff", 'rx="12" stroke="#8a9eb4" stroke-width="2"')
            body += text(x + 97.5, yy + 37, title, 23, 'text-anchor="middle" font-weight="700"')
            body += text(x + 97.5, yy + 72, subtitle, 21, 'text-anchor="middle"')
    marker = 'marker-end="url(#c09-training-loop-arrow)"'
    for start in [225, 440, 655]: body += line(start, 161, start + 17, 161, BLUE, 3, marker)
    body += line(772, 207, 772, 285, BLUE, 3, marker)
    for start in [675, 460, 245]: body += line(start, 336, start - 17, 336, BLUE, 3, marker)
    body += f'<path d="M 127 290 V 248 H 12 V 161 H 26" fill="none" stroke="{BLUE}" stroke-width="3" {marker}/>'
    body += text(30, 432, "梯度清零在本次 backward 前；裁剪在 backward 后、step 前。", 22)
    body += text(30, 474, "验证只前向：eval + no_grad；恢复训练时还要恢复优化器与采样状态。", 21)
    save(folder, "training-loop", 505, "图09-1：一次训练迭代的职责与顺序", "先取批次并设置学习率，清除旧梯度，前向计算损失，反向计算梯度，按需裁剪，更新参数和优化器状态，再做评估记录并进入下一批。", body)

    body = text(30, 42, "同一个二次目标，不同更新规则产生不同路径", 28, 'font-weight="700"')
    body += text(30, 78, "J(θ)=0.5(θ₁²+10θ₂²)，初始(4,2)，各运行30步；细线为等高线。", 21)
    left, top, width, height = 90, 120, 720, 330
    sx = lambda x: left + (x + 2.5) / 7 * width
    sy = lambda y: top + height - (y + 2.5) / 5 * height
    for xx in [-2, 0, 2, 4]:
        body += line(sx(xx), top, sx(xx), top + height)
        body += text(sx(xx), top + height + 30, str(xx), 21, 'text-anchor="middle"')
    for yy in [-2, 0, 2]:
        body += line(left, sy(yy), left + width, sy(yy))
        body += text(left - 12, sy(yy) + 7, str(yy), 21, 'text-anchor="end"')
    for level in [1, 4, 16, 40]:
        segment = []
        for index in range(361):
            angle = index / 360 * 2 * math.pi
            a, b = math.sqrt(2 * level) * math.cos(angle), math.sqrt(2 * level / 10) * math.sin(angle)
            if -2.5 <= a <= 4.5 and -2.5 <= b <= 2.5:
                segment.append((sx(a), sy(b)))
            else:
                body += path(segment, "#bcc9d7", 1.5); segment = []
        body += path(segment, "#bcc9d7", 1.5)
    for key, color, dash in [("sgd", BLUE, ""), ("momentum", ORANGE, 'stroke-dasharray="9 4"'), ("adam", GREEN, 'stroke-dasharray="3 3"')]:
        points = [(sx(a), sy(b)) for a, b in report["quadratic_paths"][key]["points"]]
        body += path(points, color, 3, dash)
        body += f'<circle cx="{points[-1][0]}" cy="{points[-1][1]}" r="4" fill="{color}"/>'
    body += text(820, 493, "θ₁", 23, 'text-anchor="end"') + text(34, 118, "θ₂", 23)
    body += text(30, 535, "蓝实线 SGD η=0.15；橙虚线 Momentum η=0.08、μ=0.9。", 22)
    body += text(30, 575, "绿点线 Adam η=0.15；参数为教学配置，不作优化器通用排名。", 22)
    save(folder, "optimizer-paths", 610, "图09-2：三种优化器在同一二次目标上的真实路径", "二次函数两个方向曲率不同，SGD、带动量SGD与Adam从同一点开始，各用明确学习率运行三十步，显示真实二维参数路径。", body)

    experiments = report["experiments"]
    body = text(30, 42, "49参数MLP：记录整份训练数据的平均平方误差", 28, 'font-weight="700"')
    body += text(30, 78, "同一初始化与样本顺序；均看过3840个样本，即60遍训练数据。", 21)
    configurations = [("sgd", BLUE, ""), ("momentum", ORANGE, 'stroke-dasharray="10 4"'), ("adam", GREEN, 'stroke-dasharray="4 3"'), ("adamw_decay", PURPLE, 'stroke-dasharray="10 3 3 3"')]
    body += curve_plot([(experiments[k]["history"], color, dash) for k, color, dash in configurations], 95, 130, 710, 300)
    body += text(30, 120, "MSE（对数轴）", 21) + text(810, 495, "已看数据遍数", 22, 'text-anchor="end"')
    for x, yy, label, color in [(30, 535, "SGD η=0.03", BLUE), (450, 535, "Momentum η=0.03、μ=0.9", ORANGE), (30, 577, "Adam η=0.01、λ=0", GREEN), (450, 577, "AdamW η=0.01、λ=0.01", PURPLE)]:
        body += line(x, yy - 7, x + 32, yy - 7, color, 4) + text(x + 45, yy, label, 21)
    body += text(30, 620, "图中为数据MSE；AdamW含额外衰减，该组同时改变了正则化设置。", 21)
    save(folder, "mlp-training-curves", 650, "图09-4：小MLP各配置的真实训练损失", "四组曲线来自CPU训练日志，每组都处理六十遍训练数据。纵轴为对数刻度的数据MSE。动量、学习率和权重衰减配置在图例明确，不能据此推断某优化器对所有任务都最好。", body)

    body = text(30, 42, "控制变量：学习率与批次的两组对照", 28, 'font-weight="700"')
    body += text(30, 78, "均用Adam、λ=0、同一模型与数据；批次对照固定已看数据量。", 21)
    body += text(55, 126, "学习率：0.01 与 0.001", 23, 'font-weight="700"')
    body += text(485, 126, "批次：16 与 64", 23, 'font-weight="700"')
    body += curve_plot([(experiments["adam"]["history"], BLUE, ""), (experiments["adam_small_lr"]["history"], ORANGE, 'stroke-dasharray="9 4"')], 85, 165, 300, 260)
    body += curve_plot([(experiments["adam"]["history"], BLUE, ""), (experiments["adam_full_batch"]["history"], ORANGE, 'stroke-dasharray="9 4"')], 515, 165, 300, 260)
    body += text(85, 490, "蓝线 η=0.01；橙虚线 0.001", 20)
    body += text(515, 490, "蓝线 B=16；橙虚线 B=64", 20)
    body += text(30, 535, "两面板横轴：已看数据遍数；纵轴：训练MSE，对数刻度。", 22)
    body += text(30, 576, "B=16更新240次，B=64更新60次；固定数据量不等于固定更新次数。", 21)
    save(folder, "learning-rate-batch-ablation", 610, "图09-5：学习率和批次的单因素对照", "左图只改变Adam学习率，右图只改变批次大小并固定处理样本总数。批次十六执行二百四十次更新，批次六十四执行六十次，差异不能完全归于梯度噪声。", body)

    body = text(30, 42, "学习率调度：先按更新编号定义，再看真实日志", 28, 'font-weight="700"')
    body += text(30, 78, "常数0.01；或20步线性warmup，再余弦下降到0.001。", 21)
    left, top, width, height = 95, 130, 710, 280
    sx = lambda step: left + step / 239 * width
    sy = lambda rate: top + height - rate / .011 * height
    for tick in [0.0, .005, .01]:
        body += line(left, sy(tick), left + width, sy(tick))
        body += text(left - 15, sy(tick) + 7, f"{tick:g}", 21, 'text-anchor="end"')
    body += line(left, top, left, top + height, INK, 2) + line(left, top + height, left + width, top + height, INK, 2)
    for tick in [0, 19, 100, 200, 239]:
        body += text(sx(tick), top + height + 31, str(tick), 21, 'text-anchor="middle"')
    for key, color, dash in [("adamw_decay", BLUE, 'stroke-dasharray="9 4"'), ("adamw_warmup_cosine", ORANGE, "")]:
        points = [(sx(row["step"] - 1), sy(row["lr"])) for row in experiments[key]["history"][1:]]
        body += path(points, color, 3, dash)
    body += text(40, 115, "ηₖ", 23) + text(810, 480, "更新编号 k（从0开始）", 22, 'text-anchor="end"')
    body += text(30, 529, "蓝虚线：常数；橙实线：warmup + cosine。第19步达到0.01。", 22)
    body += text(30, 570, "两者实际终点效果在正文比较；调度不是保证改进的开关。", 22)
    save(folder, "learning-rate-schedule", 605, "图09-3：逐步记录的常数与预热余弦学习率", "两种调度的学习率都从实际训练日志提取。更新零到十九线性升到零点零一，随后余弦下降，最后一次更新编号二百三十九时为零点零零一。", body)


if __name__ == "__main__":
    source = Path("build/experiments/09-training-and-optimizers/results.json")
    write_figures(json.loads(source.read_text(encoding="utf-8")), Path(__file__).resolve().parents[2] / "docs/assets/09-training-and-optimizers")
