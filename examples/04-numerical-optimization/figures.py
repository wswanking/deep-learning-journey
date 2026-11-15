"""Pure-vector SVG figures generated from chapter 04 analytic experiments."""
from __future__ import annotations
import html
import math
from pathlib import Path

INK, MUTED, BLUE, TEAL, ORANGE, RED, GRID = "#17344e", "#506b80", "#245ec5", "#007d75", "#ad5a10", "#b32a3d", "#d1dee7"
COLORS = [ORANGE, TEAL, BLUE, RED]


def text(x, y, label, size=24, color=INK, anchor="start"):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{html.escape(str(label))}</text>'


def line(x1, y1, x2, y2, color=GRID, width=2, dash=""):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>'


def arrow(x1, y1, x2, y2, color=INK):
    a = math.atan2(y2 - y1, x2 - x1)
    ax, ay = x2 - 11 * math.cos(a - math.pi / 6), y2 - 11 * math.sin(a - math.pi / 6)
    bx, by = x2 - 11 * math.cos(a + math.pi / 6), y2 - 11 * math.sin(a + math.pi / 6)
    return line(x1, y1, x2, y2, color, 3) + f'<path d="M {ax:.2f} {ay:.2f} L {x2:.2f} {y2:.2f} L {bx:.2f} {by:.2f}" fill="none" stroke="{color}" stroke-width="3"/>'


def rect(x, y, width, height, fill="#eff5fa"):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="{fill}" stroke="{GRID}" stroke-width="2"/>'


def curve(points, color, width=3, dash=""):
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>'


def save(directory, name, title, description, width, height, body):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc><rect width="{width}" height="{height}" fill="white"/><g font-family="Microsoft YaHei, Segoe UI, sans-serif">{body}</g></svg>\n'
    (directory / name).write_text(svg, encoding="utf-8")


def write_figures(report, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    body = text(40, 48, "图04-1  浮点间距随尺度变化：3位二进制有效数字的玩具格式", 28)
    for y, values, step, label in [(180, [1, 1.25, 1.5, 1.75], 0.25, "指数0"), (350, [2, 2.5, 3, 3.5], 0.5, "指数1")]:
        body += text(50, y + 8, label, 24) + arrow(160, y, 750, y)
        positions = [190 + (v - values[0]) * 320 for v in values]
        for x, value in zip(positions, values):
            body += line(x, y - 12, x, y + 12, BLUE, 3) + text(x, y + 48, f"{value:g}", 24, INK, "middle")
        body += line(positions[0], y - 45, positions[1], y - 45, TEAL, 3)
        body += text((positions[0] + positions[1]) / 2, y - 58, "间距" + str(step), 22, TEAL, "middle")
    body += rect(800, 115, 270, 270)
    body += text(820, 160, "有效位相同", 26) + text(820, 215, "数值范围扩大2倍", 23)
    body += text(820, 270, "相邻间距也扩大2倍", 22) + text(820, 330, "并非等间距实数轴", 23)
    body += text(50, 460, "两行使用相同数值尺度，仅平移起点；玩具格式不等于float32的位数。", 22, MUTED)
    save(directory, "04-01-floating-spacing.svg", "浮点间距与尺度", "三位二进制有效数字的教学格式在1附近间距0.25，在2附近间距0.5；两行采用同一数值尺度。", 1120, 500, body)

    positive = report["softmax_cases"][0]
    p = positive["stable_probability"]
    body = text(40, 45, "图04-2  同一softmax公式，选择可计算的中间值", 29)
    for x in [40, 440, 840]:
        body += rect(x, 100, 320, 125) + rect(x, 335, 320, 125)
    body += text(60, 140, "输入分数z", 25) + text(60, 185, "(1000, 1001, 1002)", 23)
    body += text(460, 140, "直接计算 exp(z)", 25, RED) + text(460, 185, "(∞, ∞, ∞)：上溢", 24, RED)
    body += text(860, 140, "直接相除", 25, RED) + text(860, 185, "∞ / ∞ → NaN", 25, RED)
    body += arrow(365, 160, 430, 160, RED) + arrow(765, 160, 830, 160, RED)
    body += text(60, 375, "减去最大值 m=1002", 24, TEAL) + text(60, 420, "得到 (-2, -1, 0)", 24)
    body += text(460, 375, "指数及求和", 25, TEAL) + text(460, 420, "S ≈ 1.5032，至少含一个1", 21)
    body += text(860, 375, "归一化概率", 25, TEAL) + text(860, 420, "(" + ", ".join(f"{v:.4f}" for v in p) + ")", 22)
    body += arrow(365, 395, 430, 395, TEAL) + arrow(765, 395, 830, 395, TEAL)
    body += text(50, 285, "稳定路径保持同一数学结果，并控制指数输入不大于0。", 25)
    body += text(50, 520, "LSE(z) = m + log S ≈ 1002.4076；log概率直接用 (z−m)−log S。", 24)
    save(directory, "04-02-stable-softmax.svg", "稳定softmax与log-sum-exp流程", "原始大分数直接求指数会溢出；减最大值后最大指数项为1，概率及对数归一化可以稳定计算。", 1200, 560, body)

    cases = report["gradient_descent"]
    body = text(40, 44, "图04-3  同一二次目标，不同学习率的实际下降路径", 29)
    for index, case in enumerate(cases):
        ox, oy = (index % 2) * 550 + 35, (index // 2) * 390 + 90
        xmin, xmax = -0.4, 2.3
        limit = 3.2 if index == 3 else 1.35
        ymin, ymax = -limit, limit
        left, top, width, height = ox + 55, oy + 34, 420, 245
        sx = lambda x: left + (x - xmin) / (xmax - xmin) * width
        sy = lambda y: top + (ymax - y) / (ymax - ymin) * height
        body += text(ox, oy, f"η={case['learning_rate']:g}：" + ["缓慢", "较快收敛", "振荡收敛", "振荡发散"][index], 25, COLORS[index])
        body += line(left, top + height, left + width, top + height, INK) + line(left, top, left, top + height, INK)
        for tick in [0, 1, 2]:
            body += line(sx(tick), top + height - 4, sx(tick), top + height + 4, INK)
            body += text(sx(tick), top + height + 26, tick, 18, MUTED, "middle")
        for tick in ([-2, 0, 2] if index == 3 else [-1, 0, 1]):
            body += line(left - 4, sy(tick), left + 4, sy(tick), INK)
            body += text(left - 10, sy(tick) + 6, tick, 18, MUTED, "end")
        for level in [0.25, 0.5, 1, 2, 4, 8, 16]:
            segments, segment = [], []
            for t in range(241):
                angle = t * 2 * math.pi / 240
                x, y = math.sqrt(2 * level) * math.cos(angle), math.sqrt(level / 2) * math.sin(angle)
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    segment.append((sx(x), sy(y)))
                elif segment:
                    segments.append(segment); segment = []
            if segment:
                segments.append(segment)
            for points in segments:
                if len(points) > 1:
                    body += curve(points, GRID, 1.5)
        points = case["points"][:7] if index == 3 else case["points"]
        body += curve([(sx(p[0]), sy(p[1])) for p in points], COLORS[index], 3)
        for p in points:
            body += f'<circle cx="{sx(p[0]):.2f}" cy="{sy(p[1]):.2f}" r="3" fill="{COLORS[index]}"/>'
        body += f'<circle cx="{sx(0):.2f}" cy="{sy(0):.2f}" r="5" fill="{INK}"/>'
        body += text(left + width - 5, top + height + 33, "θ₁", 23) + text(left - 42, top + 10, "θ₂", 23)
        body += text(ox + 80, oy + 325, "仅画前6步；完整40步见图04-4" if index == 3 else "图中点为实际更新；共40步", 20, MUTED)
    save(directory, "04-03-gradient-paths.svg", "不同学习率的二维下降路径", "目标为theta1平方加4倍theta2平方之和的一半，初始点2,1；学习率0.03缓慢、0.2收敛、0.45振荡收敛、0.55振荡发散，发散面板只画前6步并标明。", 1120, 900, body)

    left, top, width, height = 100, 100, 700, 420
    sx = lambda k: left + k / report["configuration"]["steps"] * width
    sy = lambda value: top + (7 - math.log10(value)) / 15 * height
    body = text(40, 45, "图04-4  实际目标值随更新步数变化（纵轴为对数刻度）", 28)
    for exponent in [-8, -4, 0, 4, 7]:
        y = sy(10.0 ** exponent)
        body += line(left, y, left + width, y) + text(left - 12, y + 7, "10^" + str(exponent), 21, MUTED, "end")
    for k in [0, 10, 20, 30, 40]:
        x = sx(k)
        body += line(x, top, x, top + height) + text(x, top + height + 34, k, 22, MUTED, "middle")
    body += arrow(left, top + height, left + width + 20, top + height) + arrow(left, top + height, left, top - 20)
    body += text(110, 82, "目标J(θ)", 24) + text(610, 585, "更新步k（次）", 24)
    for index, case in enumerate(cases):
        body += curve([(sx(k), sy(v)) for k, v in enumerate(case["values"])], COLORS[index], 3, ["", "10 5", "3 5", "10 4 2 4"][index])
        y = 160 + index * 80
        body += line(840, y, 885, y, COLORS[index], 4, ["", "10 5", "3 5", "10 4 2 4"][index]) + text(900, y + 8, "η=" + str(case["learning_rate"]), 24, COLORS[index])
    body += text(840, 505, "相同初始点、精度与步数", 20, MUTED)
    body += text(50, 630, "来源：本机CPU解析小实验；曲线是实际记录，不是手绘的预计趋势。", 23, MUTED)
    save(directory, "04-04-learning-rates.svg", "四种学习率的实际目标曲线", "同一初始点和40次更新，纵轴按10为底的对数刻度显示目标值，较大学习率0.55在初期下降后快速增大。", 1160, 665, body)
