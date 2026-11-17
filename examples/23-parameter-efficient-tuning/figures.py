"""Native SVG diagrams and report-driven plots; no raster elements."""
from pathlib import Path
import html
import math

C = {"ink": "#24374c", "blue": "#1475b8", "light": "#eaf4fc",
     "orange": "#bc570e", "warm": "#fff2e6", "gray": "#edf0f4", "line": "#afc4d6"}


def text(x, y, value, size=18, color=None):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color or C["ink"]}">{html.escape(str(value))}</text>'


def box(x, y, width, height, color=None):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="5" fill="{color or C["light"]}" stroke="{C["line"]}"/>'


def arrow(x, y, xx, yy, color=None):
    return f'<path d="M{x},{y}L{xx},{yy}" fill="none" stroke="{color or C["blue"]}" stroke-width="2" marker-end="url(#arrow)"/>'


def save(folder, filename, title, body, height=620):
    folder.mkdir(parents=True, exist_ok=True)
    source = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}" role="img">
<title>{html.escape(title)}</title><desc>原创教学示意或本机真实报告图；LoRA依据Hu等2021第4节，QLoRA依据Dettmers等2023第2至3节。所有图形与文字均为矢量。</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10Z" fill="{C['blue']}"/></marker></defs>
<rect width="1080" height="{height}" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">{body}</g></svg>'''
    (folder / filename).write_text(source, encoding="utf8")


def write_principles(folder):
    b = text(25, 35, "图23-1  冻结原权重，训练低秩增量；同一个输入走两条支路", 22)
    b += box(40, 200, 125, 75) + text(66, 231, "输入x", 22) + text(62, 260, "k维列向量", 16)
    b += arrow(175, 235, 220, 235)
    b += '<path d="M220,235V115H290M220,235V355H290" stroke="#1475b8" stroke-width="2" fill="none"/>'
    b += box(300, 70, 300, 95, C["gray"]) + text(330, 110, "W₀：d×k，冻结", 22) + text(330, 145, "基座输出 W₀x", 18)
    b += box(300, 310, 170, 95) + text(320, 350, "A：r×k", 22) + text(320, 385, "压到r维", 18)
    b += arrow(480, 358, 540, 358)
    b += box(550, 310, 170, 95) + text(570, 350, "B：d×r", 22) + text(570, 385, "升到d维", 18)
    b += arrow(730, 358, 805, 358) + text(755, 334, "×s", 18)
    b += '<path d="M610,118H845V230M815,358H845V280" stroke="#1475b8" stroke-width="2" fill="none"/>'
    b += '<circle cx="845" cy="255" r="25" fill="#fff2e6" stroke="#bc570e"/>'
    b += text(833, 264, "+", 27, C["orange"]) + arrow(880, 255, 930, 255) + text(945, 250, "h", 25)
    b += text(80, 460, "h = W₀x + sBAx；s = α/r。只更新A、B，偏置与其他权重在本章也冻结。", 19)
    b += text(80, 505, "手算：Ax=3，BAx=(6,−3)，W₀x=(1,0)，s=1 → h=(7,−3)。", 19)
    b += text(80, 550, "rank(ΔW) ≤ r；这约束增量，不要求W₀本身低秩，也不保证总权重低秩。", 18, C["orange"])
    save(folder, "01-low-rank-branch.svg", "低秩增量的双支路形状与手算", b, 600)

    b = text(25, 35, "图23-2  冻结参数仍能传梯度；no_grad会切断所包住的计算", 22)
    for i, (label, subtitle, color) in enumerate([
        ("前层LoRA", "A₁、B₁可训练", C["light"]),
        ("隐藏状态u", "需要接收梯度", C["light"]),
        ("冻结线性W₂", "不积累参数梯度", C["gray"]),
        ("损失L", "计算目标误差", C["warm"])]):
        b += box(45 + 260*i, 105, 215, 95, color)
        b += text(65 + 260*i, 145, label, 21) + text(65 + 260*i, 180, subtitle, 17)
        if i < 3:
            b += arrow(267 + 260*i, 150, 295 + 260*i, 150)
    b += '<path d="M920,220V255H155V220" stroke="#bc570e" stroke-width="3" fill="none"/>'
    b += text(190, 290, "反向：∂L/∂u = W₂ᵀ g；W₂不更新，但u与更早的LoRA仍需这条梯度。", 19)
    b += box(50, 335, 475, 160) + text(75, 375, "A随机非零，B=0", 22)
    b += text(75, 415, "开始：BA=0，输出与原基座一致。", 18)
    b += text(75, 455, "首步：∇A=0，∇B通常非零。", 18)
    b += box(555, 335, 475, 160, C["warm"]) + text(580, 375, "若A=0，B=0", 22, C["orange"])
    b += text(580, 415, "两因子梯度都为0，支路无法起步。", 18)
    b += text(580, 455, "不要把“零增量”误写成“双零因子”。", 18)
    b += text(60, 550, "requires_grad=False冻结参数；eval()控制训练行为；no_grad()关闭计算记录。三者不同。", 18)
    save(folder, "02-freezing-and-gradients.svg", "冻结与梯度传播、零初始化的区别", b, 600)

    b = text(25, 35, "图23-3  量化是近似表示；存4bit，不等于用4bit完成全部计算", 22)
    b += box(45, 85, 475, 165, C["gray"]) + text(65, 123, "论文QLoRA：冻结量化基座", 21)
    b += text(65, 165, "NF4非均匀码本 + 分块缩放", 19)
    b += text(65, 205, "双重量化缩放元数据；分页优化器应对峰值", 17)
    b += text(65, 234, "解量化到计算dtype，梯度流向浮点LoRA", 17)
    b += box(560, 85, 475, 165, C["warm"]) + text(580, 123, "本章CPU替代：不是NF4 QLoRA", 21, C["orange"])
    b += text(580, 165, "逐行均匀对称−7…7，打包进4bit码", 18)
    b += text(580, 205, "尺度float64；每次前向解量化到float64", 17)
    b += text(580, 234, "保留浮点embedding/norm；没有分页优化器", 17)
    b += text(50, 310, "手算一行 w=[−1, −0.3, 0.2, 0.9]，δ=max|w|/7=1/7。", 20)
    b += text(50, 355, "整数 q=[−7, −2, 1, 6] → 4bit码 q+8=[1,6,9,14]。", 19)
    b += text(50, 400, "每两个码打包：1+16×6=97；9+16×14=233。码0未用。", 19)
    b += text(50, 445, "重建 ŵ=[−1, −2/7, 1/7, 6/7]；最大误差2/35≈0.05714。", 19)
    b += text(50, 490, "未饱和舍入误差≤δ/2；基座输出、梯度传播也会受到量化误差影响。", 18)
    b += text(50, 550, "4权重：原float64为32B；本例打包2B+尺度8B=10B，仍非纯粹“8倍缩小”。", 18, C["orange"])
    save(folder, "03-quantization-contract.svg", "NF4 QLoRA概念与明确区分的CPU均匀量化手算", b, 600)


def write_figures(report, folder):
    folder = Path(folder)
    write_principles(folder)
    trained = report["runs"][2:]
    colors = [C["blue"], C["orange"], "#327858"]
    names = ["full FT", "LoRA", "uniform-int4+LoRA"]
    b = text(25, 35, "图23-5  同数据/步数/批次，按验证回答NLL选择；训练曲线不含测试", 22)
    all_values = [p[k] for run in trained for p in run["history"] for k in ["train_nll", "validation_nll"]]
    top = math.ceil(max(all_values))
    for panel, key in enumerate(["train_nll", "validation_nll"]):
        left = 90 + panel * 510
        b += text(left, 85, ["训练回答平均NLL", "验证回答平均NLL"][panel], 20)
        for tick in range(5):
            value = top*tick/4
            y = 420 - 285*value/top
            b += f'<path d="M{left},{y}H{left+400}" stroke="#e0e7ef"/>'
            b += text(left-40, y+5, f"{value:g}", 14)
        for ri, run in enumerate(trained):
            points = " ".join(f'{left+400*p["step"]/report["protocol"]["updates"]},{420-285*p[key]/top}' for p in run["history"])
            b += f'<polyline points="{points}" fill="none" stroke="{colors[ri]}" stroke-width="2.5"/>'
        for tick in range(5):
            value = report["protocol"]["updates"]*tick/4
            b += text(left+400*tick/4-10, 450, f"{value:g}", 14)
        b += text(left+150, 485, "更新步数", 17)
    for i, name in enumerate(names):
        b += text(80+320*i, 535, name, 18, colors[i])
    b += text(60, 580, "float64 CPU，单种子；统一学习率不代表每种方法都经过充分调参。", 17)
    save(folder, "05-validation-curves.svg", "全参数、LoRA及均匀量化替代的真实训练验证曲线", b)

    b = text(25, 35, "图23-4  实际持久张量账本：不含激活、解量化临时权重与进程RSS", 22)
    fields = ["frozen_parameter_bytes", "frozen_buffer_bytes", "trainable_parameter_bytes",
              "gradient_bytes", "optimizer_tensor_bytes"]
    legends = ["冻结浮点参数", "冻结量化/其他buffer", "可训练参数", "参数梯度", "Adam状态/步数"]
    palette = ["#b5cce4", "#a7c8b7", "#1475b8", "#bc570e", "#9763a8"]
    totals = [sum(run["tensor_ledger"][k] for k in fields) for run in trained]
    maximum = max(totals)
    for i, run in enumerate(trained):
        y = 115+i*100
        b += text(40, y+24, names[i], 18)
        cursor = 285
        for k, col in zip(fields, palette):
            width = 650*run["tensor_ledger"][k]/maximum
            b += f'<rect x="{cursor}" y="{y}" width="{width}" height="40" fill="{col}"/>'
            cursor += width
        b += text(cursor+10, y+26, f"{totals[i]/1024:.2f} KiB", 16)
        b += text(285, y+67, f'可训练 {run["tensor_ledger"]["trainable_parameters"]} 参数；检查点 {run["checkpoint_file_bytes"]} B', 16)
    for i, (name, col) in enumerate(zip(legends, palette)):
        x = 45+(i%3)*340
        y = 460+(i//3)*45
        b += f'<rect x="{x}" y="{y-17}" width="20" height="20" fill="{col}"/>'
        b += text(x+30, y, name, 17)
    b += text(45, 575, "仅张量字节，不是内存峰值；int4每次前向解量化会额外分配float64矩阵。", 18, C["orange"])
    save(folder, "04-tensor-ledger.svg", "本机实际持久张量字节及适配器体积", b)

    b = text(25, 35, "图23-6  冻结后测试：回答字节NLL与贪心完整回答正确率须分别看", 22)
    labels = ["冻结基座", "仅均匀int4", "full FT", "LoRA", "均匀int4+LoRA"]
    b += text(45, 90, "方法", 19) + text(350, 90, "测试回答平均NLL", 19)
    b += text(650, 90, "完整回答/24", 19) + text(865, 90, "格式有效/24", 19)
    for i, run in enumerate(report["runs"]):
        y = 140+65*i
        b += f'<path d="M40,{y+20}H1030" stroke="#e0e7ef"/>'
        b += text(45, y, labels[i], 19)
        b += text(370, y, f'{run["metrics"]["test"]["mean_nll"]:.6f}', 19)
        b += text(700, y, f'{round(run["test_generation"]["exact_match"]*24)} / 24', 19)
        b += text(910, y, f'{round(run["test_generation"]["format_valid"]*24)} / 24', 19)
    b += text(45, 505, "3个监督byte/例：两字母回答 + LF；无序字母对按组隔离，测试仅24例。", 18)
    b += text(45, 550, "LF为本任务停止约定；低teacher-forced损失不保证生成时复制/反转正确。", 18, C["orange"])
    b += text(45, 590, "同一基座与数据；量化会改变初始函数；单种子小样本不能推出普遍性能排序。", 17)
    save(folder, "06-generation-results.svg", "低秩微调真实测试损失和生成成功失败", b, 640)
