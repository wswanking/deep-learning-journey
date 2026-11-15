"""Chapter 11: small, reproducible CPU convolution and CNN/MLP experiment."""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, time, tracemalloc
from pathlib import Path

START = time.perf_counter()
tracemalloc.start()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
IMPORT_SECONDS = time.perf_counter() - START
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
torch.manual_seed(11)

def corr_numpy(x, w, b, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """NCHW, OIHW; groups=1; symmetric zero padding; literal formula."""
    n, ci, h, width = x.shape
    co, wi, kh, kw = w.shape
    assert ci == wi
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    assert oh > 0 and ow > 0
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.empty((n, co, oh, ow), dtype=np.float64)
    for a in range(n):
        for o in range(co):
            for i in range(oh):
                for j in range(ow):
                    patch = xp[a, :, i * sh:i * sh + dh * (kh - 1) + 1:dh,
                               j * sw:j * sw + dw * (kw - 1) + 1:dw]
                    out[a, o, i, j] = np.sum(patch * w[o]) + b[o]
    return out

def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def make_data(n, seed, shift=0):
    """Self-generated task: horizontal=0, vertical=1; no external dataset."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(2), n // 2)
    rng.shuffle(y)
    x = rng.normal(0, .08, (n, 1, 16, 16))
    positions = rng.integers(1, 6, (n, 2))
    amplitudes = rng.uniform(.8, 1.2, n)
    for a in range(n):
        i, j = positions[a] + shift
        if y[a] == 0:
            x[a, 0, i, j:j + 5] += amplitudes[a]
        else:
            x[a, 0, i:i + 5, j] += amplitudes[a]
    return torch.from_numpy(x), torch.from_numpy(y), positions.tolist()

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.head = nn.Linear(8, 2)
    def forward(self, x):
        h = F.max_pool2d(F.relu(self.conv1(x)), 2)
        h = F.relu(self.conv2(h))
        h = h.mean(dim=(2, 3))
        return self.head(h)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

def metric(model, pair):
    model.eval()
    with torch.no_grad():
        logits = model(pair[0])
        return {"cross_entropy": F.cross_entropy(logits, pair[1]).item(),
                "accuracy": (logits.argmax(1) == pair[1]).double().mean().item()}

def train_model(cls, train, validation, seed):
    candidates = []
    for lr in (.01, .03):
        torch.manual_seed(seed)
        model = cls()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        history = []
        best = None
        for step in range(201):
            if step % 10 == 0:
                tr = metric(model, train)
                va = metric(model, validation)
                history.append({"step": step, "train": tr, "validation": va})
                if best is None or va["cross_entropy"] < best["validation_loss"]:
                    best = {"step": step, "validation_loss": va["cross_entropy"],
                            "state": copy.deepcopy(model.state_dict())}
            if step == 200:
                break
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train[0]), train[1])
            loss.backward()
            opt.step()
        candidates.append({"lr": lr, "best": best, "history": history})
    chosen = min(candidates, key=lambda c: c["best"]["validation_loss"])
    model = cls()
    model.load_state_dict(chosen["best"]["state"])
    return model, chosen, [{"lr": c["lr"], "best_step": c["best"]["step"],
                           "validation_loss": c["best"]["validation_loss"]}
                          for c in candidates]

def figures(folder, report, data):
    """Editable SVG paths/text/rectangles only; no bitmap embeddings."""
    folder.mkdir(parents=True, exist_ok=True)
    def text(x, y, s, size=16, color="#26384c"):
        return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{s}</text>'
    def svg(name, title, body, h=520):
        out = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{h}" viewBox="0 0 1080 {h}" role="img">'
               f'<title>{title}</title><rect width="1080" height="{h}" fill="white"/>'
               '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">' + body + '</g></svg>')
        (folder / name).write_text(out, encoding="utf-8")
    body = text(28, 32, "图11-4  自制数据与位置迁移：标签由线的方向决定", 22)
    for panel, (pair, label) in enumerate([(data["test"], "IID测试：左上起点"), (data["shift"], "配对迁移：起点各轴 +6")]):
        x, y = pair
        for cls in (0, 1):
            idx = int(torch.nonzero(y == cls)[0].item())
            xx = x[idx, 0].numpy()
            ox = 80 + panel * 520 + cls * 220
            oy = 85; cell = 11
            body += text(ox, 65, label if cls == 0 else ("横线 / 0" if cls == 0 else "竖线 / 1"), 15)
            for i in range(16):
                for j in range(16):
                    val = max(0, min(1, float(xx[i, j])))
                    gray = round(255 - 190 * val)
                    color = f"rgb({gray},{gray},{gray})"
                    body += f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell}" height="{cell}" fill="{color}" stroke="#dce4ec" stroke-width=".3"/>'
            body += text(ox, 285, f"标签 {cls}，16 × 16，单通道", 14)
    body += text(40, 340, "训练/验证/IID测试：192/64/64；迁移测试与IID测试共享噪声、标签和振幅，仅移动图案。", 17)
    body += text(40, 380, "迁移集预先定义，只在选定模型后报告；它代表位置分布改变，不能冒充同分布测试。", 17)
    body += text(40, 420, "图由run.py中固定种子数据直接绘制；每个像素是可缩放SVG矩形，非嵌入位图。", 16)
    svg("04-data-and-shift.svg", "自制横竖线数据与配对位置迁移", body, 460)
    body = text(28, 32, "图11-5  相同数据与预算：验证损失曲线和冻结模型测试指标", 22)
    colors = {"cnn": "#1475b8", "mlp": "#c65b12"}
    ox, oy, width, height = 80, 75, 560, 300
    body += f'<path d="M{ox},{oy}v{height}h{width}" fill="none" stroke="#56677b"/>'
    for v in (0, .2, .4, .6, .8):
        yy = oy + height * (1 - v / .8)
        body += f'<path d="M{ox},{yy}h{width}" stroke="#dde5ee"/>'
        body += text(35, yy + 5, f"{v:.1f}", 13)
    for step in (0, 50, 100, 150, 200):
        xx = ox + width * step / 200
        body += text(xx - 8, oy + height + 24, str(step), 13)
    for name, color in colors.items():
        hist = report["models"][name]["history"]
        pts = " ".join(f"{ox+width*r['step']/200:.2f},{oy+height*(1-min(.8,r['validation']['cross_entropy'])/.8):.2f}" for r in hist)
        body += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>'
        body += text(690, 105 if name == "cnn" else 245, f"{name.upper()}：{report['models'][name]['parameters']} 个参数", 20, color)
        m = report["models"][name]
        yy = 140 if name == "cnn" else 280
        body += text(690, yy, f"IID准确率 {100*m['test']['accuracy']:.1f}%", 18, color)
        body += text(690, yy+30, f"迁移准确率 {100*m['shift']['accuracy']:.1f}%", 18, color)
        body += text(690, yy+60, f"选中 lr={m['learning_rate']}，步 {m['selected_step']}", 16, color)
    body += text(80, 445, "横轴：完整训练批次的参数更新次数；纵轴：按样本平均的验证交叉熵（自然对数）。", 16)
    body += text(80, 480, "两条曲线分别来自各自选中的学习率；训练用Adam，曲线非固定随机初始化之间的公平因果实验。", 15)
    svg("05-training-and-results.svg", "CNN与MLP的真实验证曲线与位置迁移结果", body)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build/experiments/11-convolutional-networks"))
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    checks = []
    def check(name, passed, **details):
        if not passed:
            raise AssertionError((name, details))
        checks.append({"name": name, "passed": True, **details})
    x = np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3)
    w = np.array([[1., 2.], [0., -1.]]).reshape(1, 1, 2, 2)
    result = corr_numpy(x, w, np.zeros(1))
    check("hand_cross_correlation", np.array_equal(result[0, 0], [[0, 2], [6, 8]]), output=result.tolist())
    conv = corr_numpy(x, w[:, :, ::-1, ::-1], np.zeros(1))
    check("kernel_flip_changes_operation", np.array_equal(conv[0, 0], [[12, 14], [18, 20]]), flipped_output=conv.tolist())
    tx = torch.from_numpy(x.copy()).requires_grad_()
    tw = torch.from_numpy(w.copy()).requires_grad_()
    tb = torch.zeros(1, requires_grad=True)
    out = F.conv2d(tx, tw, tb)
    check("numpy_torch_match", np.array_equal(out.detach().numpy(), result))
    out.sum().backward()
    check("shared_kernel_gradient_sums_positions", np.array_equal(tw.grad.numpy()[0,0], [[12,16],[24,28]]) and tb.grad.item()==4, kernel_gradient=tw.grad.tolist(), bias_gradient=tb.grad.item())
    check("input_gradient_overlapping_windows", np.array_equal(tx.grad.numpy()[0,0], [[1,3,2],[1,2,1],[0,-1,-1]]), input_gradient=tx.grad.tolist())
    rng = np.random.default_rng(11)
    rx = rng.normal(size=(2, 3, 7, 8))
    rw = rng.normal(size=(4, 3, 3, 2)); rb = rng.normal(size=4)
    expected = corr_numpy(rx, rw, rb, stride=(2, 3), padding=(1, 0), dilation=(2, 1))
    actual = F.conv2d(torch.from_numpy(rx), torch.from_numpy(rw), torch.from_numpy(rb), stride=(2,3), padding=(1,0), dilation=(2,1)).numpy()
    err = float(np.max(np.abs(expected-actual)))
    check("multi_channel_stride_padding_dilation", expected.shape==(2,4,3,3) and err<1e-12, shape=list(expected.shape), max_error=err)
    pairx = np.concatenate([x, 2*x], axis=1)
    pairw = np.concatenate([w, .5*w], axis=1)
    multi = corr_numpy(pairx, pairw, np.zeros(1))
    check("channel_sum_not_concat", np.array_equal(multi, 2*result), output=multi.tolist())
    # Stride1 valid convolution commutes with translations on the interior.
    z = torch.randn(1,1,7,8); kk = torch.randn(1,1,3,3)
    a = F.conv2d(z,kk); moved = F.conv2d(torch.roll(z,1,dims=3),kk)
    check("interior_translation_equivariance", torch.allclose(moved[:,:,:,1:],a[:,:,:,:-1],atol=1e-12))
    p = torch.tensor([[[[1.,4.,2.],[3.,0.,5.],[2.,6.,1.]]]],requires_grad=True)
    pm = F.max_pool2d(p,2,stride=1)
    pm.sum().backward()
    check("max_pool_and_gradient", np.array_equal(pm.detach().numpy()[0,0],[[4,5],[6,6]]) and p.grad[0,0,2,1].item()==2, output=pm.detach().tolist(), gradient=p.grad.tolist())
    ap = F.avg_pool2d(p.detach(),2,stride=1)
    check("average_pool", np.array_equal(ap.numpy()[0,0],[[2,2.75],[2.75,3]]), output=ap.tolist())
    neg = F.max_pool2d(-torch.ones(1,1,2,2),2,stride=1,padding=1)
    check("max_pool_padding_is_negative_infinity", torch.all(neg==-1).item())
    # Receptive field: kernel3/stride1 -> pool2/stride2 -> kernel3/stride1.
    rf,jump = 1,1
    rf_table=[]
    for kernel,stride in [(3,1),(2,2),(3,1)]:
        rf += (kernel-1)*jump; jump *= stride
        rf_table.append({"kernel":kernel,"stride":stride,"field":rf,"jump":jump})
    check("receptive_field_recurrence", rf==8 and jump==2, stages=rf_table)
    trainx,trainy,pos = make_data(192,20261115)
    valx,valy,vpos = make_data(64,20261116)
    testx,testy,tpos = make_data(64,20261117)
    shiftx,shifty,spos = make_data(64,20261117,shift=6)
    check("paired_shift_protocol", torch.equal(testy,shifty) and tpos==spos and len(set([digest(trainx.numpy()),digest(valx.numpy()),digest(testx.numpy())]))==3)
    pairs={"train":(trainx,trainy),"validation":(valx,valy),"test":(testx,testy),"shift":(shiftx,shifty)}
    report={"seed":20261115,"dtype":"float64","device":"cpu","threads":1,
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},
            "checks":checks,"models":{},"protocol":{"sizes":{k:len(v[1]) for k,v in pairs.items()},
             "data":"self_generated_16x16_horizontal_vertical_line_length5_noise_sd0.08",
             "train_validation_test_seeds":[20261115,20261116,20261117],
             "source_top_left_start_inclusive":[1,5],"shift_pixels_both_axes":6,
             "loss":"mean_sample_cross_entropy_natural_log",
             "search_learning_rates":[.01,.03],"steps":200,"validation_every":10,
             "optimizer":"Adam default betas=(0.9,0.999), eps=1e-8, weight_decay=0",
             "preprocessing":"none; raw synthetic pixels",
             "test_selection":"none; all model choices frozen before evaluating either test set",
             "sha256":{k:{"x":digest(v[0].numpy()),"y":digest(v[1].numpy())} for k,v in pairs.items()}}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,cls in [("cnn",TinyCNN),("mlp",TinyMLP)]:
        begin=time.perf_counter()
        model,chosen,candidates=train_model(cls,pairs["train"],pairs["validation"],20261118)
        checkpoint=args.output_dir/f"{name}.pt"
        torch.save(model.state_dict(),checkpoint)
        restored=cls()
        restored.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
        with torch.no_grad():
            restore_error=(model(testx)-restored(testx)).abs().max().item()
        check(name+"_checkpoint_exact_restore",restore_error==0, max_error=restore_error)
        report["models"][name]={"parameters":sum(p.numel() for p in model.parameters()),
           "learning_rate":chosen["lr"],"selected_step":chosen["best"]["step"],
           "candidates":candidates,"history":chosen["history"],
           "train":metric(model,pairs["train"]),"validation":metric(model,pairs["validation"]),
           "test":metric(model,pairs["test"]),"shift":metric(model,pairs["shift"]),
           "training_seconds":time.perf_counter()-begin}
    check("parameter_count",report["models"]["cnn"]["parameters"]==354 and report["models"]["mlp"]["parameters"]==4146)
    report["elapsed_seconds"]=time.perf_counter()-START
    report["import_seconds"]=IMPORT_SECONDS
    report["python_traced_peak_bytes"]=tracemalloc.get_traced_memory()[1]
    report["memory_scope"]="Python-tracked allocations only; excludes much native tensor memory; not process RSS"
    (args.output_dir/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    figures(args.figure_dir or args.output_dir/"figures",report,pairs)
    print(json.dumps({"passed_checks":len(checks),"elapsed_seconds":report["elapsed_seconds"],
       "import_seconds":IMPORT_SECONDS,"models":{k:{s:v[s] for s in ["parameters","learning_rate","selected_step","test","shift","training_seconds"]} for k,v in report["models"].items()}},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
