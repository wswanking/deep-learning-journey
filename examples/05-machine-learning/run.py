"""Chapter 05: reproducible CPU linear baselines with train/validation/test isolation."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import tracemalloc
from xml.sax.saxutils import escape

START = time.perf_counter()
tracemalloc.start()
import numpy as np
import torch
import torch.nn.functional as F
IMPORT_TIME = time.perf_counter() - START

SEED = 20261115
STEPS = 300
INTERVAL = 10
LEARNING_RATES = [0.05, 0.2]
DTYPE = torch.float64
COLORS = ["#176aa0", "#ad5015", "#087c63"]
INK = "#183044"


def plain(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [plain(item) for item in value]
    return value


def tensor(value, *, labels=False):
    return torch.tensor(value, dtype=torch.long if labels else DTYPE, device="cpu")


def make_datasets():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-2, 2, size=(240, 1))
    regression = (x, 2.5 * x[:, 0] - 0.4 + rng.normal(0, 0.15, size=240))
    x = rng.normal(size=(300, 2))
    score = 1.7 * x[:, 0] - 1.2 * x[:, 1] + 0.2
    probability = 1 / (1 + np.exp(-score))
    binary = (x, (rng.random(300) < probability).astype(np.float64))
    centers = np.array([[-1.8, -1.2], [1.8, -1.2], [0, 1.8]], dtype=np.float64)
    labels = np.repeat(np.arange(3), 100)
    x = centers[labels] + rng.normal(0, 0.65, size=(300, 2))
    multiclass = (x, labels)
    return {"regression": regression, "binary": binary, "multiclass": multiclass}


def split_indices(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    return {"train": order[:n*3//5], "validation": order[n*3//5:n*4//5], "test": order[n*4//5:]}


def prepare(raw_x, raw_y, indices, task):
    # No holdout values are used to compute feature statistics.
    train_x = raw_x[indices["train"]]
    mean, std = train_x.mean(axis=0), train_x.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1, std)
    result = {
        split: (tensor((raw_x[ids] - mean) / std),
                tensor(raw_y[ids], labels=task == "multiclass"))
        for split, ids in indices.items()
    }
    return result, mean, std


def loss_and_gradient(task, X, y, W, b):
    logits = X @ W.T + b
    n = len(X)
    if task == "regression":
        assert logits[:, 0].shape == y.shape
        residual = logits[:, 0] - y
        loss = (residual ** 2).mean()
        E = 2 * residual[:, None]
    elif task == "binary":
        assert logits[:, 0].shape == y.shape
        loss = F.binary_cross_entropy_with_logits(logits[:, 0], y, reduction="mean")
        E = (torch.sigmoid(logits[:, 0]) - y)[:, None]
    else:
        loss = F.cross_entropy(logits, y, reduction="mean")
        E = torch.softmax(logits, dim=1) - F.one_hot(y, num_classes=W.shape[0]).to(DTYPE)
    return loss, E.T @ X / n, E.mean(dim=0)


def evaluate(task, X, y, W, b):
    loss, _, _ = loss_and_gradient(task, X, y, W, b)
    out = {"loss": float(loss)}
    logits = X @ W.T + b
    if task == "regression":
        out["rmse"] = math.sqrt(out["loss"])
    elif task == "binary":
        out["accuracy"] = float(((logits[:, 0] >= 0).to(DTYPE) == y).to(DTYPE).mean())
    else:
        out["accuracy"] = float((logits.argmax(dim=1) == y).to(DTYPE).mean())
    return out


def baseline(task, train_y, y):
    # Every baseline is fitted to training targets only.
    if task == "regression":
        value = train_y.mean()
        mse = float(((y - value) ** 2).mean())
        return {"loss": mse, "rmse": math.sqrt(mse), "prediction": float(value)}
    classes = 2 if task == "binary" else 3
    counts = torch.bincount(train_y.to(torch.long), minlength=classes).to(DTYPE)
    probabilities = counts / counts.sum()
    major = int(probabilities.argmax())
    loss = float(-torch.log(probabilities[y.to(torch.long)]).mean())
    accuracy = float((y == major).to(DTYPE).mean())
    return {"loss": loss, "accuracy": accuracy, "majority_class": major,
            "training_class_probabilities": plain(probabilities)}


def train_candidate(task, train, validation, outputs, lr):
    X, y = train
    W, b = torch.zeros((outputs, X.shape[1]), dtype=DTYPE), torch.zeros(outputs, dtype=DTYPE)
    history, best = [], None
    for step in range(STEPS + 1):
        if step % INTERVAL == 0:
            training = evaluate(task, X, y, W, b)
            held_out = evaluate(task, *validation, W, b)
            history.append({"step": step, "train": training, "validation": held_out})
            if best is None or held_out["loss"] < best["validation_loss"]:
                best = {"step": step, "validation_loss": held_out["loss"], "W": W.clone(), "b": b.clone()}
        if step == STEPS: break
        loss, grad_w, grad_b = loss_and_gradient(task, X, y, W, b)
        assert torch.isfinite(loss) and torch.isfinite(grad_w).all()
        W -= lr * grad_w
        b -= lr * grad_b
    return {"learning_rate": lr, "history": history, **best}


def verify_math(checks):
    # Independent NumPy gradients and PyTorch autograd both check the derived updates.
    X = tensor([[1, -1], [0.5, 2], [-2, 1]])
    for task, y, rows in [
        ("regression", tensor([1, -2, 0.5]), 1),
        ("binary", tensor([1, 0, 1]), 1),
        ("multiclass", tensor([0, 2, 1], labels=True), 3),
    ]:
        W = torch.arange(rows * 2, dtype=DTYPE).reshape(rows, 2) / 10
        b = torch.arange(rows, dtype=DTYPE) / 20
        loss, gw, gb = loss_and_gradient(task, X, y, W, b)
        aw, ab = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
        auto_loss = loss_and_gradient(task, X, y, aw, ab)[0]
        auto_gw, auto_gb = torch.autograd.grad(auto_loss, (aw, ab))
        torch.testing.assert_close(gw, auto_gw, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(gb, auto_gb, rtol=1e-10, atol=1e-12)
        xn, yn, zn = X.numpy(), y.numpy(), (X @ W.T + b).numpy()
        if task == "regression":
            en = 2 * (zn[:, 0] - yn)[:, None]
            ln = np.mean((zn[:, 0] - yn) ** 2)
        elif task == "binary":
            en = (1 / (1 + np.exp(-zn[:, 0])) - yn)[:, None]
            ln = np.mean(np.logaddexp(0, zn[:, 0]) - yn * zn[:, 0])
        else:
            shifted = zn - zn.max(axis=1, keepdims=True)
            pn = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            en = pn - np.eye(rows)[yn]
            ln = np.mean(np.logaddexp.reduce(zn, axis=1) - zn[np.arange(len(yn)), yn])
        np.testing.assert_allclose(gw.numpy(), en.T @ xn / len(xn), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(gb.numpy(), en.mean(axis=0), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(float(loss), ln, rtol=1e-10, atol=1e-12)
        checks.append({"name": task + "_manual_numpy_autograd", "passed": True})

    # Hand calculations from the chapter, plus extreme-logit numerical stability.
    mse = float(((tensor([0, 1]) - tensor([1, 3])) ** 2).mean())
    bce = float(F.binary_cross_entropy_with_logits(tensor([math.log(3)]), tensor([1])))
    ce = float(F.cross_entropy(tensor([[math.log(2), 0, 0]]), tensor([0], labels=True)))
    assert math.isclose(mse, 2.5) and math.isclose(bce, -math.log(0.75))
    assert math.isclose(ce, math.log(2))
    assert torch.isfinite(F.binary_cross_entropy_with_logits(tensor([1000, -1000]), tensor([0, 1])))
    assert torch.isfinite(F.cross_entropy(tensor([[1000, 0, -1000]]), tensor([2], labels=True)))
    checks.append({"name": "hand_examples_and_stability", "passed": True,
                   "mse": mse, "bce": bce, "ce": ce})


# The plotting helpers write only SVG paths, shapes and text; no bitmap elements.
def text(x, y, value, size=19, color=INK, anchor="start"):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>'


def line(x1, y1, x2, y2, color="#bdccd5", width=1.5, dash=""):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>'


def svg(path, title, height, body):
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{escape(title)}</title><desc id="desc">本章固定种子CPU实验的实际数据和选定模型，线条、散点与文字均为矢量。</desc>'
        f'<rect width="1200" height="{height}" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif">{body}</g></svg>\n',
        encoding="utf-8",
    )


def axes(left, top, width, height, xlim, ylim, xlabel, ylabel):
    def point(x, y):
        return left + (x-xlim[0])/(xlim[1]-xlim[0])*width, top + height - (y-ylim[0])/(ylim[1]-ylim[0])*height
    body = ""
    for x in np.linspace(*xlim, 5):
        px, _ = point(x, ylim[0])
        body += line(px, top, px, top+height, "#e0e7ec", 1)
        body += text(px, top+height+26, f"{x:.1f}", 15, anchor="middle")
    for y in np.linspace(*ylim, 5):
        _, py = point(xlim[0], y)
        body += line(left, py, left+width, py, "#e0e7ec", 1)
        body += text(left-10, py+5, f"{y:.2g}", 15, anchor="end")
    body += line(left, top+height, left+width, top+height, INK, 2)
    body += line(left, top, left, top+height, INK, 2)
    body += text(left+width/2, top+height+57, xlabel, 19, anchor="middle")
    body += text(left, top-14, ylabel, 19)
    return body, point


def curve(points, color, dash=""):
    coords = " ".join(f"{'M' if i == 0 else 'L'}{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points))
    return f'<path d="{coords}" fill="none" stroke="{color}" stroke-width="3"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>'


def marker(x, y, color, kind, size=4):
    if kind == "train":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size}" fill="{color}" fill-opacity=".6"/>'
    if kind == "validation":
        return f'<rect x="{x-size:.2f}" y="{y-size:.2f}" width="{2*size}" height="{2*size}" fill="white" stroke="{color}" stroke-width="1.5"/>'
    return line(x-size, y-size, x+size, y+size, color, 1.5) + line(x-size, y+size, x+size, y-size, color, 1.5)


def plot_results(figure_dir, raw, models):
    figure_dir.mkdir(parents=True, exist_ok=True)
    # Regression scatter and fitted line; all displays occur after model selection.
    result = models["regression"]
    X, y = raw["regression"]
    body = text(40, 42, "图05-3  回归拟合：展示方案冻结后的全部样本", 27)
    graph, point = axes(100, 110, 1000, 430, (-2.2, 2.2), (-6.3, 5.5), "原始输入 x（无量纲）", "目标 y（无量纲）")
    body += graph
    for split, ids in result["indices"].items():
        for i in ids:
            body += marker(*point(X[i, 0], y[i]), "#54748a", split, 3.4)
    grid = np.linspace(-2.2, 2.2, 101)
    original_w = float(result["W"][0,0]) / float(result["std"][0])
    original_b = float(result["b"][0]) - original_w * float(result["mean"][0])
    body += curve([point(x, original_w*x+original_b) for x in grid], COLORS[0])
    body += line(100, 620, 155, 620, COLORS[0], 3) + text(170, 627, "已冻结的选定模型", 20)
    for x, split, label in [(490,"train","训练：圆点"),(710,"validation","验证：空方"),(930,"test","测试：叉号")]:
        body += marker(x,620,"#54748a",split,5) + text(x+15,627,label,19)
    body += text(100, 676, f"拟合 y = {original_w:.4f} x {original_b:+.4f}；数据生成均值 y = 2.5x − 0.4，噪声标准差0.15。", 20)
    svg(figure_dir/"03-regression-fit.svg", "合成回归数据与实际拟合", 720, body)

    # Binary/multiclass decisions in raw feature coordinates; points are held-out test examples.
    body = text(40,42,"图05-4  分类边界：输出概率，边界由线性得分决定",27)
    for col, task, label in [(0,"binary","二分类：z=0处为直线边界"),(1,"multiclass","三分类：比较三个线性得分")]:
        model = models[task]
        panel_x = 80 + col*590
        body += text(panel_x, 92, label, 23)
        graph, point = axes(panel_x, 140, 450, 350, (-3.5, 3.5), (-3.5, 3.5), "原始特征 x₁", "原始特征 x₂")
        # A coarse vector background identifies predictions; it is NOT a raster heatmap.
        cell = 7/24
        W, b = model["W"].numpy(), model["b"].numpy()
        for xi in np.arange(-3.5, 3.5, cell):
            for yi in np.arange(-3.5, 3.5, cell):
                normalized = (np.array([xi+cell/2, yi+cell/2])-model["mean"])/model["std"]
                logits = W@normalized + b
                pred = int(logits[0] >= 0) if task=="binary" else int(logits.argmax())
                px, py = point(xi, yi+cell)
                body += f'<rect x="{px:.2f}" y="{py:.2f}" width="{450/24+.1:.2f}" height="{350/24+.1:.2f}" fill="{COLORS[pred]}" fill-opacity=".09"/>'
        body += graph
        # Exact pairwise equal-score lines; for multiclass, show only where the two beat the third.
        for first in range(len(W)):
            for second in range(first+1, len(W)) if task=="multiclass" else [-1]:
                w = W[first] if second == -1 else W[first]-W[second]
                offset = b[first] if second == -1 else b[first]-b[second]
                wr = w/model["std"]
                br = offset - np.dot(wr,model["mean"])
                segments=[]
                for x in np.linspace(-3.5,3.5,501):
                    if abs(wr[1]) < 1e-12: continue
                    yy=-(wr[0]*x+br)/wr[1]
                    if not (-3.5<=yy<=3.5): continue
                    logits=W@((np.array([x,yy])-model["mean"])/model["std"])+b
                    if task=="multiclass" and logits[first] < logits.max()-1e-9: continue
                    segments.append(point(x,yy))
                if len(segments)>1: body+=curve(segments,INK)
        xx, yy = raw[task]
        for i in model["indices"]["test"]:
            cls = int(yy[i])
            px,py = point(xx[i,0],xx[i,1])
            if cls==0: body+=marker(px,py,COLORS[cls],"train",4.5)
            elif cls==1: body+=marker(px,py,COLORS[cls],"validation",4.5)
            else: body+=f'<path d="M{px:.2f},{py-5:.2f} L{px-5:.2f},{py+4:.2f} L{px+5:.2f},{py+4:.2f} Z" fill="white" stroke="{COLORS[cls]}" stroke-width="1.5"/>'
        body += text(panel_x, 583, "仅显示冻结模型后的测试点；形状表示真实类别。",17)
    body += text(80,640,"类别0：蓝色圆点；类别1：橙色空方；类别2：绿色三角。浅色区域表示模型预测。",21)
    body += text(80,681,"背景由SVG矩形组成；黑线是实际模型的得分相等处，不是数据真实边界。",20)
    svg(figure_dir/"04-classification-boundaries.svg","二分类和三分类的实际边界",725,body)

    body = text(40,42,"图05-5  学习曲线：训练集更新参数，验证集选择检查点",27)
    for col,(task,title) in enumerate([("regression","回归：MSE"),("binary","二分类：BCE"),("multiclass","三分类：CE")]):
        model=models[task]; history=model["history"]; left=70+col*380
        maximum=max(row[s]["loss"] for row in history for s in ["train","validation"])*1.06
        body+=text(left,96,title,23)
        graph,point=axes(left,145,325,300,(0,STEPS),(0,maximum),"梯度更新步","平均损失")
        body+=graph
        for split,color,dash in [("train",COLORS[0],""),("validation",COLORS[1],"8 5")]:
            body+=curve([point(row["step"],row[split]["loss"]) for row in history],color,dash)
        px,_=point(model["step"],0)
        body+=line(px,145,px,445,COLORS[2],1.8,"3 5")
        body+=text(left,554,f"lr={model['learning_rate']}，选中第{model['step']}步",20)
    body+=line(80,610,135,610,COLORS[0],3)+text(150,617,"训练：实线",21)
    body+=line(440,610,495,610,COLORS[1],3,"8 5")+text(510,617,"验证：虚线",21)
    body+=line(840,610,895,610,COLORS[2],2,"3 5")+text(910,617,"选中检查点",21)
    body+=text(80,668,"曲线记录同一候选从第0到300步；最终参数恢复到验证损失最低的已记录步。",21)
    body+=text(80,708,"三图纵轴量纲和范围不同，不能跨任务用曲线高度排名；测试集不参与曲线。",20)
    svg(figure_dir/"05-learning-curves.svg","实际训练与验证曲线及选定检查点",750,body)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("build/experiments/05-machine-learning"))
    parser.add_argument("--figure-dir",type=Path,help="Optional output directory for the three result SVGs")
    args=parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(SEED)
    checks=[]; verify_math(checks)
    datasets=make_datasets(); models={}; reports={}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for task_number,(task,(X,y)) in enumerate(datasets.items()):
        indices=split_indices(len(X),SEED+task_number+1)
        all_ids=np.concatenate(list(indices.values()))
        assert len(set(all_ids.tolist()))==len(X) and set(all_ids.tolist())==set(range(len(X)))
        checks.append({"name":task+"_split_is_disjoint_and_complete","passed":True})
        splits,mean,std=prepare(X,y,indices,task)
        np.testing.assert_allclose(mean,X[indices["train"]].mean(axis=0),rtol=0,atol=0)
        np.testing.assert_allclose(splits["train"][0].mean(dim=0).numpy(),0,atol=1e-14)
        checks.append({"name":task+"_train_only_standardization","passed":True})
        outputs=3 if task=="multiclass" else 1
        # Candidate trainers receive no test inputs or targets.
        candidates=[train_candidate(task,splits["train"],splits["validation"],outputs,lr) for lr in LEARNING_RATES]
        chosen=min(candidates,key=lambda run:run["validation_loss"])
        W,b=chosen["W"],chosen["b"]
        selected_validation=evaluate(task,*splits["validation"],W,b)
        base_validation=baseline(task,splits["train"][1],splits["validation"][1])
        assert selected_validation["loss"] < base_validation["loss"]
        checks.append({"name":task+"_validation_beats_training_fitted_constant","passed":True})
        # Parameters and preprocessing are frozen BEFORE accessing test metrics.
        frozen={"task":task,"W":plain(W),"b":plain(b),"mean":plain(mean),"std":plain(std),
                "learning_rate":chosen["learning_rate"],"step":chosen["step"]}
        checkpoint=args.output_dir/(task+"-model.json")
        checkpoint.write_text(json.dumps(frozen,indent=2)+"\n",encoding="utf-8")
        recovered=json.loads(checkpoint.read_text(encoding="utf-8"))
        rw,rb=tensor(recovered["W"]),tensor(recovered["b"])
        torch.testing.assert_close(splits["validation"][0]@W.T+b,splits["validation"][0]@rw.T+rb,rtol=0,atol=0)
        checks.append({"name":task+"_json_model_round_trip","passed":True})
        metrics={split:evaluate(task,*pair,W,b) for split,pair in splits.items()}
        baseline_metrics={split:baseline(task,splits["train"][1],pair[1]) for split,pair in splits.items()}
        serial_candidates=[{"learning_rate":run["learning_rate"],"selected_step":run["step"],
                            "selected_validation_loss":run["validation_loss"],"history":run["history"]}
                           for run in candidates]
        reports[task]={"n":len(X),"input_features":X.shape[1],"output_features":outputs,
                       "parameters":outputs*(X.shape[1]+1),"split_ids":plain(indices),
                       "split_sizes":{name:len(ids) for name,ids in indices.items()},
                       "data_sha256":hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),
                       "preprocessing":{"fit_split":"train","mean":plain(mean),"std_ddof0":plain(std)},
                       "selected":frozen,"metrics":metrics,"constant_baseline":baseline_metrics,
                       "candidates":serial_candidates,"test_policy":"evaluated only after validation selection and model freeze"}
        if task=="regression":
            raw_w=plain(W[0]/tensor(std));raw_b=float(b[0]-(W[0]*tensor(mean/std)).sum())
            reports[task]["original_unit_coefficients"]={"w":raw_w,"b":raw_b}
        models[task]={**chosen,"indices":indices,"mean":mean,"std":std}
        print(task, json.dumps({"selected_lr":chosen["learning_rate"],"selected_step":chosen["step"],
                               "metrics":metrics},ensure_ascii=False))
    # A transparent failure case: a linear logit cannot separate XOR.
    xor_x=tensor([[0,0],[0,1],[1,0],[1,1]])
    xor_y=tensor([0,1,1,0]);W=torch.zeros((1,2),dtype=DTYPE);b=torch.zeros(1,dtype=DTYPE)
    for step in range(200):
        _,gw,gb=loss_and_gradient("binary",xor_x,xor_y,W,b)
        W-=0.2*gw;b-=0.2*gb
    xor_metrics=evaluate("binary",xor_x,xor_y,W,b)
    assert math.isclose(xor_metrics["loss"],math.log(2),abs_tol=1e-12)
    assert xor_metrics["accuracy"]==0.5
    checks.append({"name":"xor_linear_failure_is_reproduced","passed":True})
    # A small example demonstrating contaminated preprocessing statistics.
    train=np.array([0.,2.]);holdouts=np.array([100.,102.])
    leakage={"train_mean":float(train.mean()),"incorrect_all_data_mean":float(np.concatenate([train,holdouts]).mean()),
             "interpretation":"Holdout information changes preprocessing; this does not assert every leakage inflates every metric."}
    assert leakage["train_mean"]==1 and leakage["incorrect_all_data_mean"]==51
    checks.append({"name":"leakage_statistics_counterexample","passed":True})
    figure_dir=args.figure_dir or args.output_dir/"figures"
    plot_results(figure_dir,datasets,models)
    current,peak=tracemalloc.get_traced_memory()
    report={"chapter":"05","status":"passed","timestamp_local":datetime.now().astimezone().isoformat(timespec="seconds"),
            "environment":{"python":platform.python_version(),"executable":sys.executable,"platform":platform.platform(),
                           "numpy":np.__version__,"pytorch":torch.__version__,"device":"cpu","dtype":"float64",
                           "torch_threads":torch.get_num_threads(),"seed":SEED},
            "config":{"steps_per_candidate":STEPS,"record_interval":INTERVAL,"learning_rates":LEARNING_RATES,
                      "optimizer":"full-batch gradient descent using explicit derived gradients","initialization":"all zeros",
                      "selection":"minimum validation loss over recorded steps and both learning rates",
                      "regularization":"none","data":"project-generated synthetic arrays; no external downloads"},
            "tasks":reports,"failure_case":{"xor":xor_metrics},"leakage_example":leakage,
            "checks":checks,"summary":{"passed":len(checks),"failed":0},
            "resources":{"import_seconds":IMPORT_TIME,"elapsed_seconds_including_imports":time.perf_counter()-START,
                         "python_traced_peak_bytes":peak,
                         "memory_scope":"Python tracemalloc only, excludes full native tensor memory and process RSS"},
            "figure_directory":str(figure_dir)}
    (args.output_dir/"results.json").write_text(json.dumps(plain(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"{len(checks)} checks passed. Report: {(args.output_dir/'results.json').resolve()}")


if __name__=="__main__":
    main()
