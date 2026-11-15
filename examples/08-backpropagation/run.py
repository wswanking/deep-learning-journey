"""Chapter 08: manual reverse mode, finite differences, autograd and graph lifecycle."""
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
from xml.sax.saxutils import escape

START=time.perf_counter();tracemalloc.start()
import numpy as np
import torch
IMPORT_SECONDS=time.perf_counter()-START
NAMES=("W1","b1","W2","b2")
DTYPE=torch.float64


def plain(v):
    if isinstance(v,torch.Tensor): return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [plain(x) for x in v]
    return v


def t(v,grad=False):
    return torch.tensor(v,dtype=DTYPE,device="cpu",requires_grad=grad)


def numpy_forward(parameters,X,y,activation="tanh"):
    W1,b1,W2,b2=[parameters[k] for k in NAMES]
    z=X@W1.T+b1
    h=np.tanh(z) if activation=="tanh" else np.maximum(z,0)
    output=h@W2.T+b2
    residual=output-y
    loss=np.mean(np.sum(residual**2,axis=1))/2
    return loss,{"z":z,"h":h,"output":output,"residual":residual}


def numpy_backward(parameters,X,y,activation="tanh"):
    loss,cache=numpy_forward(parameters,X,y,activation)
    delta2=cache["residual"]/len(X)
    dh=delta2@parameters["W2"]
    derivative=1-cache["h"]**2 if activation=="tanh" else (cache["z"]>0)
    delta1=dh*derivative
    gradients={"W1":delta1.T@X,"b1":delta1.sum(axis=0),
               "W2":delta2.T@cache["h"],"b2":delta2.sum(axis=0)}
    return loss,gradients,{**cache,"delta2":delta2,"dh":dh,"delta1":delta1,
                           "input_gradient":delta1@parameters["W1"]}


def torch_forward(parameters,X,y,activation="tanh"):
    z=X@parameters["W1"].T+parameters["b1"]
    h=torch.tanh(z) if activation=="tanh" else torch.relu(z)
    output=h@parameters["W2"].T+parameters["b2"]
    # Mean over samples only; sum over output components before averaging.
    loss=((output-y)**2).sum(dim=1).mean()/2
    return loss,output


def finite_difference(parameters,X,y,epsilon,activation="tanh"):
    gradients={name:np.zeros_like(value) for name,value in parameters.items()}
    for name,value in parameters.items():
        for index in np.ndindex(value.shape):
            positive={key:v.copy() for key,v in parameters.items()}
            negative={key:v.copy() for key,v in parameters.items()}
            positive[name][index]+=epsilon;negative[name][index]-=epsilon
            gradients[name][index]=(numpy_forward(positive,X,y,activation)[0]-
                                     numpy_forward(negative,X,y,activation)[0])/(2*epsilon)
    return gradients


def error(left,right):
    return max(float(np.max(np.abs(left[name]-right[name]))) for name in NAMES)


def plot(figure_dir,scan,history):
    figure_dir.mkdir(parents=True,exist_ok=True)
    def text(x,y,s,size=20,anchor="start"):
        return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="#183044">{escape(str(s))}</text>'
    def line(x1,y1,x2,y2,color="#d6e1e9",dash=""):
        return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="2"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
    def path(points,color,dash=""):
        return '<path d="'+" ".join(f"{'M' if i==0 else 'L'}{x:.2f},{y:.2f}" for i,(x,y) in enumerate(points))+'" fill="none" stroke="'+color+'" stroke-width="3"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
    body=text(40,44,"图08-4  梯度核对与更新：有限差分有扰动误差，手写与AD轨迹相合",27)
    # Use log10 axes for both finite-difference perturbation and error.
    left,top,w,h=100,140,440,340
    yvalues=[math.log10(max(row["max_absolute_error"],1e-18)) for row in scan]
    low=math.floor(min(yvalues))-1;high=math.ceil(max(yvalues))+1
    def point1(x,y): return left+(math.log10(x)+8)/6*w,top+h-(math.log10(max(y,1e-18))-low)/(high-low)*h
    body+=text(left,97,"全26个参数的中心差分",23)+text(left,126,"最大绝对梯度误差（log₁₀坐标）",18)
    for exponent in range(-8,-1):
        px,py=point1(10.**exponent,10.**low)
        body+=line(px,top,px,top+h)+text(px,top+h+27,f"10^{exponent}",15,"middle")
    for exponent in range(low,high+1,2):
        px,py=point1(1e-8,10.**exponent)
        body+=line(left,py,left+w,py)+text(left-10,py+5,f"10^{exponent}",15,"end")
    body+=path([point1(row["epsilon"],row["max_absolute_error"]) for row in scan],"#176aa0")
    body+=text(left+w/2,top+h+64,"差分扰动 ε（log₁₀坐标）",19,"middle")
    left2=690;ymax=max(row["manual_loss"] for row in history)*1.06
    def point2(step,loss): return left2+step/200*410,top+h-loss/ymax*h
    body+=text(left2,97,"同初始化的两种更新实现",23)+text(left2,126,"训练半平方损失（仅对样本平均）",18)
    for step in [0,50,100,150,200]:
        px,py=point2(step,0);body+=line(px,top,px,top+h)+text(px,top+h+27,step,16,"middle")
    for val in np.linspace(0,ymax,5):
        px,py=point2(0,val);body+=line(left2,py,left2+410,py)+text(left2-10,py+5,f"{val:.3f}",15,"end")
    body+=path([point2(row["step"],row["manual_loss"]) for row in history],"#176aa0")
    body+=path([point2(row["step"],row["autograd_loss"]) for row in history],"#ad5015","9 5")
    body+=text(left2+205,top+h+64,"完整批次梯度更新步",19,"middle")
    body+=line(100,610,155,610,"#176aa0")+text(170,617,"手写反向传播：实线",21)
    body+=line(620,610,675,610,"#ad5015","9 5")+text(690,617,"自动微分：虚线（与实线重合）",21)
    body+=text(100,672,"左图为固定平滑tanh网络的真实误差扫描；右图仅用于核对梯度更新，不评估泛化。",21)
    body+=text(100,713,"全部元素为矢量；固定CPU float64、种子8、学习率0.1，实际报告保存每个检查点。",21)
    figure_dir.joinpath("04-gradient-experiment.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 755" role="img" aria-labelledby="title desc">'
        '<title id="title">梯度有限差分误差与实际训练轨迹</title><desc id="desc">本章实际CPU运行数据绘制，未使用位图。</desc>'
        '<rect width="1200" height="755" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,Arial,sans-serif">'+body+'</g></svg>\n',
        encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("build/experiments/08-backpropagation"))
    parser.add_argument("--figure-dir",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(8)
    checks=[]
    def record(name,**values):
        checks.append({"name":name,"passed":True,**plain(values)});print("PASS",name)

    # A rational ReLU hand example, with all preactivations away from its kink.
    p={"W1":np.array([[1.,-1.],[.5,.5]]),"b1":np.array([0.,0.]),
       "W2":np.array([[2.,-1.]]),"b2":np.array([.5])}
    X,y=np.array([[1.,2.]]),np.array([[1.]])
    before,g,cache=numpy_backward(p,X,y,"relu")
    expected={"W1":np.array([[0.,0.],[2.,4.]]),"b1":np.array([0.,2.]),
              "W2":np.array([[0.,-3.]]),"b2":np.array([-2.])}
    assert before==2 and np.array_equal(cache["output"],[[-1.]])
    assert error(g,expected)==0
    tp={name:t(value,True) for name,value in p.items()}
    loss,_=torch_forward(tp,t(X),t(y),"relu")
    ag=torch.autograd.grad(loss,tuple(tp.values()))
    ag={name:v.detach().numpy() for name,v in zip(NAMES,ag)}
    assert error(g,ag)==0
    fd=finite_difference(p,X,y,1e-5,"relu")
    assert error(g,fd)<1e-8
    updated={name:p[name]-.1*g[name] for name in NAMES}
    after,after_cache=numpy_forward(updated,X,y,"relu")
    assert math.isclose(after,.13005,rel_tol=1e-12)
    record("hand_relu_forward_backward_update",before_loss=before,gradients=g,
           input_gradient=cache["input_gradient"],after_parameters=updated,
           after_output=after_cache["output"],after_loss=after,
           finite_difference_error=error(g,fd))

    # Shared variable: contributions from both consumers must be added.
    w=t(1,True);x=t(2,True);a=w*x;s=a+w;L=s*s/2
    gw,gx=torch.autograd.grad(L,(w,x))
    assert gw.item()==9 and gx.item()==3
    record("shared_parameter_path_accumulation",loss=L,grad_w=gw,grad_x=gx)

    # Smooth non-square 3 -> 4 -> 2 network: all 26 parameters are checked.
    rng=np.random.default_rng(8)
    smooth={"W1":rng.normal(0,.3,(4,3)),"b1":rng.normal(0,.1,4),
            "W2":rng.normal(0,.3,(2,4)),"b2":rng.normal(0,.1,2)}
    sx,sy=rng.normal(size=(5,3)),rng.normal(size=(5,2))
    sl,sg,_=numpy_backward(smooth,sx,sy)
    sp={name:t(v,True) for name,v in smooth.items()}
    tl,_=torch_forward(sp,t(sx),t(sy))
    autograd={name:v.detach().numpy() for name,v in zip(NAMES,torch.autograd.grad(tl,tuple(sp.values())))}
    np.testing.assert_allclose(sl,tl.item(),rtol=1e-12,atol=1e-12)
    assert error(sg,autograd)<1e-12
    scan=[]
    for epsilon in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
        fd=finite_difference(smooth,sx,sy,epsilon)
        scan.append({"epsilon":epsilon,"max_absolute_error":error(sg,fd)})
    assert scan[3]["max_absolute_error"]<1e-7
    record("all_26_parameters_manual_autograd_finite_difference",
           manual_autograd_max_error=error(sg,autograd),finite_difference_scan=scan,
           gradient_shapes={name:value.shape for name,value in sg.items()})

    # Broadcasting bias gradients must sum across the batch axis.
    bx=t([[1,2],[3,4]]);bias=t([0,0],True)
    objective=((bx+bias)**2).sum()/4
    gb,=torch.autograd.grad(objective,(bias,))
    torch.testing.assert_close(gb,t([2,3]),rtol=0,atol=0)
    record("broadcast_reverse_sums_shared_axis",bias_gradient=gb)

    # backward accumulates into .grad across freshly rebuilt graphs.
    leaf=t(3,True);(leaf**2).backward();first=leaf.grad.clone()
    (leaf**2).backward();second=leaf.grad.clone()
    leaf.grad=None;(leaf**2).backward();cleared=leaf.grad.clone()
    assert first.item()==6 and second.item()==12 and cleared.item()==6
    record("backward_gradient_buffer_accumulates",first=first,second=second,after_clear=cleared)

    leaf=t(3,True);returned,=torch.autograd.grad(leaf**2,(leaf,))
    assert returned.item()==6 and leaf.grad is None
    record("autograd_grad_returns_without_populating_leaf_buffer",returned=returned,leaf_grad=leaf.grad)

    # This square saves data for backward; a second traversal normally raises.
    leaf=t(3,True);graph=leaf**2;graph.backward()
    try: graph.backward()
    except RuntimeError as e: released=type(e).__name__
    else: raise AssertionError("Expected second-backward graph-release error")
    record("saved_graph_reuse_is_rejected",observed_error=released)

    leaf=t(3,True);graph=leaf**2;graph.backward(retain_graph=True);graph.backward()
    assert leaf.grad.item()==12
    record("intentional_retained_graph_and_buffer_sum",grad=leaf.grad)

    w=t(3,True);a=t(2,True);h=w*2
    blocked,ga=torch.autograd.grad(h.detach()*a,(w,a),allow_unused=True)
    assert blocked is None and ga.item()==6
    record("detach_cuts_derivative_path",weight_gradient=blocked,downstream_gradient=ga)

    leaf=t(2,True)
    with torch.no_grad(): evaluation=leaf**2
    assert not evaluation.requires_grad
    record("no_grad_skips_recording",output=evaluation,requires_grad=evaluation.requires_grad)

    leaf=t([2.],True);saved=leaf*leaf
    with torch.no_grad(): leaf.add_(1)
    try: saved.sum().backward()
    except RuntimeError as e: modified=type(e).__name__
    else: raise AssertionError("Expected modified-saved-tensor error")
    record("inplace_modification_of_saved_value_is_rejected",observed_error=modified)

    vx=t([2.,3.],True);vf=torch.stack((vx[0]*vx[1],vx[0]+vx[1]))
    vjp,=torch.autograd.grad(vf,(vx,),grad_outputs=t([1.,2.]))
    torch.testing.assert_close(vjp,t([5,4]),rtol=0,atol=0)
    record("vector_jacobian_product",output=vf,cotangent=[1,2],input_gradient=vjp)

    branch=[]
    for value in [-2.,2.]:
        leaf=t(value,True);out=leaf**2 if leaf.item()>0 else 3*leaf
        branch_grad,=torch.autograd.grad(out,(leaf,))
        branch.append({"input":value,"output":float(out.detach()),"gradient":float(branch_grad)})
    assert branch[0]["gradient"]==3 and branch[1]["gradient"]==4
    record("dynamic_execution_follows_actual_branch",examples=branch)

    # Compare entire update trajectories on training-only synthetic data.
    tx=rng.normal(size=(32,2));ty=(.7*np.tanh(1.2*tx[:,0]-.5*tx[:,1])+.2)[:,None]
    parameters={"W1":rng.normal(0,.3,(3,2)),"b1":np.zeros(3),
                "W2":rng.normal(0,.3,(1,3)),"b2":np.zeros(1)}
    tp={name:t(v,True) for name,v in parameters.items()}
    history=[]
    for step in range(201):
        manual_loss,manual_gradients,_=numpy_backward(parameters,tx,ty)
        auto_loss,_=torch_forward(tp,t(tx),t(ty))
        if step%10==0:
            divergence=max(np.max(np.abs(parameters[name]-tp[name].detach().numpy())) for name in NAMES)
            assert divergence<1e-10
            history.append({"step":step,"manual_loss":manual_loss,"autograd_loss":float(auto_loss.detach()),
                            "max_parameter_divergence":float(divergence)})
        if step==200: break
        for parameter in tp.values(): parameter.grad=None
        auto_loss.backward()
        for name in NAMES: parameters[name]-=.1*manual_gradients[name]
        with torch.no_grad():
            for parameter in tp.values(): parameter-=.1*parameter.grad
    assert history[-1]["manual_loss"] < history[0]["manual_loss"]/10
    record("manual_and_autograd_training_trajectories_agree",initial_loss=history[0]["manual_loss"],
           final_loss=history[-1]["manual_loss"],
           max_parameter_divergence=max(row["max_parameter_divergence"] for row in history))

    args.output_dir.mkdir(parents=True,exist_ok=True)
    figure_dir=args.figure_dir or args.output_dir/"figures";plot(figure_dir,scan,history)
    _,peak=tracemalloc.get_traced_memory()
    report={"chapter":"08","status":"passed","timestamp_local":datetime.now().astimezone().isoformat(timespec="seconds"),
            "environment":{"python":platform.python_version(),"executable":sys.executable,"platform":platform.platform(),
                           "numpy":np.__version__,"pytorch":torch.__version__,"device":"cpu","dtype":"float64",
                           "torch_threads":torch.get_num_threads(),"seed":8},
            "config":{"loss":"sum over output components, mean over samples, multiply by 1/2",
                      "finite_difference_epsilon":1e-5,"finite_difference_scan":scan,
                      "training":{"n":32,"shape":[2,3,1],"parameters":13,"steps":200,"learning_rate":.1,
                                  "optimizer":"full-batch gradient descent","data":"synthetic deterministic generator",
                                  "split":"training-only numerical comparison; no generalization claim"},
                      "hand_network":{"shape":[2,2,1],"parameters":9,"activation":"ReLU","learning_rate":.1}},
            "checks":checks,"training_history":history,"summary":{"passed":len(checks),"failed":0},
            "resources":{"import_seconds":IMPORT_SECONDS,"elapsed_seconds_including_imports":time.perf_counter()-START,
                         "python_traced_peak_bytes":peak,"memory_scope":"Python tracemalloc only, excludes full native tensor memory/process RSS"},
            "figure_directory":str(figure_dir)}
    (args.output_dir/"results.json").write_text(json.dumps(plain(report),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"checks":len(checks),"initial_loss":history[0]["manual_loss"],
                      "final_loss":history[-1]["manual_loss"],"manual_autograd_gradient_error":error(sg,autograd),
                      "epsilon_1e_minus_5_error":scan[3]["max_absolute_error"]},ensure_ascii=False))
    print("Report:",(args.output_dir/"results.json").resolve())


if __name__=="__main__": main()
