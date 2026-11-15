"""Chapter 20: objective/data/causality checks and count-model language baselines on CPU."""
from __future__ import annotations
import argparse,hashlib,json,math,platform,sys,time,tracemalloc
from pathlib import Path
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf8")
START=time.perf_counter();tracemalloc.start()
import numpy as np
import torch
from torch.nn import functional as F
from corpus import canonical,clean_records,load_splits,chunks,collate
IMPORT_SECONDS=time.perf_counter()-START
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(20)
torch.use_deterministic_algorithms(True)
def byte_pairs(records):
    x=[];y=[]
    for r in records:
        seq=list(r["text"].encode("utf8"));x.extend(seq[:-1]);y.extend(seq[1:])
    return np.array(x,dtype=np.int64),np.array(y,dtype=np.int64)
def fit(records,kind,alpha):
    x,y=byte_pairs(records)
    if kind=="unigram":
        count=np.bincount(y,minlength=256).astype(np.float64)
        prob=(count+alpha)/(len(y)+256*alpha)
    else:
        count=np.zeros((256,256));np.add.at(count,(x,y),1)
        prob=(count+alpha)/(count.sum(1,keepdims=True)+256*alpha)
    return {"kind":kind,"alpha":alpha,"counts":count,"probabilities":prob}
def evaluate(model,records):
    x,y=byte_pairs(records)
    p=model["probabilities"];chosen=p[y] if model["kind"]=="unigram" else p[x,y]
    total=float(-np.log(chosen).sum());n=len(y)
    if not n:raise ValueError("No valid targets")
    mean=total/n
    return {"target_tokens":n,"negative_log_likelihood_sum":total,"nats_per_byte":mean,
            "perplexity_per_byte":math.exp(mean),"bits_per_byte":mean/math.log(2)}
def figures(folder,report):
    folder.mkdir(parents=True,exist_ok=True)
    def t(x,y,s,size=17,color="#24374c"):
        s=str(s).replace("&","&amp;").replace("<","&lt;")
        return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{s}</text>'
    def save(name,title,b,h=520):
        (folder/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{h}" viewBox="0 0 1080 {h}" role="img"><title>{title}</title><rect width="1080" height="{h}" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">{b}</g></svg>',encoding="utf8")
    b=t(25,35,"图20-4  同一256字节词表下：训练/验证/测试的真实每字节困惑度",22)
    vals=[report["models"][name][s]["perplexity_per_byte"] for name in ("unigram","bigram") for s in ("train","validation","test")]
    top=math.ceil(max(vals)/2)*2
    for v in np.linspace(0,top,5):
        yy=370-260*v/top;b+=f'<path d="M90,{yy}H950" stroke="#e0e7ef"/>'+t(40,yy+5,f"{v:g}",14)
    colors=["#b5cce4","#79a8d4","#1475b8"]
    for mi,name in enumerate(("unigram","bigram")):
        for si,s in enumerate(("train","validation","test")):
            v=report["models"][name][s]["perplexity_per_byte"];x=150+mi*420+si*95;hh=260*v/top
            b+=f'<rect x="{x}" y="{370-hh}" width="65" height="{hh}" fill="{colors[si]}"/>'
            b+=t(x,360-hh,f"{v:.3f}",15)
            b+=t(x,397,{"train":"训练","validation":"验证","test":"测试"}[s],16)
        b+=t(190+mi*420,438,name+"，α="+str(report["models"][name]["alpha"]),19)
    b+=t(80,480,"所有计数仅来自训练文档；α只按验证集选择；测试仅报告冻结后的结果。",17)
    save("04-count-model-results.svg","计数语言模型的真实字节困惑度比较",b)
    b=t(25,35,"图20-5  指标为何会虚假变好：独立玩具反例，不是正式模型成绩",22)
    leak=report["diagnostics"]["count_leak"]
    b+=t(60,100,"把测试文本加入计数，等于提前看到答案。",20)
    for i,(label,val,col) in enumerate([("仅训练计数",leak["proper_test_ppl"],"#1475b8"),("违规加入测试计数",leak["leaked_test_ppl"],"#bc570e")]):
        b+=t(60,155+i*90,label,18,col)
        b+=f'<rect x="340" y="{130+i*90}" width="{580*val/leak["proper_test_ppl"]}" height="40" fill="{col}"/>'
        b+=t(350+580*val/leak["proper_test_ppl"],158+i*90,f"{val:.2f}",17,col)
    b+=t(60,340,"无移位的“预测当前输入”可被复制策略作弊：",20)
    b+=t(60,380,f"未移位损失 {report['diagnostics']['copy_current']['unshifted_loss']:.8f}，正确移位 {report['diagnostics']['copy_current']['shifted_loss']:.6f}",18)
    b+=t(60,425,"前者不是更好的语言模型；它求解了已经可见的当前token。",18,"#bc570e")
    b+=t(60,480,"玩具计数使用α=0.1，训练aaaaab、测试xxxxxxy；正式测试数据没有加入训练。",16)
    save("05-leakage-and-shift-diagnostics.svg","计数泄漏与标签未移位的真实反例",b)
def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=Path("build/experiments/20-language-model-pretraining"));p.add_argument("--figure-dir",type=Path);args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checks=[]
    def check(name,passed,**details):
        if not passed:raise AssertionError((name,details))
        checks.append({"name":name,"passed":True,**details})
    splits,meta=load_splits()
    check("clean_deduplicate_and_drop",meta["raw_records"]==40 and meta["clean_records"]==36 and len(meta["duplicates"])==2 and len(meta["dropped"])==2)
    gs={s:set(meta["split_groups"][s]) for s in splits}
    hs={s:{r["sha256"] for r in rows} for s,rows in splits.items()}
    check("group_and_hash_split_disjoint",all(not(gs[a]&gs[b]) and not(hs[a]&hs[b]) for a,b in [("train","validation"),("train","test"),("validation","test")]))
    bad=False
    try:clean_records([{"id":"a","group_id":"g0","text":"same"},{"id":"b","group_id":"g1","text":"same"}])
    except ValueError:bad=True
    check("conflicting_duplicate_group_rejected",bad)
    raw=[json.loads(s) for s in (Path(__file__).parent/"data/raw.jsonl").read_text(encoding="utf8").splitlines()]
    naive_train={hashlib.sha256(canonical(r["text"]).encode()).hexdigest() for r in raw[:24]}
    naive_test={hashlib.sha256(canonical(r["text"]).encode()).hexdigest() for r in raw[30:]}
    overlap=len(naive_train&naive_test)
    check("naive_row_split_leaks_duplicates",overlap==2,overlapping_hashes=overlap)
    example=[{"id":"toy","group_id":"toy","text":"ABCD"}]
    part=chunks(example,2)
    check("shift_and_chunk_no_target_duplication",[(r["input_ids"],r["labels"]) for r in part]==[([65,66],[66,67]),([67],[68])])
    docs=[{"id":"d0","group_id":"g0","text":"AB"},{"id":"d1","group_id":"g1","text":"CD"}]
    dx,dy=byte_pairs(docs)
    check("no_cross_document_transition",list(zip(dx,dy))==[(65,66),(67,68)])
    nul_parts=chunks([{"id":"nul","group_id":"nul","text":"A\0B"}],4)
    nul_batch=collate(nul_parts,4)
    check("real_nul_is_valid_not_padding",nul_batch["input_ids"].tolist()==[[65,0,0,0]] and nul_batch["labels"].tolist()==[[0,66,-100,-100]] and nul_batch["attention_mask"].tolist()==[[True,True,False,False]])
    all_parts=chunks(splits["train"],64);batch=collate(all_parts,64)
    n=int((batch["labels"]!=-100).sum())
    expected=sum(len(r["text"].encode())-1 for r in splits["train"])
    check("valid_target_count",n==expected and torch.equal(batch["attention_mask"],batch["labels"]!=-100),targets=n)
    logits=torch.randn(len(all_parts),64,256,requires_grad=True)
    flat=logits.detach().numpy();labels=batch["labels"].numpy();mask=labels!=-100
    selected=flat[mask];targets=labels[mask];stable=selected-selected.max(1,keepdims=True)
    np_sum=float((np.log(np.exp(stable).sum(1))-stable[np.arange(n),targets]).sum())
    loss=F.cross_entropy(logits.transpose(1,2),batch["labels"],ignore_index=-100,reduction="sum")/n
    error=abs(loss.item()-np_sum/n);loss.backward()
    check("masked_cross_entropy_numpy_torch",error<1e-12,max_error=error)
    check("padding_gradient_zero",bool((logits.grad[~batch["attention_mask"]]==0).all()))
    copy=torch.full((3,256),0.)
    for i,c in enumerate([65,66,67]):copy[i,c]=20
    good=F.cross_entropy(copy,torch.tensor([66,67,68])).item();badloss=F.cross_entropy(copy,torch.tensor([65,66,67])).item()
    check("unshifted_copy_cheat",good>19.9 and badloss<1e-6,unshifted_loss=badloss,shifted_loss=good)
    mean=-np.log([.5,.25,.125]).mean()
    check("hand_perplexity",abs(math.exp(mean)-4)<1e-12,nll_mean=float(mean),ppl=math.exp(mean))
    correct=math.exp((math.log(2)+3*math.log(8))/4)
    check("token_weighted_not_batch_weighted",abs(correct-math.sqrt(32))<1e-12 and abs(correct-4)>1,correct=correct,mean_batch_loss_ppl=4,mean_batch_ppl=5)
    check("uniform_not_ppl_upper_bound",math.exp(-math.log(.01))>2,uniform_binary_ppl=2,wrong_confident_binary_ppl=100)
    model_path=Path(__file__).resolve().parents[1]/"21-tiny-language-model"
    sys.path.insert(0,str(model_path))
    from model import TinyDecoderLM,TinyLMConfig
    probe=TinyDecoderLM(TinyLMConfig(d_model=16,n_heads=2,n_layers=1,d_ff=32,context_length=64)).eval()
    a=torch.tensor([[65,66,67,68,69]]);b=torch.tensor([[65,66,67,120,121]])
    with torch.no_grad():oa=probe(a);ob=probe(b)
    causal_error=(oa[:,:3]-ob[:,:3]).abs().max().item()
    check("shared_decoder_future_causality",causal_error<1e-12,max_prefix_error=causal_error)
    padded=F.pad(a,(0,3));valid=torch.tensor([[True]*5+[False]*3])
    with torch.no_grad():op=probe(padded,attention_mask=valid)
    paderror=(op[:,:5]-oa).abs().max().item()
    check("shared_decoder_right_padding",paderror<1e-12,max_prefix_error=paderror)
    results={}
    for kind in ("unigram","bigram"):
        candidates=[]
        for alpha in (.01,.1,1.):
            fitted=fit(splits["train"],kind,alpha)
            candidates.append((evaluate(fitted,splits["validation"])["nats_per_byte"],fitted))
        score,selected=min(candidates,key=lambda z:z[0])
        before=hashlib.sha256(selected["counts"].tobytes()).hexdigest()
        stats={s:evaluate(selected,rows) for s,rows in splits.items()}
        check(kind+"_evaluation_does_not_fit_test",hashlib.sha256(selected["counts"].tobytes()).hexdigest()==before)
        np.savez(args.output_dir/(kind+".npz"),counts=selected["counts"],probabilities=selected["probabilities"],alpha=selected["alpha"])
        with np.load(args.output_dir/(kind+".npz")) as restored:
            check(kind+"_probability_restore",np.array_equal(restored["probabilities"],selected["probabilities"]))
        results[kind]={"alpha":selected["alpha"],"validation_search":[{"alpha":m["alpha"],"nats_per_byte":v} for v,m in candidates],
            **stats,"independent_probabilities":255 if kind=="unigram" else 256*255}
    toytr=[{"text":"aaaaab"}];toyte=[{"text":"xxxxxxy"}]
    proper=evaluate(fit(toytr,"bigram",.1),toyte)["perplexity_per_byte"]
    leaked=evaluate(fit(toytr+toyte,"bigram",.1),toyte)["perplexity_per_byte"]
    check("toy_test_count_leak_reduces_reported_loss",leaked<proper,proper_ppl=proper,leaked_ppl=leaked)
    for name,rows in splits.items():
        (args.output_dir/(name+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
    (args.output_dir/"corpus-manifest.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf8")
    report={"versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},"seed":20,"dtype":"float64","device":"cpu","threads":1,"checks":checks,"corpus":meta,
       "context_length":64,"models":results,"probe":{"shared_api":"examples/21-tiny-language-model/model.py","trained":False,"parameters":sum(p.numel() for p in probe.parameters()),"purpose":"causal and right-padding contract only, not a language-model score"},
       "diagnostics":{"naive_overlap_hashes":overlap,"copy_current":{"unshifted_loss":badloss,"shifted_loss":good},"count_leak":{"proper_test_ppl":proper,"leaked_test_ppl":leaked}},
       "elapsed_seconds":time.perf_counter()-START,"import_seconds":IMPORT_SECONDS,"python_traced_peak_bytes":tracemalloc.get_traced_memory()[1],"memory_scope":"Python-tracked allocations; excludes much native tensor memory; not process RSS"}
    (args.output_dir/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    figures(args.figure_dir or args.output_dir/"figures",report)
    print(json.dumps({"checks_passed":len(checks),"seconds":report["elapsed_seconds"],"corpus":meta,"models":results,"probe":report["probe"],"diagnostics":report["diagnostics"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
