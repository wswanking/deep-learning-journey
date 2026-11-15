"""Chapter 14: token boundaries, deterministic toy BPE, lookup gradients, small embeddings."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, platform, sys, time, tracemalloc, unicodedata
if hasattr(sys.stdout, 'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
START=time.perf_counter();tracemalloc.start()
import numpy as np
import torch
from torch.nn import functional as F
IMPORT_SECONDS=time.perf_counter()-START
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64);torch.use_deterministic_algorithms(True);torch.manual_seed(14)

def merge_pair(seq,pair,new):
    result=[];i=0
    while i<len(seq):
        if i+1<len(seq) and (seq[i],seq[i+1])==pair:
            result.append(new);i+=2
        else:
            result.append(seq[i]);i+=1
    return result

def learn_bpe(sequences,base_size,steps):
    data=[(list(seq),freq) for seq,freq in sequences]
    rules=[];hist=[sum(len(s)*f for s,f in data)]
    for step in range(steps):
        counts=collections.Counter()
        for seq,freq in data:
            for pair in zip(seq,seq[1:]):
                counts[pair]+=freq
        if not counts:
            break
        pair=min(counts,key=lambda p:(-counts[p],p))
        new=base_size+len(rules)
        before=hist[-1]
        data=[(merge_pair(seq,pair,new),freq) for seq,freq in data]
        after=sum(len(s)*f for s,f in data);hist.append(after)
        rules.append({"left":pair[0],"right":pair[1],"new":new,
                      "pair_count":counts[pair],"tokens_removed":before-after})
    return rules,hist,data

def apply_bpe(seq,rules):
    seq=list(seq)
    for r in rules:
        seq=merge_pair(seq,(r["left"],r["right"]),r["new"])
    return seq

def token_bytes(rules):
    vocab={i:bytes([i]) for i in range(256)}
    for r in rules:
        vocab[r["new"]]=vocab[r["left"]]+vocab[r["right"]]
    return vocab

def save_figures(folder,report):
    folder.mkdir(parents=True,exist_ok=True)
    def t(x,y,s,size=17,color="#26384c"):
        s=str(s).replace("&","&amp;").replace("<","&lt;")
        return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{s}</text>'
    def save(name,title,body,h=520):
        (folder/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{h}" viewBox="0 0 1080 {h}" role="img"><title>{title}</title><rect width="1080" height="{h}" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">{body}</g></svg>',encoding="utf8")
    b=t(25,35,"图14-3  实际byte BPE：训练合并压缩与未见文本切分",23)
    hist=report["byte_bpe"]["weighted_training_tokens"]
    ox,oy,w,h=80,85,540,280
    b+=f'<path d="M{ox},{oy}v{h}h{w}" fill="none" stroke="#52677e"/>'
    maxv=hist[0]
    for frac in (0,.25,.5,.75,1):
        y=oy+h*(1-frac)
        b+=f'<path d="M{ox},{y}h{w}" stroke="#dee7ef"/>'
        b+=t(25,y+5,f"{frac*maxv:.0f}",13)
    pts=" ".join(f"{ox+w*i/(len(hist)-1):.2f},{oy+h*(1-v/maxv):.2f}" for i,v in enumerate(hist))
    b+=f'<polyline points="{pts}" fill="none" stroke="#1475b8" stroke-width="3"/>'
    for i in (0,4,8,12,16):
        b+=t(ox+w*i/16-8,395,str(i),13)
    b+=t(80,430,"横轴：从训练文本学习的合并次数；纵轴：按文档频次加权的token数。",16)
    b+=t(690,100,"固定256个初始字节",20)
    b+=t(690,135,f"+ {len(report['byte_bpe']['merges'])}个合并token",19)
    b+=t(690,185,"测试文本：我爱AI。",19)
    row=next(r for r in report["tokenization"] if r["text"]=="我爱AI。")
    b+=t(690,220,f"码点 {row['codepoints']}，字节 {row['utf8_bytes']}",18)
    b+=t(690,255,f"冻结BPE：{row['byte_bpe_tokens']} tokens",18)
    b+=t(690,305,"未见字“𠮷”：4字节可回退",18)
    b+=t(690,340,"不代表已学会该字语义",17)
    b+=t(80,480,"在测试集重学合并会泄漏分布；本图不使用外部tokenizer或商业模型的分词结果。",16)
    save("03-bpe-experiment.svg","原创byte BPE的真实合并压缩结果",b)
    b=t(25,35,"图14-5  小型embedding训练：损失与6个中心词的余弦相似度",22)
    hist=report["embedding"]["history"];ox,oy,w,h=70,100,400,280
    b+=f'<path d="M{ox},{oy}v{h}h{w}" fill="none" stroke="#52677e"/>'
    for v in (1,1.5,2,2.5):
        y=oy+h*(1-(v-1)/1.6);b+=f'<path d="M{ox},{y}h{w}" stroke="#e1e7ee"/>'+t(25,y+5,str(v),13)
    pts=" ".join(f"{ox+w*r['step']/200:.2f},{oy+h*(1-(r['loss']-1)/1.6):.2f}" for r in hist)
    b+=f'<polyline points="{pts}" fill="none" stroke="#1475b8" stroke-width="3"/>'
    yy=oy+h*(1-(math.log(3)-1)/1.6)
    b+=f'<path d="M{ox},{yy}h{w}" stroke="#bc570e" stroke-dasharray="6 5"/>'
    for step in (0,50,100,150,200):b+=t(ox+w*step/200-6,oy+h+24,str(step),13)
    b+=t(215,440,"更新次数",15)
    b+=t(80,475,"虚线：均匀3个上下文的下界 log 3",16)
    sim=report["embedding"]["center_cosines"];words=report["embedding"]["center_words"]
    for i,name in enumerate(words):
        b+=t(635+54*i,88,name,15)
        b+=t(590,125+48*i,name,15)
        for j in range(6):
            v=sim[i][j];q=(v+1)/2
            col=f"rgb({round(250-190*q)},{round(250-115*q)},{round(250-35*q)})"
            b+=f'<rect x="{625+54*j}" y="{100+48*i}" width="54" height="48" fill="{col}" stroke="white"/>'
            b+=t(630+54*j,129+48*i,f"{v:.2f}",13,"white" if q>.75 else "#26384c")
    b+=t(600,430,"只展示参与输入训练的6个中心词；",16)
    b+=t(600,460,"两组共现结构由任务人为规定。",16)
    b+=t(60,505,"由18个原创训练对、CPU float64实际得到；不是真实语义评测，也不是泛化成绩。",16)
    save("05-embedding-experiment.svg","小型完整softmax词嵌入训练真实结果",b,540)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=Path("build/experiments/14-tokenization-and-embeddings"))
    p.add_argument("--figure-dir",type=Path)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    checks=[]
    def check(name,passed,**details):
        if not passed:raise AssertionError((name,details))
        checks.append({"name":name,"passed":True,**details})
    # Original small weighted dictionary; terminal symbol is one atomic symbol.
    words=[("abab",3),("abac",2),("cab",1)]
    base={"a":0,"b":1,"c":2,"</w>":3};symbols={v:k for k,v in base.items()}
    rules,hand_hist,hand_data=learn_bpe([([base[c] for c in s]+[3],f) for s,f in words],4,3)
    display=[]
    for r in rules:
        symbols[r["new"]]=symbols[r["left"]]+symbols[r["right"]]
        display.append({"left":symbols[r["left"]],"right":symbols[r["right"]],
                        "new":symbols[r["new"]],"weighted_count":r["pair_count"]})
    check("three_hand_bpe_merges",[r["weighted_count"] for r in display]==[9,4,3] and [r["new"] for r in display]==["ab","ab</w>","abab</w>"],merges=display)
    check("non_overlapping_merge",merge_pair([0,0,0],(0,0),4)==[4,0])
    check("frozen_hand_segmentation",apply_bpe([0,1,0,1,3],rules)==[6] and apply_bpe([2,0,1,3],rules)==[2,5])
    unknown=[c for c in "abaz" if c not in base]
    check("character_bpe_unknown_base_symbol",unknown==["z"],unknown=unknown)
    train=[("我爱学习。",4),("我爱数学。",3),("你爱学习。",2),("I love learning.",3),("I love math.",2),("AI helps learning.",1)]
    br,bh,_=learn_bpe([(list(s.encode("utf8")),f) for s,f in train],256,16)
    bv=token_bytes(br)
    check("byte_vocabulary_and_merge_budget",len(bv)==272 and len(br)==16)
    samples=["我爱学习AI。","I love learning AI.","我爱AI。","café","cafe\u0301","𠮷"]
    rows=[]
    for s in samples:
        seq=apply_bpe(s.encode("utf8"),br)
        decoded=b"".join(bv[i] for i in seq).decode("utf8")
        assert decoded==s
        rows.append({"text":s,"codepoints":len(s),"utf8_bytes":len(s.encode("utf8")),
                     "whitespace_pieces":len(s.split()),"byte_bpe_tokens":len(seq),"token_ids":seq})
    check("byte_bpe_lossless_roundtrip_unseen_unicode",True,samples=len(samples))
    a="café";b="cafe\u0301"
    check("unicode_nfc_policy",a!=b and unicodedata.normalize("NFC",a)==unicodedata.normalize("NFC",b),codepoints_before=[len(a),len(b)],bytes_before=[len(a.encode()),len(b.encode())])
    invalid=False
    try:bytes([0xe6]).decode("utf8")
    except UnicodeDecodeError:invalid=True
    check("individual_byte_not_always_valid_utf8",invalid)
    spec={"schema_version":1,"base":"all_256_utf8_bytes_no_special_tokens","normalization":"none","pretokenization":"none; merges may cross spaces within a document","tie_break":"ascending integer pair ids","merges":br}
    spec_path=args.output_dir/"byte-bpe.json";spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding="utf8")
    loaded=json.loads(spec_path.read_text(encoding="utf8"))
    check("tokenizer_restore",all(apply_bpe(s.encode(),loaded["merges"])==r["token_ids"] for s,r in zip(samples,rows)))
    e=np.array([[0,0],[-1,0],[1,2],[0,1],[-1,2]],dtype=np.float64)
    ids=np.array([[2,3,2],[4,0,0]],dtype=np.int64)
    onehot=np.eye(5)[ids]
    lookup=e[ids];matmul=onehot@e
    check("one_hot_lookup_numpy",np.array_equal(lookup,matmul))
    embedding=torch.nn.Embedding(5,2,padding_idx=0)
    with torch.no_grad():embedding.weight.copy_(torch.from_numpy(e))
    actual=embedding(torch.from_numpy(ids));actual.sum().backward()
    expected_grad=np.array([[0,0],[0,0],[2,2],[1,1],[1,1]],dtype=np.float64)
    check("embedding_gather_and_repeated_gradient",np.array_equal(actual.detach().numpy(),lookup) and np.array_equal(embedding.weight.grad.numpy(),expected_grad),output=lookup.tolist(),gradient=expected_grad.tolist())
    perm=np.array([4,2,0,3,1]);inverse=np.argsort(perm)
    check("id_renumber_with_matching_rows",np.array_equal(e[perm][inverse[ids]],lookup))
    pair=torch.tensor([[2,3,0]])
    eh=embedding(pair).detach();mask=pair!=0
    masked=(eh*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
    naive=eh.mean(1)
    check("padding_idx_does_not_mask_mean",torch.equal(masked,torch.tensor([[.5,1.5]])) and not torch.equal(masked,naive),masked=masked.tolist(),unmasked=naive.tolist())
    apple=np.array([1.,1.]);fruit=np.array([2.,0.]);company=np.array([0.,2.])
    check("static_vs_context_function",np.array_equal(apple+fruit,[3,1]) and np.array_equal(apple+company,[1,3]))
    # Complete-softmax skip-gram style objective. Only 6 center rows are queried.
    vocabulary=["猫","狗","兔","苹果","香蕉","梨","跑","跳","睡","甜","酸","熟"]
    centers=[];targets=[]
    for c in range(6):
        for o in (range(6,9) if c<3 else range(9,12)):
            centers.append(c);targets.append(o)
    ci=torch.tensor(centers);ta=torch.tensor(targets)
    torch.manual_seed(14)
    E=torch.nn.Parameter(.1*torch.randn(12,8));U=torch.nn.Parameter(.1*torch.randn(12,8))
    en=E.detach().numpy();un=U.detach().numpy()
    logits_np=en[centers]@un.T;ex=np.exp(logits_np-logits_np.max(axis=1,keepdims=True));probs=ex/ex.sum(axis=1,keepdims=True)
    dz=(probs-np.eye(12)[targets])/len(targets)
    ge=np.zeros_like(en);np.add.at(ge,centers,dz@un);gu=dz.T@en[centers]
    ad=torch.autograd.grad(F.cross_entropy(E[ci]@U.T,ta),(E,U))
    error=max(np.max(np.abs(ge-ad[0].numpy())),np.max(np.abs(gu-ad[1].numpy())))
    check("full_softmax_manual_gradients",error<1e-14,max_error=float(error))
    opt=torch.optim.Adam([E,U],lr=.05);history=[]
    for step in range(201):
        loss=F.cross_entropy(E[ci]@U.T,ta)
        if step%20==0:history.append({"step":step,"loss":loss.item()})
        if step==200:break
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    check("embedding_optimization_loss_decreases",history[-1]["loss"]<history[0]["loss"] and history[-1]["loss"]>=math.log(3)-1e-12)
    center=F.normalize(E[:6].detach(),dim=1);sim=(center@center.T).numpy()
    checkpoint={"vocabulary":vocabulary,"input_embedding":E.detach(),"output_embedding":U.detach(),"protocol":{"seed":14,"pairs":list(zip(centers,targets)),"optimizer":"Adam lr=.05 defaults","steps":200}}
    torch.save(checkpoint,args.output_dir/"embedding.pt")
    restored=torch.load(args.output_dir/"embedding.pt",weights_only=True)
    check("embedding_checkpoint_restore",torch.equal(restored["input_embedding"][ci]@restored["output_embedding"].T,E.detach()[ci]@U.detach().T))
    nearest={vocabulary[i]:[vocabulary[j] for j in np.argsort(-sim[i]) if j!=i][:2] for i in range(6)}
    report={"versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"unicode_database":unicodedata.unidata_version},
       "device":"cpu","dtype":"float64","threads":1,"seed":14,"checks":checks,
       "hand_bpe":{"weighted_words":words,"merges":display,"weighted_tokens":hand_hist},
       "byte_bpe":{"training_documents":train,"merges":br,"weighted_training_tokens":bh,"vocab_size":len(bv),"tokenizer_sha256":hashlib.sha256(spec_path.read_bytes()).hexdigest()},
       "tokenization":rows,"embedding":{"vocabulary":vocabulary,"center_words":vocabulary[:6],"parameters":192,
       "training_pairs":len(centers),"history":history,"lower_bound_nats":math.log(3),"center_cosines":sim.tolist(),"nearest_centers":nearest,
       "initial_loss":history[0]["loss"],"final_loss":history[-1]["loss"],
       "protocol":"18 exhaustive synthetic center/context pairs; training demonstration only, no validation/test or semantic generalization claim"},
       "elapsed_seconds":time.perf_counter()-START,"import_seconds":IMPORT_SECONDS,"python_traced_peak_bytes":tracemalloc.get_traced_memory()[1],
       "memory_scope":"Python-tracked allocations, excludes much native tensor memory; not process RSS"}
    (args.output_dir/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    save_figures(args.figure_dir or args.output_dir/"figures",report)
    print(json.dumps({"checks_passed":len(checks),"elapsed_seconds":report["elapsed_seconds"],"import_seconds":IMPORT_SECONDS,"hand_merges":display,"byte_counts":bh,"tokenization":rows,"initial_loss":history[0]["loss"],"final_loss":history[-1]["loss"],"nearest_centers":nearest},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
