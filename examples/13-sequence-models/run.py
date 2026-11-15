"""CPU sequence regression and checked RNN/LSTM/GRU mechanics, no downloads."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
START=time.perf_counter()
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
DT=torch.float64

def require(condition,message):
    if not condition:raise AssertionError(message)

def mechanics():
    # Two-step scalar RNN: compare hand BPTT, central differences, and autograd.
    wh=torch.tensor(.5,dtype=DT,requires_grad=True)
    h1=torch.tanh(torch.tensor(1.,dtype=DT));h2=torch.tanh(wh*h1);loss=.5*h2.square()
    auto=float(torch.autograd.grad(loss,wh)[0]);manual=float(h2.detach()*(1-h2.detach().square())*h1)
    f=lambda w:.5*math.tanh(w*math.tanh(1))**2
    diff=(f(.5+1e-6)-f(.5-1e-6))/(2e-6)
    require(abs(auto-manual)<1e-14 and abs(auto-diff)<1e-9,'scalar BPTT disagreement')
    # Full shared-parameter BPTT also includes a nonzero first hidden state.
    xs=np.array([.2,-.1,.5]);h0=.3;wx=.7;w=.4;target=.2
    hs=[h0]
    for x in xs:hs.append(math.tanh(wx*x+w*hs[-1]))
    deltas=np.zeros(3);carry=hs[-1]-target
    for t in range(2,-1,-1):
        deltas[t]=carry*(1-hs[t+1]**2);carry=w*deltas[t]
    shared=float(sum(deltas[t]*hs[t] for t in range(3)))
    wt=torch.tensor(w,dtype=DT,requires_grad=True);ht=torch.tensor(h0,dtype=DT)
    for x in xs:ht=torch.tanh(wx*float(x)+wt*ht)
    ash=float(torch.autograd.grad(.5*(ht-target)**2,wt)[0])
    require(abs(shared-ash)<1e-14,'shared BPTT contributions not summed')
    # Isolated zero-trajectory tanh chain has exact derivative w**T at h0=0.
    sensitivity=[]
    for weight in [.5,1.,1.5]:
        initial=torch.tensor(0.,dtype=DT,requires_grad=True);h=initial
        for _ in range(40):h=torch.tanh(weight*h)
        grad=float(torch.autograd.grad(h,initial)[0]);require(abs(grad-weight**40)<1e-8*max(1,abs(grad)),'chain derivative')
        sensitivity.append({'weight':weight,'steps':40,'gradient':grad})
    # LSTM ordering in PyTorch: i,f,g,o. Predeclare gate values using inverse activations.
    cell=nn.LSTMCell(1,1).to(dtype=DT)
    with torch.no_grad():
        for p in cell.parameters():p.zero_()
        cell.bias_ih.copy_(torch.tensor([math.log(.2/.8),math.log(.9/.1),math.atanh(.5),math.log(.8/.2)],dtype=DT))
    hl,cl=cell(torch.zeros(1,1,dtype=DT),(torch.zeros(1,1,dtype=DT),torch.ones(1,1,dtype=DT)))
    require(abs(float(cl.detach())-1)<1e-14 and abs(float(hl.detach())-.8*math.tanh(1))<1e-14,'LSTM hand example')
    # PyTorch GRU uses reset after the recurrent affine transform.
    gru=nn.GRUCell(1,1).to(dtype=DT)
    with torch.no_grad():
        for p in gru.parameters():p.zero_()
        gru.bias_ih.copy_(torch.tensor([math.log(.25/.75),math.log(.8/.2),.1],dtype=DT))
        gru.weight_hh[2,0]=1.
    hg=gru(torch.zeros(1,1,dtype=DT),torch.tensor([[.6]],dtype=DT));gexpect=.8*.6+.2*math.tanh(.1+.25*.6)
    require(abs(float(hg.detach())-gexpect)<1e-14,'GRU hand example')
    # Vector-valued candidate-reset location really matters, even with zero biases.
    reset=torch.tensor([.2,.8],dtype=DT);prev=torch.tensor([1.,2.],dtype=DT);matrix=torch.tensor([[1.,2.],[3.,4.]],dtype=DT)
    after=reset*(matrix@prev);before=matrix@(reset*prev)
    require(not torch.allclose(after,before),'GRU convention counterexample')
    # Full vs deliberately detached graph: both start from same numeric state.
    h_init=torch.tensor(.3,dtype=DT,requires_grad=True);h=torch.tanh(.7*.2+.4*h_init)
    h_detached=h.detach().requires_grad_(True);tail=torch.tanh(.7*(-.1)+.4*h_detached)
    full=torch.tanh(.7*(-.1)+.4*h)
    connected=torch.autograd.grad(full,h_init)[0]
    cut=torch.autograd.grad(tail,h_init,allow_unused=True)[0]
    require(cut is None and float(connected)!=0,'detach boundary')
    # Norm clipping controls large gradient; it cannot reconstruct vanished information.
    parameter=nn.Parameter(torch.zeros(2,dtype=DT));parameter.grad=torch.tensor([3.,4.],dtype=DT)
    before_norm=float(nn.utils.clip_grad_norm_([parameter],2.))
    require(before_norm==5. and abs(float(parameter.grad.norm())-2)<1e-6,'gradient clipping')
    return {'scalar_rnn':{'h1':float(h1),'h2':float(h2.detach()),'loss':float(loss.detach()),'manual_wh_gradient':manual,'autograd_wh_gradient':auto,'central_difference':diff},'shared_bptt':{'states':hs,'deltas':deltas.tolist(),'manual_gradient':shared,'autograd_gradient':ash},'sensitivity':sensitivity,'lstm':{'f':.9,'i':.2,'g':.5,'o':.8,'c':float(cl.detach()),'h':float(hl.detach())},'gru':{'r':.25,'z':.8,'candidate':math.tanh(.25),'h':float(hg.detach()),'reset_after':after.tolist(),'reset_before':before.tolist()},'detach':{'full_initial_gradient':float(connected),'detached_initial_gradient':None},'clipping':{'before':before_norm,'after':parameter.grad.tolist()}}

class SequenceModel(nn.Module):
    def __init__(self,kind,hidden):
        super().__init__();klass={'RNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[kind]
        self.recurrent=klass(1,hidden,batch_first=True);self.readout=nn.Linear(hidden,1)
    def forward(self,x):
        hidden,_=self.recurrent(x)  # state defaults to zeros for every independent sequence
        return self.readout(hidden)

def tensors(data):return torch.from_numpy(data).to(dtype=DT)
def mse(pred,x,lag):return (pred[:,lag:]-x[:,:-lag]).square().mean()

def train(kind,seed,train_x,val_x,lag,hidden,epochs,folder):
    torch.manual_seed(seed);model=SequenceModel(kind,hidden).to(dtype=DT);opt=torch.optim.Adam(model.parameters(),lr=.01)
    curves={'epoch':[],'train':[],'validation':[]};best=math.inf;best_state=None;best_epoch=0;clip_events=0;max_norm=0.
    def record(epoch):
        nonlocal best,best_state,best_epoch
        model.eval()
        with torch.no_grad():tr=float(mse(model(train_x),train_x,lag));va=float(mse(model(val_x),val_x,lag))
        curves['epoch'].append(epoch);curves['train'].append(tr);curves['validation'].append(va)
        if va<best:best=va;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    record(0)
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad(set_to_none=True);loss=mse(model(train_x),train_x,lag);loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'nonfinite gradient')
        norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.));clip_events+=norm>1;max_norm=max(max_norm,norm);opt.step()
        if epoch%10==0 or epoch==epochs:record(epoch)
    model.load_state_dict(best_state);model.eval()
    checkpoint=folder/'checkpoints'/f'{kind.lower()}-seed{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save(best_state,checkpoint)
    restored=SequenceModel(kind,hidden).to(dtype=DT);restored.load_state_dict(torch.load(checkpoint,weights_only=True));restored.eval()
    with torch.no_grad():require(torch.equal(model(val_x),restored(val_x)),'checkpoint restore mismatch')
    # Full-sequence evaluation must match causal streaming with carried recurrent state.
    with torch.no_grad():
        sequence,_=model.recurrent(val_x[:2]);state=None;pieces=[]
        for t in range(val_x.shape[1]):
            out,state=model.recurrent(val_x[:2,t:t+1],state);pieces.append(out)
        stream=torch.cat(pieces,dim=1);require(torch.allclose(sequence,stream,rtol=1e-12,atol=1e-12),'stream/full mismatch')
    return model,{'kind':kind,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'best_validation_mse':best,'curves':curves,'gradient_clip_events':clip_events,'maximum_preclip_norm':max_norm,'checkpoint':str(checkpoint.relative_to(ROOT)),'restored_exact':True,'stream_equals_full':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=400);parser.add_argument('--skip-figures',action='store_true');args=parser.parse_args()
    require(args.epochs>=1,'epochs must be positive')
    torch.set_num_threads(1);torch.set_default_dtype(DT);torch.use_deterministic_algorithms(True)
    begin=time.perf_counter();folder=ROOT/'build/experiments/13-sequence-models';folder.mkdir(parents=True,exist_ok=True)
    checked=mechanics();rng=np.random.default_rng(20261115);length=20;lag=4;hidden=12
    arrays=[rng.uniform(-1,1,size=(n,length,1)).astype(np.float64) for n in [64,32,32]]
    train_x,val_x,test_x=map(tensors,arrays);runs=[];models=[]
    for kind in ['RNN','LSTM','GRU']:
        model,report=train(kind,17,train_x,val_x,lag,hidden,args.epochs,folder);runs.append(report);models.append(model)
    # One reproducibility rerun, before final test evaluation.
    repeat,rep=train('RNN',17,train_x,val_x,lag,hidden,args.epochs,folder)
    require(rep['curves']==runs[0]['curves'],'RNN exact curve reproducibility')
    with torch.no_grad():
        require(torch.equal(repeat(val_x),models[0](val_x)),'RNN exact restored predictions')
        for model,run in zip(models,runs):
            pred=model(test_x);run['test_mse']=float(mse(pred,test_x,lag));run['train_mse']=float(mse(model(train_x),train_x,lag));run['sample_prediction']=pred[0,:,0].tolist()
        zero_baseline=float(test_x[:,:-lag].square().mean())
    elapsed=time.perf_counter()-begin
    report={'run_at':datetime.now().astimezone().isoformat(timespec='seconds'),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':1},'protocol':{'data_seed':20261115,'model_seed':17,'sequence_length':length,'lag':lag,'input_size':1,'hidden_size':hidden,'epochs':args.epochs,'validation_interval':10,'batch_size':64,'optimizer':'Adam','learning_rate':.01,'betas':[.9,.999],'epsilon':1e-8,'gradient_max_norm':1.,'loss':'MSE over B*(T-lag) scalar predictions; first lag outputs masked','test_after_all_runs':True,'one_seed_only':True},'data':{'counts':[64,32,32],'sha256':[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],'array_bytes':sum(a.nbytes for a in arrays),'sample_input':arrays[2][0,:,0].tolist(),'sample_target':arrays[2][0,:-lag,0].tolist()},'mechanics':checked,'runs':runs,'zero_test_mse':zero_baseline,'repeated_rnn_curves_exact':True,'resources':{'training_and_checks_seconds':elapsed,'including_import_seconds':time.perf_counter()-START,'process_peak_memory':'not measured'}}
    (folder/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not args.skip_figures:
        from figures import write_figures
        write_figures(report,ROOT/'docs/assets/13-sequence-models')
    print(json.dumps({'status':'passed','runs':[{k:r[k] for k in ['kind','parameters','best_epoch','best_validation_mse','test_mse','gradient_clip_events']} for r in runs],'zero_test_mse':zero_baseline,'resources':report['resources']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
