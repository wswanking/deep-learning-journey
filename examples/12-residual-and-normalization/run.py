"""Chapter 12: real CPU residual, normalization-mode and transfer experiments."""
from __future__ import annotations
import argparse
import copy
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, residual=True):
        super().__init__(); self.first=nn.Linear(8,8); self.second=nn.Linear(8,8); self.residual=residual
    def forward(self,x):
        branch=self.second(torch.tanh(self.first(x)))
        return x+branch if self.residual else branch


class Network(nn.Module):
    def __init__(self,residual=True,classes=4):
        super().__init__(); self.stem=nn.Linear(2,8); self.blocks=nn.Sequential(*(Block(residual) for _ in range(3))); self.head=nn.Linear(8,classes)
    def features(self,x): return self.blocks(torch.tanh(self.stem(x)))
    def forward(self,x): return self.head(self.features(x))
    def backbone_parameters(self): return list(self.stem.parameters())+list(self.blocks.parameters())


def dataset(count,seed,task):
    rng=torch.Generator().manual_seed(seed)
    x=4*torch.rand((count,2),generator=rng,dtype=torch.float64)-2
    if task=='source': y=(x[:,0]>=0).long()+2*(x[:,1]>=0).long()
    else: y=(x[:,0]+0.5*x[:,1]>=0).long()
    return x,y


@torch.no_grad()
def measure(model,data):
    model.eval(); x,y=data; scores=model(x)
    return {'loss':float(nn.functional.cross_entropy(scores,y)), 'accuracy':float((scores.argmax(-1)==y).double().mean())}


def train(model,train_data,validation,steps,optimizer):
    history=[{'step':0,'train':measure(model,train_data),'validation':measure(model,validation)}]
    x,y=train_data
    model.train(); optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(x),y).backward()
    initial_stem_norm=None if model.stem.weight.grad is None else float(model.stem.weight.grad.norm())
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(x),y); loss.backward(); optimizer.step()
        history.append({'step':step+1,'train':measure(model,train_data),'validation':measure(model,validation)})
    return {'history':history,'initial_stem_gradient_norm':initial_stem_norm,'final_train':history[-1]['train'],'final_validation':history[-1]['validation']}


def normalization():
    first=torch.tensor([[1.,2.],[3.,6.]],dtype=torch.float64)
    other=torch.tensor([[1.,2.],[-3.,-6.]],dtype=torch.float64)
    bn=nn.BatchNorm1d(2,eps=1e-5,momentum=1.0,affine=True).double()
    bn.train(); training=bn(first).detach().clone(); running_mean=bn.running_mean.clone(); running_var=bn.running_var.clone()
    bn.eval(); evaluation=bn(first).detach().clone()
    def batch_output(x):
        module=nn.BatchNorm1d(2,affine=False,track_running_stats=False).double()
        return module(x).detach()
    ln=nn.LayerNorm(2,eps=1e-5,elementwise_affine=False).double()
    ln.train(); ln_train=ln(first).detach(); ln.eval(); ln_eval=ln(first).detach()
    frozen=nn.BatchNorm1d(2).double(); frozen.requires_grad_(False); before=frozen.running_mean.clone()
    frozen.train()
    with torch.no_grad(): frozen(first)
    buffer_changed=not torch.equal(before,frozen.running_mean)
    bn_single_raises=False
    try: nn.BatchNorm1d(2).double()(first[:1])
    except ValueError: bn_single_raises=True
    bn_spatial=nn.BatchNorm2d(2).double()(torch.arange(8,dtype=torch.float64).reshape(1,2,2,2))
    gradient_input=torch.tensor([2.,-1.],dtype=torch.float64,requires_grad=True)
    matrix=torch.tensor([[.2,0.],[0.,-.1]],dtype=torch.float64)
    output=gradient_input+matrix@gradient_input
    output.sum().backward()
    cancellation=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    (cancellation-cancellation).backward()
    return {'input':first.tolist(),'mean':first.mean(0).tolist(),'biased_variance':first.var(0,correction=0).tolist(),
            'train_output':training.tolist(),'running_mean':running_mean.tolist(),'running_variance':running_var.tolist(),'eval_output':evaluation.tolist(),
            'bn_first_sample_batch_a':batch_output(first)[0].tolist(),'bn_first_sample_batch_b':batch_output(other)[0].tolist(),
            'ln_first_sample_batch_a':ln(first)[0].tolist(),'ln_first_sample_batch_b':ln(other)[0].tolist(),
            'ln_train_eval_equal':bool(torch.equal(ln_train,ln_eval)),
            'frozen_bn_running_buffer_changed':buffer_changed,'bn_one_value_per_channel_raises':bn_single_raises,
            'bn_batch_one_with_spatial_values_is_finite':bool(torch.isfinite(bn_spatial).all()),
            'residual_hand_output':output.detach().tolist(),'residual_hand_gradient':gradient_input.grad.tolist(),
            'residual_cancellation_gradient':float(cancellation.grad)}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,default=Path('build/experiments/12-residual-and-normalization')); parser.add_argument('--write-figures',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    source_train=dataset(256,101,'source'); source_val=dataset(256,102,'source'); target_train=dataset(32,201,'target'); target_val=dataset(256,202,'target')
    comparison=[]; transfer=[]; all_checks=[]
    def require(condition,name):
        if not bool(condition): raise RuntimeError(name)
        all_checks.append(name)
    for seed in [11,22,33]:
        torch.manual_seed(seed); base=Network(True).double(); initial=copy.deepcopy(base.state_dict()); pretrained=None
        for residual in [False,True]:
            model=Network(residual).double(); model.load_state_dict(initial)
            result=train(model,source_train,source_val,200,torch.optim.Adam(model.parameters(),lr=.01,foreach=False))
            result.update({'seed':seed,'residual':residual,'parameter_count':sum(p.numel() for p in model.parameters())}); comparison.append(result)
            require(result['final_train']['loss']<result['history'][0]['train']['loss'],f'source_loss_decreases_{seed}_{residual}')
            if residual: pretrained=copy.deepcopy(model.state_dict())
        torch.manual_seed(seed+100); same_head=copy.deepcopy(nn.Linear(8,2).double().state_dict())
        for mode in ['scratch','frozen','finetune']:
            model=Network(True).double(); model.load_state_dict(initial if mode=='scratch' else pretrained); model.head=nn.Linear(8,2).double(); model.head.load_state_dict(same_head)
            backbone=model.backbone_parameters(); before=[p.detach().clone() for p in backbone]
            if mode=='frozen':
                for parameter in backbone: parameter.requires_grad_(False)
                groups=[{'params':list(model.head.parameters()),'lr':.02}]
            else: groups=[{'params':backbone,'lr':.001},{'params':list(model.head.parameters()),'lr':.02}]
            result=train(model,target_train,target_val,100,torch.optim.Adam(groups,foreach=False))
            change=max(float((a-b).detach().abs().max()) for a,b in zip(backbone,before))
            result.update({'seed':seed,'mode':mode,'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'maximum_backbone_change':change}); transfer.append(result)
            require(result['final_train']['loss']<result['history'][0]['train']['loss'],f'target_loss_decreases_{seed}_{mode}')
            require(change==0 if mode=='frozen' else change>0,f'backbone_update_policy_{seed}_{mode}')
            if seed==11 and mode=='finetune':
                checkpoint=args.output_dir/'target-finetuned-seed11.pt'; torch.save(model.state_dict(),checkpoint)
                reloaded=Network(True,classes=2).double(); reloaded.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')); reloaded.eval(); model.eval()
                with torch.no_grad(): reload_error=float((model(target_val[0])-reloaded(target_val[0])).abs().max())
                require(reload_error==0,'target_model_reloads_identical_logits')
    demo=normalization()
    require(comparison[0]['parameter_count']==492,'source_network_has_492_parameters')
    require(all(r['parameter_count']==474 for r in transfer),'target_network_has_474_parameters')
    require(all(r['trainable_parameter_count']==18 for r in transfer if r['mode']=='frozen'),'frozen_head_has_18_trainable_parameters')
    require(np.allclose(demo['mean'],[2,4]),'bn_hand_mean')
    require(np.allclose(demo['biased_variance'],[1,4]),'bn_train_population_variance')
    require(np.allclose(demo['running_variance'],[2,8]),'bn_running_unbiased_variance')
    require(not np.allclose(demo['train_output'],demo['eval_output']),'bn_modes_differ_on_same_input')
    require(not np.allclose(demo['bn_first_sample_batch_a'],demo['bn_first_sample_batch_b']),'bn_output_depends_on_batch_companions')
    require(np.allclose(demo['ln_first_sample_batch_a'],demo['ln_first_sample_batch_b']),'ln_output_independent_of_batch_companions')
    require(demo['ln_train_eval_equal'],'ln_modes_match')
    require(demo['frozen_bn_running_buffer_changed'],'freezing_parameters_does_not_freeze_bn_buffers')
    require(demo['bn_one_value_per_channel_raises'] and demo['bn_batch_one_with_spatial_values_is_finite'],'bn_limit_is_values_per_channel_not_just_batch_size')
    require(np.allclose(demo['residual_hand_output'],[2.4,-.9]),'residual_hand_forward')
    require(np.allclose(demo['residual_hand_gradient'],[1.2,.9]),'residual_identity_plus_jacobian')
    require(demo['residual_cancellation_gradient']==0,'residual_has_no_unconditional_gradient_lower_bound')
    report={'chapter':'12','collected_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'python_executable':sys.executable,'device':'CPU','dtype':'float64','threads':1},
            'data':{'source':'independent synthetic Uniform[-2,2]^2; no downloads','source_task':'quadrant class (x1>=0)+2*(x2>=0)','target_task':'binary class x1+0.5*x2>=0', 'source_train_count':256,'source_validation_count':256,'target_train_count':32,'target_validation_count':256,'data_seeds':[101,102,201,202],'model_seeds':[11,22,33],'test_set':None},
            'training':{'source_steps':200,'source_optimizer':'Adam lr=.01','target_steps':100,'target_optimizer':'Adam head lr=.02 backbone lr=.001 where unfrozen','batch':'full batch','weight_decay':0,'normalization_in_trained_models':False,'checkpoint_scope':'inference weights only; not exact training resume'},
            'source_comparison':comparison,'transfer':transfer,'normalization_and_residual_demonstrations':demo,
            'inference_reload_max_error':reload_error,'resources':{'elapsed_seconds_excluding_imports':time.perf_counter()-started,'source_parameter_payload_bytes':492*8,'target_parameter_payload_bytes':474*8,'memory_scope':'parameter numerical payload only; no process peak measurement'},
            'checks':{'status':'passed','count':len(all_checks),'names':all_checks}}
    out=args.output_dir/'results.json'; out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.write_figures:
        from figures import write_figures
        write_figures(report,Path(__file__).resolve().parents[2]/'docs/assets/12-residual-and-normalization')
    compact={key:report[key] for key in ['collected_at','environment','training','normalization_and_residual_demonstrations','resources','checks']}
    compact['source_results']=[{k:r[k] for k in ['seed','residual','parameter_count','initial_stem_gradient_norm','final_train','final_validation']} for r in comparison]
    compact['transfer_results']=[{k:r[k] for k in ['seed','mode','trainable_parameter_count','maximum_backbone_change','final_train','final_validation']} for r in transfer]
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
