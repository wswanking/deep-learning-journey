"""A tiny CPU Transformer block, checked against NumPy and PyTorch primitives."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import time
START=time.perf_counter()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];DT=torch.float64

def check(test,message):
 if not test:raise AssertionError(message)
def maximum(x):return float(x.detach().abs().max())
def positional(length,width):
 check(width%2==0,'even positional width required')
 p=torch.arange(length,dtype=DT)[:,None];freq=torch.exp(torch.arange(0,width,2,dtype=DT)*(-math.log(10000.)/width));out=torch.zeros(length,width,dtype=DT)
 out[:,0::2]=torch.sin(p*freq);out[:,1::2]=torch.cos(p*freq);return out

class Block(nn.Module):
 def __init__(self,width=8,heads=2,ffn=16):
  super().__init__();check(width%heads==0,'head divisibility');self.width=width;self.heads=heads;self.head_width=width//heads
  self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width);self.output=nn.Linear(width,width)
  self.norm1=nn.LayerNorm(width,eps=1e-5);self.norm2=nn.LayerNorm(width,eps=1e-5);self.ff1=nn.Linear(width,ffn);self.ff2=nn.Linear(ffn,width)
 def attention(self,x,blocked=None):
  b,t,d=x.shape
  split=lambda projection:projection(x).reshape(b,t,self.heads,self.head_width).transpose(1,2)
  q,k,v=map(split,[self.q,self.k,self.v]);scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_width)
  if blocked is not None:
   if blocked.shape!=(t,t) or blocked.dtype!=torch.bool:raise ValueError('blocked must be boolean TxT')
   if bool(blocked.all(dim=-1).any()):raise ValueError('each query requires at least one allowed key')
   scores=scores.masked_fill(blocked,float('-inf'))
  weights=torch.softmax(scores,dim=-1);heads=weights@v;merged=heads.transpose(1,2).contiguous().reshape(b,t,d)
  return self.output(merged),weights,(q,k,v)
 def forward(self,x,blocked=None):
  attention,weights,qkv=self.attention(self.norm1(x),blocked);mid=x+attention
  return mid+self.ff2(F.relu(self.ff1(self.norm2(mid)))),weights,qkv

def numpy_block(model,x,blocked):
 state={k:v.detach().numpy() for k,v in model.state_dict().items()}
 def linear(a,prefix):return a@state[prefix+'.weight'].T+state[prefix+'.bias']
 def norm(a,prefix):
  mean=a.mean(-1,keepdims=True);var=((a-mean)**2).mean(-1,keepdims=True)
  return (a-mean)/np.sqrt(var+1e-5)*state[prefix+'.weight']+state[prefix+'.bias']
 b,t,d=x.shape;z=norm(x,'norm1');split=lambda prefix:linear(z,prefix).reshape(b,t,model.heads,model.head_width).transpose(0,2,1,3)
 q,k,v=map(split,['q','k','v']);scores=q@k.transpose(0,1,3,2)/math.sqrt(model.head_width);scores=np.where(blocked,-np.inf,scores)
 # Stable row softmax. Our valid causal masks always retain at least self.
 scores=scores-np.max(scores,axis=-1,keepdims=True);weights=np.exp(scores);weights=weights/weights.sum(-1,keepdims=True)
 merged=(weights@v).transpose(0,2,1,3).reshape(b,t,d);mid=x+linear(merged,'output');out=mid+linear(np.maximum(linear(norm(mid,'norm2'),'ff1'),0),'ff2')
 return out,weights

def reference_layer(model):
 ref=nn.TransformerEncoderLayer(8,2,16,dropout=0.,activation='relu',batch_first=True,norm_first=True,layer_norm_eps=1e-5).to(dtype=DT)
 with torch.no_grad():
  ref.self_attn.in_proj_weight.copy_(torch.cat([model.q.weight,model.k.weight,model.v.weight]));ref.self_attn.in_proj_bias.copy_(torch.cat([model.q.bias,model.k.bias,model.v.bias]));ref.self_attn.out_proj.load_state_dict(model.output.state_dict())
  ref.linear1.load_state_dict(model.ff1.state_dict());ref.linear2.load_state_dict(model.ff2.state_dict());ref.norm1.load_state_dict(model.norm1.state_dict());ref.norm2.load_state_dict(model.norm2.state_dict())
 return ref.eval()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--skip-figures',action='store_true');args=parser.parse_args()
 torch.set_num_threads(1);torch.set_default_dtype(DT);torch.use_deterministic_algorithms(True);torch.manual_seed(1601);begin=time.perf_counter()
 model=Block().to(dtype=DT).eval();embedding=nn.Embedding(8,8).to(dtype=DT);lm_head=nn.Linear(8,8).to(dtype=DT)
 ids=torch.tensor([[1,3,2,4,6],[2,1,5,3,7]],dtype=torch.long);pe=positional(5,8);base=embedding(ids).detach();x=(base+pe).detach();mask=torch.ones(5,5,dtype=torch.bool).triu(1)
 out,weights,qkv=model(x,mask);np_out,np_weights=numpy_block(model,x.numpy(),mask.numpy());np_error=float(np.max(np.abs(np_out-out.detach().numpy())))
 check(np_error<1e-12 and np.max(np.abs(np_weights-weights.detach().numpy()))<1e-12,'NumPy full block disagreement')
 reference=reference_layer(model);ref_out=reference(x,src_mask=mask);reference_error=maximum(out-ref_out);check(reference_error<1e-12,'reference TransformerEncoderLayer disagreement')
 q,k,v=qkv;manual_heads=weights@v
 sdpa=F.scaled_dot_product_attention(q,k,v,attn_mask=~mask,dropout_p=0.)
 sdpa_error=maximum(manual_heads-sdpa);check(sdpa_error<1e-12,'SDPA True=allowed disagreement')
 check(maximum(weights.sum(-1)-1)<1e-14 and maximum(weights.masked_select(mask[None,None]))==0,'causal weights')
 perturbed=x.clone();perturbed[:,3:]+=torch.tensor([30.,-20.,10.,-5.,2.,4.,-9.,7.],dtype=DT)
 changed,_,_=model(perturbed,mask);causal_delta=maximum(out[:,:3]-changed[:,:3]);check(causal_delta==0,'future content leaked into prefix')
 dense,_,_=model(x,None);dense_changed,_,_=model(perturbed,None);dense_delta=maximum(dense[:,:3]-dense_changed[:,:3]);check(dense_delta>1e-6,'unmasked perturbation did not demonstrate future access')
 # Loss at t=2 (zero based) depends on past/current positions, but not later positions.
 xg=x.clone().requires_grad_(True);yg,_,_=model(xg,mask);loss=yg[0,2].square().sum();grad=torch.autograd.grad(loss,xg)[0]
 future_gradient=maximum(grad[0,3:]);check(future_gradient==0 and maximum(grad[0,:3])>0,'causal gradient boundary')
 eps=1e-6;xp=x.clone();xm=x.clone();xp[0,0,0]+=eps;xm[0,0,0]-=eps
 fd=float(((model(xp,mask)[0][0,2].square().sum()-model(xm,mask)[0][0,2].square().sum())/(2*eps)).detach());ad=float(grad[0,0,0]);check(abs(fd-ad)<1e-7,'input finite difference disagreement')
 # Without position information, noncausal block is permutation equivariant.
 perm=torch.tensor([2,0,4,1,3]);inv=torch.argsort(perm);plain=model(base,None)[0];permuted=model(base[:,perm],None)[0][:,inv];equivariance_error=maximum(plain-permuted);check(equivariance_error<1e-12,'permutation equivariance')
 with_positions=model(base+pe,None)[0];reassigned=model(base[:,perm]+pe,None)[0][:,inv];position_delta=maximum(with_positions-reassigned);check(position_delta>1e-6,'fixed position encoding had no effect')
 # Entirely blocked query is rejected rather than silently softmax(-inf,...,-inf).
 rejected=False
 try:model(x,torch.ones(5,5,dtype=torch.bool))
 except ValueError:rejected=True
 check(rejected,'all-masked rows must be rejected')
 # Hand example: scalar queries (1,0), keys (1,0), values (2,4).
 hq=torch.tensor([[[[1.],[0.]]]],dtype=DT);hk=hq.clone();hv=torch.tensor([[[[2.],[4.]]]],dtype=DT)
 hand_dense=F.scaled_dot_product_attention(hq,hk,hv,dropout_p=0.);hand_causal=F.scaled_dot_product_attention(hq,hk,hv,is_causal=True,dropout_p=0.)
 check(abs(float(hand_dense[0,0,0,0])-(2*math.e+4)/(math.e+1))<1e-14,'attention hand first row');check(abs(float(hand_causal[0,0,0,0])-2)<1e-14 and float(hand_causal[0,0,1,0])==3,'causal hand rows')
 # Teacher-forced next-token objective, no training or language claims.
 tokens=torch.tensor([[1,2,3,4,5]],dtype=torch.long);inputs=tokens[:,:-1];targets=tokens[:,1:];features=embedding(inputs)+positional(4,8);hidden=model(features,torch.ones(4,4,dtype=torch.bool).triu(1))[0];logits=lm_head(hidden)
 ce=F.cross_entropy(logits.reshape(-1,8),targets.reshape(-1));manual_ce=-torch.log_softmax(logits,-1).gather(-1,targets.unsqueeze(-1)).mean();check(torch.equal(ce,manual_ce) or abs(float(ce.detach()-manual_ce.detach()))<1e-14,'shifted CE reduction')
 model.zero_grad(set_to_none=True);ce.backward();check(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'block parameter gradient')
 folder=ROOT/'build/experiments/16-transformer';folder.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),folder/'block.pt');restored=Block().to(dtype=DT);restored.load_state_dict(torch.load(folder/'block.pt',weights_only=True));restored.eval();check(torch.equal(out.detach(),restored(x,mask)[0].detach()),'block save restore')
 report={'run_at':datetime.now().astimezone().isoformat(timespec='seconds'),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':1},'configuration':{'seed':1601,'batch':2,'length':5,'width':8,'heads':2,'head_width':4,'ffn_width':16,'epsilon':1e-5,'activation':'ReLU','normalization':'Pre-LN','dropout':0,'block_parameters':sum(p.numel() for p in model.parameters()),'embedding_and_head_parameters':sum(p.numel() for p in embedding.parameters())+sum(p.numel() for p in lm_head.parameters()),'training_updates':0},'errors':{'numpy':np_error,'reference_layer':reference_error,'sdpa':sdpa_error,'causal_prefix_perturbation':causal_delta,'unmasked_prefix_perturbation':dense_delta,'future_input_gradient':future_gradient,'autograd_input':ad,'central_difference_input':fd,'permutation_without_positions':equivariance_error,'permutation_with_reassigned_positions':position_delta},'hand':{'unmasked_attention':hand_dense.flatten().tolist(),'causal_attention':hand_causal.flatten().tolist(),'layer_norm_1_3':[-1/math.sqrt(1+1e-5),1/math.sqrt(1+1e-5)]},'positions':pe.tolist(),'ids':ids.tolist(),'attention_first_sample':weights[0].detach().tolist(),'block_output_first_sample':out[0].detach().tolist(),'objectives':{'input':inputs.tolist(),'target':targets.tolist(),'logit_shape':list(logits.shape),'mean_next_token_ce':float(ce.detach()),'all_parameter_gradients_finite':True},'checks':{'all_masked_rejected':rejected,'checkpoint_restored_exact':True},'resources':{'checks_seconds':time.perf_counter()-begin,'including_import_seconds':time.perf_counter()-START,'input_feature_bytes':x.numel()*x.element_size(),'attention_weights_bytes':weights.numel()*weights.element_size(),'peak_process_memory':'not measured'}}
 (folder/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if not args.skip_figures:
  from figures import write_figures
  write_figures(report,ROOT/'docs/assets/16-transformer')
 print(json.dumps({'status':'passed','configuration':report['configuration'],'errors':report['errors'],'objectives':report['objectives'],'resources':report['resources']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
