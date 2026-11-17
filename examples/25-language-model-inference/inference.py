"""Inference-only adapters for the read-only chapter-21 decoder; no padding or sliding window."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class CachedDecoder:
    def __init__(self,model):
        self.model=model
    @torch.inference_mode()
    def __call__(self,ids,cache=None):
        m=self.model;c=m.config
        if m.training:raise ValueError('cache requires eval mode')
        if ids.ndim!=2 or ids.dtype!=torch.long or ids.shape[1]<1:raise ValueError('nonempty [B,T] long input required')
        if cache is not None and len(cache)!=c.n_layers:raise ValueError('one KV pair per layer required')
        past=0 if cache is None else cache[0][0].shape[2]
        if past+ids.shape[1]>c.context_length:raise ValueError('context limit exceeded; no silent sliding window')
        positions=torch.arange(past,past+ids.shape[1],device=ids.device)
        x=m.embedding_dropout(m.token_embedding(ids)+m.position_embedding(positions)[None])
        result=[];batch,chunk,width=x.shape;hd=width//c.n_heads
        for layer,block in enumerate(m.blocks):
            att=block.attention;z=block.ln_attention(x)
            def split(proj):return proj(z).reshape(batch,chunk,c.n_heads,hd).transpose(1,2)
            q,k,v=[split(p) for p in [att.q_proj,att.k_proj,att.v_proj]]
            if cache is not None:
                oldk,oldv=cache[layer]
                expected=(batch,c.n_heads,past,hd)
                if oldk.shape!=expected or oldv.shape!=expected:raise ValueError('inconsistent cache shape')
                if oldk.dtype!=k.dtype or oldv.dtype!=v.dtype or oldk.device!=k.device or oldv.device!=v.device:raise ValueError('cache dtype/device mismatch')
                k=torch.cat([oldk,k],dim=2);v=torch.cat([oldv,v],dim=2)
            # For a new chunk, query r has absolute position past+r, not r.
            allow=torch.arange(past+chunk,device=ids.device)[None,:]<=positions[:,None]
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=allow[None,None],dropout_p=0.0)
            x=x+att.out_proj(y.transpose(1,2).contiguous().reshape(batch,chunk,width))
            x=x+block.ffn(block.ln_ffn(x));result.append((k,v))
        return m.lm_head(m.final_norm(x)),result

def sampling_distribution(logits,temperature=1.0,top_k=None,top_p=1.0):
    """Temperature -> stable top-k -> top-p -> renormalize; 1D finite logits only."""
    if logits.ndim!=1 or not bool(torch.isfinite(logits).all()):raise ValueError('finite 1D logits required')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('temperature must be positive; greedy is separate')
    if not 0<top_p<=1:raise ValueError('top_p must be in (0,1]')
    size=logits.numel()
    if top_k is not None and (not isinstance(top_k,int) or isinstance(top_k,bool) or not 1<=top_k<=size):raise ValueError('invalid top_k')
    scaled=logits/temperature;order=torch.argsort(scaled,descending=True,stable=True)
    if top_k is not None:order=order[:top_k]
    probs=torch.softmax(scaled[order],dim=0)
    if top_p<1:
        remove=torch.cat([torch.zeros(1,dtype=torch.bool,device=logits.device),torch.cumsum(probs,0)[:-1]>=top_p])
        order=order[~remove];probs=probs[~remove]
    probs=probs/probs.sum();out=torch.zeros_like(logits);out[order]=probs
    return out

class RowInt8Linear(nn.Module):
    """Real int8 weight storage, float32 dequantization + F.linear on every call; NOT an int8 kernel."""
    def __init__(self,linear):
        super().__init__();w=linear.weight.detach().float();scale=w.abs().amax(1,keepdim=True)/127
        scale=torch.where(scale==0,torch.ones_like(scale),scale)
        self.register_buffer('qweight',torch.round(w/scale).clamp(-127,127).to(torch.int8))
        self.register_buffer('scale',scale)
        self.register_buffer('bias',None if linear.bias is None else linear.bias.detach().clone())
    def forward(self,x):return F.linear(x,self.qweight.to(x.dtype)*self.scale.to(x.dtype),self.bias)

def quantize_linears(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Linear):setattr(module,name,RowInt8Linear(child))
        else:quantize_linears(child)
    return module

def tensor_storage_bytes(model):
    return sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
