"""Shared chapter-21 CPU tiny decoder API, reused read-only by later tutorial examples."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int = 256
    context_length: int = 64
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.0


class CausalAttention(nn.Module):
    def __init__(self,config):
        super().__init__(); self.n_heads=config.n_heads; self.head_dim=config.d_model//config.n_heads; self.dropout=config.dropout
        self.q_proj=nn.Linear(config.d_model,config.d_model,bias=False)
        self.k_proj=nn.Linear(config.d_model,config.d_model,bias=False)
        self.v_proj=nn.Linear(config.d_model,config.d_model,bias=False)
        self.out_proj=nn.Linear(config.d_model,config.d_model,bias=False)
    def forward(self,x,key_valid):
        batch,length,width=x.shape
        def heads(projection):return projection(x).reshape(batch,length,self.n_heads,self.head_dim).transpose(1,2)
        q,k,v=[heads(p) for p in [self.q_proj,self.k_proj,self.v_proj]]
        causal=torch.ones((length,length),dtype=torch.bool,device=x.device).tril()
        allow=causal[None,None,:,:]&key_valid[:,None,None,:]
        result=F.scaled_dot_product_attention(q,k,v,attn_mask=allow,dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(result.transpose(1,2).contiguous().reshape(batch,length,width))


class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__(); self.up_proj=nn.Linear(config.d_model,config.d_ff); self.down_proj=nn.Linear(config.d_ff,config.d_model); self.dropout=nn.Dropout(config.dropout)
    def forward(self,x):return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))


class DecoderBlock(nn.Module):
    def __init__(self,config):
        super().__init__(); self.ln_attention=nn.LayerNorm(config.d_model); self.attention=CausalAttention(config); self.ln_ffn=nn.LayerNorm(config.d_model); self.ffn=FeedForward(config)
    def forward(self,x,key_valid):
        x=x+self.attention(self.ln_attention(x),key_valid)
        return x+self.ffn(self.ln_ffn(x))


class TinyDecoderLM(nn.Module):
    """Return logits [B,T,V]. attention_mask True=real key; only right padding is supported."""
    def __init__(self,config=TinyLMConfig()):
        super().__init__()
        if config.d_model%config.n_heads:raise ValueError('d_model must be divisible by n_heads')
        if min(config.context_length,config.n_layers,config.d_ff,config.vocab_size)<1:raise ValueError('configuration sizes must be positive')
        self.config=config; self.token_embedding=nn.Embedding(config.vocab_size,config.d_model); self.position_embedding=nn.Embedding(config.context_length,config.d_model)
        self.embedding_dropout=nn.Dropout(config.dropout); self.blocks=nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers)); self.final_norm=nn.LayerNorm(config.d_model); self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.apply(self._initialize)
    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=.02)
            if isinstance(module,nn.Linear) and module.bias is not None:nn.init.zeros_(module.bias)
    def forward(self,input_ids,attention_mask=None):
        if input_ids.ndim!=2 or input_ids.dtype!=torch.long:raise ValueError('input_ids must be a rank-2 torch.long tensor')
        batch,length=input_ids.shape
        if not 1<=length<=self.config.context_length:raise ValueError('input length must be between 1 and context_length')
        valid=torch.ones_like(input_ids,dtype=torch.bool) if attention_mask is None else attention_mask.to(device=input_ids.device,dtype=torch.bool)
        if valid.shape!=input_ids.shape:raise ValueError('attention_mask must match input_ids shape')
        if not bool(valid[:,0].all()) or bool((valid[:,1:]&~valid[:,:-1]).any()):raise ValueError('each sequence needs a nonempty real prefix followed only by right padding')
        positions=torch.arange(length,device=input_ids.device)
        hidden=self.embedding_dropout(self.token_embedding(input_ids)+self.position_embedding(positions)[None,:,:])
        for block in self.blocks:hidden=block(hidden,valid)
        return self.lm_head(self.final_norm(hidden))


def load_checkpoint(path: str|Path,map_location='cpu'):
    """Load a trusted chapter-21-compatible inference checkpoint; return (model, metadata)."""
    checkpoint=torch.load(path,map_location=map_location,weights_only=True)
    if checkpoint.get('schema_version')!=1:raise ValueError('unsupported tiny-LM checkpoint schema')
    if checkpoint.get('tokenizer',{}).get('type')!='utf8-byte':raise ValueError('checkpoint tokenizer must be utf8-byte')
    config=TinyLMConfig(**checkpoint['config']); model=TinyDecoderLM(config); model.load_state_dict(checkpoint['model_state']); model.to(map_location); model.eval()
    metadata={key:checkpoint[key] for key in ['schema_version','config','tokenizer','protocol'] if key in checkpoint}
    return model,metadata


def inference_checkpoint(model,protocol):
    return {'schema_version':1,'config':asdict(model.config),'model_state':model.state_dict(),'tokenizer':{'type':'utf8-byte','vocab_size':256,'special_tokens':None,'padding':'byte0 only where attention_mask=False; target ignore_index=-100'},'protocol':protocol}
