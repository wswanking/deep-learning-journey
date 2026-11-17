"""Small original dual encoder; chapter 29 may reuse vision.forward_tokens read-only."""
from dataclasses import asdict,dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class DualEncoderConfig:
    image_size:int=16
    patch_size:int=4
    width:int=32
    embedding_dim:int=16
    heads:int=4
    feedforward:int=64

class TinyVisionEncoder(nn.Module):
    def __init__(self,width=32,embedding_dim=16,image_size=16,patch_size=4,heads=4,feedforward=64):
        super().__init__()
        if image_size%patch_size or width%heads:raise ValueError('divisibility required')
        self.image_size=image_size;self.width=width;self.patch_size=patch_size
        self.patch=nn.Conv2d(3,width,kernel_size=patch_size,stride=patch_size)
        self.position=nn.Parameter(torch.randn(1,(image_size//patch_size)**2,width)*.02)
        self.block=nn.TransformerEncoderLayer(width,heads,feedforward,dropout=0,batch_first=True,norm_first=True,activation='gelu')
        self.norm=nn.LayerNorm(width);self.projection=nn.Linear(width,embedding_dim)
    def forward_tokens(self,images):
        if images.ndim!=4 or tuple(images.shape[1:])!=(3,self.image_size,self.image_size):raise ValueError('images must be [B,3,16,16] for default config')
        if not images.is_floating_point():raise ValueError('float images required')
        x=self.patch(images).flatten(2).transpose(1,2)+self.position
        return self.norm(self.block(x))
    def forward(self,images):return F.normalize(self.projection(self.forward_tokens(images).mean(1)),dim=-1,eps=1e-8)

class AttributeTextEncoder(nn.Module):
    """Two fixed semantic attribute tokens, not a pretrained natural language model."""
    def __init__(self,width=32,embedding_dim=16):
        super().__init__();self.color=nn.Embedding(3,width//2);self.shape=nn.Embedding(3,width-width//2);self.projection=nn.Linear(width,embedding_dim)
    def forward(self,attributes):
        if attributes.ndim!=2 or attributes.shape[1]!=2 or attributes.dtype!=torch.long:raise ValueError('[B,2] long color/shape attributes required')
        if bool(((attributes<0)|(attributes>2)).any()):raise ValueError('attribute out of vocabulary')
        z=torch.cat([self.color(attributes[:,0]),self.shape(attributes[:,1])],-1)
        return F.normalize(self.projection(z),dim=-1,eps=1e-8)

class TinyDualEncoder(nn.Module):
    def __init__(self,config=DualEncoderConfig()):
        super().__init__();self.config=config
        self.vision=TinyVisionEncoder(**asdict(config));self.text=AttributeTextEncoder(config.width,config.embedding_dim)
    def forward(self,images,attributes):return self.vision(images),self.text(attributes)

def save_checkpoint(path,model,protocol):
    torch.save({'schema_version':1,'config':asdict(model.config),'model_state':model.state_dict(),'protocol':protocol},path)

def load_checkpoint(path,map_location='cpu'):
    obj=torch.load(path,map_location=map_location,weights_only=True)
    if obj.get('schema_version')!=1:raise ValueError('unsupported dual encoder schema')
    model=TinyDualEncoder(DualEncoderConfig(**obj['config']));model.load_state_dict(obj['model_state']);model.to(map_location).eval()
    return model,{key:obj[key] for key in ['schema_version','config','protocol']}
