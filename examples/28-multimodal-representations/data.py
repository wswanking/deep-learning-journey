"""Original procedural RGB shapes; deterministic split seeds and no external assets."""
import hashlib,json
import numpy as np
import torch
COLORS=['红色','绿色','蓝色'];SHAPES=['圆形','方形','十字'];PALETTE=np.array([[.88,.12,.10],[.10,.75,.18],[.12,.22,.90]],dtype=np.float32)
ATTRIBUTES=torch.tensor([[c,s] for c in range(3) for s in range(3)],dtype=torch.long)
CAPTIONS=[COLORS[c]+SHAPES[s] for c,s in ATTRIBUTES.tolist()]

def render(color,shape,rng):
    yy,xx=np.mgrid[0:16,0:16].astype(np.float32);cx,cy=rng.uniform(6.5,8.5,2);radius=float(rng.uniform(3.5,4.8));x=xx-cx;y=yy-cy
    if shape==0:mask=x*x+y*y<=radius*radius
    elif shape==1:mask=(np.abs(x)<=radius)&(np.abs(y)<=radius)
    else:mask=((np.abs(x)<=radius*.37)&(np.abs(y)<=radius))|((np.abs(y)<=radius*.37)&(np.abs(x)<=radius))
    background=float(rng.uniform(.015,.065));rgb=np.full((16,16,3),background,dtype=np.float32);rgb[mask]=PALETTE[color]*float(rng.uniform(.85,1.08));rgb+=rng.normal(0,.012,rgb.shape).astype(np.float32)
    return np.clip(rgb,0,1).transpose(2,0,1).copy()

def make_split(seed,views_per_class):
    rng=np.random.default_rng(seed);images=[];labels=[]
    for c,s in ATTRIBUTES.tolist():
        for _ in range(views_per_class):images.append(render(c,s,rng));labels.append(c*3+s)
    return torch.from_numpy(np.stack(images)),torch.tensor(labels,dtype=torch.long)

def make_data():
    specs={'train':(2801,64),'validation':(2802,16),'test':(2803,24)};data={};meta={}
    hashes={}
    for name,(seed,views) in specs.items():
        images,labels=make_split(seed,views);data[name]=(images,labels);raw=images.numpy().tobytes()+labels.numpy().tobytes();hashes[name]={hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in images}
        meta[name]={'seed':seed,'views_per_class':views,'images':len(labels),'tensor_sha256':hashlib.sha256(raw).hexdigest(),'tensor_bytes':images.numel()*images.element_size()+labels.numel()*labels.element_size()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if hashes[a]&hashes[b]:raise AssertionError('exact image leakage')
    return data,{'generator_version':'rgb-shapes-v1','colors':COLORS,'shapes':SHAPES,'class_captions':CAPTIONS,'attributes':ATTRIBUTES.tolist(),'splits':meta,'exact_pixel_overlap':False,'license':'original procedural data, same repository license','scope':'same 9 concepts, independent rendering seeds; not unseen classes'}
