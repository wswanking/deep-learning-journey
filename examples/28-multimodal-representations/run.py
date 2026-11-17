"""Chapter 28: real CPU training of a small original contrastive dual encoder."""
from __future__ import annotations
import argparse,copy,hashlib,json,platform,time
from pathlib import Path
from datetime import datetime
START=time.perf_counter()
import numpy as np
import torch
from torch.nn import functional as F
from model import TinyDualEncoder,save_checkpoint,load_checkpoint
from data import make_data,ATTRIBUTES,CAPTIONS
ROOT=Path(__file__).resolve().parents[2]
def require(ok,message):
    if not bool(ok):raise AssertionError(message)

def contrastive_loss(images,text,temperature=.1):
    if len(images)!=len(text) or len(images)<2:raise ValueError('equal batch >=2 required')
    logits=images@text.T/temperature;labels=torch.arange(len(images))
    return .5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))

@torch.inference_mode()
def evaluate(model,images,labels,temperature=.1):
    model.eval();i=model.vision(images);t=model.text(ATTRIBUTES);cosine=i@t.T;logits=cosine/temperature
    nll=float(F.cross_entropy(logits,labels));pred=logits.argmax(1);image_top1=float((pred==labels).float().mean());image_top3=float((logits.topk(3,dim=1).indices==labels[:,None]).any(1).float().mean())
    # Multiple images per semantic caption are valid; text retrieval counts any matching class.
    ranked=cosine.T.argsort(dim=1,descending=True);retrieved_labels=labels[ranked];text_top1=float((retrieved_labels[:,0]==torch.arange(9)).float().mean());text_top3=float((retrieved_labels[:,:3]==torch.arange(9)[:,None]).any(1).float().mean())
    similarity=cosine.reshape(9,-1,9).mean(1);confusion=torch.zeros(9,9,dtype=torch.long)
    for actual,guess in zip(labels,pred):confusion[actual,guess]+=1
    return {'image_to_text_R1':image_top1,'image_to_text_R3':image_top3,'text_to_image_R1_class':text_top1,'text_to_image_R3_class':text_top3,'image_class_mean_nll':nll,'class_mean_cosine':similarity.tolist(),'confusion':confusion.tolist(),'predictions':pred.tolist(),'labels':labels.tolist(),'top3_image_indices_per_text':ranked[:,:3].tolist(),'candidate_texts':9,'candidate_images':len(labels),'multi_positive_relevance':'all images with same color and shape are relevant'}

def mechanics(model,images):
    # Identity embeddings and temperature1 produce exp(1)/(exp(1)+1).
    eye=torch.eye(2,dtype=torch.float64,requires_grad=True);text=torch.eye(2,dtype=torch.float64);loss=contrastive_loss(eye,text,1);loss.backward();expected=float(np.log1p(np.exp(-1)));require(abs(float(loss.detach())-expected)<1e-14,'hand contrastive loss')
    expected_grad=torch.tensor([[-.13447071068499755,.13447071068499755],[.13447071068499755,-.13447071068499755]],dtype=torch.float64);require(torch.allclose(eye.grad,expected_grad),'symmetric derivative')
    duplicate=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64);duplicate_loss=float(contrastive_loss(duplicate,duplicate,1));require(abs(duplicate_loss-np.log(2))<1e-14,'duplicate lower bound')
    with torch.inference_mode():
        tokens=model.vision.forward_tokens(images[:2]);require(tokens.shape==(2,16,32),'vision tokens shape');flat=F.unfold(images[:2],kernel_size=4,stride=4).transpose(1,2);projected=flat@model.vision.patch.weight.flatten(1).T+model.vision.patch.bias;conv=model.vision.patch(images[:2]).flatten(2).transpose(1,2);patch_error=float((projected-conv).abs().max());require(patch_error<1e-6,'patch linear equivalence');i,t=model(images[:9],ATTRIBUTES);require(torch.allclose(i.norm(dim=-1),torch.ones(9),atol=1e-6),'image norm');require(torch.allclose(t.norm(dim=-1),torch.ones(9),atol=1e-6),'text norm')
    model.train();model.zero_grad(set_to_none=True);sample=torch.stack([images[c*64] for c in range(9)]);i,t=model(sample,ATTRIBUTES);train_loss=contrastive_loss(i,t);train_loss.backward();require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'finite gradients');model.zero_grad(set_to_none=True);model.eval()
    return {'hand_loss':expected,'autograd_loss':float(loss.detach()),'hand_logit_gradient':expected_grad.tolist(),'duplicate_same_semantics_loss':duplicate_loss,'duplicate_expected_ln2':float(np.log(2)),'patch_projection_max_error':patch_error,'tokens_shape':list(tokens.shape),'all_gradients_finite':True,'normalized_features':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=1200);args=parser.parse_args();require(args.updates>0,'positive updates');torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(2800);begin=time.perf_counter();data,metadata=make_data();model=TinyDualEncoder();initial=copy.deepcopy(model).eval();checks=mechanics(model,data['train'][0]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);rng=torch.Generator().manual_seed(2804);best=copy.deepcopy(model.state_dict());best_val=float('inf');best_update=0;curves=[]
    for step in range(1,args.updates+1):
        model.train();indices=torch.arange(9)*64+torch.randint(0,64,(9,),generator=rng);images=data['train'][0][indices];optimizer.zero_grad(set_to_none=True);i,t=model(images,ATTRIBUTES);loss=contrastive_loss(i,t,.1);loss.backward();optimizer.step()
        if step%50==0 or step==1:
            val=evaluate(model,*data['validation']);curves.append({'update':step,'training_batch_loss':float(loss.detach()),'validation_class_nll':val['image_class_mean_nll'],'validation_R1':val['image_to_text_R1']})
            if val['image_class_mean_nll']<best_val:best_val=val['image_class_mean_nll'];best_update=step;best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best);model.eval();folder=ROOT/'build/experiments/28-multimodal-representations';folder.mkdir(parents=True,exist_ok=True);path=folder/'dual-encoder.pt';protocol={'generator':metadata,'training_seed':2800,'batch_seed':2804,'updates':args.updates,'best_update':best_update,'selection':'minimum validation image-to-text 9-class mean NLL','temperature':.1,'attribute_tokenizer':'color index 0..2 then shape index 0..2; not UTF8 LM tokens','image_range':[0,1],'image_layout':'BCHW RGB float32 16x16','scope':'9 known synthetic classes, unseen rendering seeds','optimizer':'AdamW','learning_rate':.001,'weight_decay':.001,'batch':9,'one_positive_per_semantic_class_per_batch':True};save_checkpoint(path,model,protocol);restored,meta=load_checkpoint(path)
    with torch.inference_mode():require(torch.equal(model.vision(data['validation'][0]),restored.vision(data['validation'][0])),'disk checkpoint restore')
    # Only after configuration and validation-selected checkpoint are frozen, evaluate test.
    reports={name:{split:evaluate(m,*data[split]) for split in ['train','validation','test']} for name,m in [('initial',initial),('trained',restored)]}
    with torch.inference_mode():
        images,labels=data['test'];features=restored.vision(images);text=restored.text(ATTRIBUTES);wrong=text.roll(1,0);wrong_accuracy=float(((features@wrong.T).argmax(1)==labels).float().mean());similarity=features@text.T;negated_order=similarity.flip(1);require(torch.equal(similarity.argmax(1),8-negated_order.argmax(1)),'candidate reorder metadata');representative=torch.stack([images[c*24] for c in range(9)]);representative_cos=restored.vision(representative)@text.T
    result={'run_at':datetime.now().astimezone().isoformat(timespec='seconds'),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float32','threads':1},'parameters':sum(p.numel() for p in model.parameters()),'vision_parameters':sum(p.numel() for p in model.vision.parameters()),'protocol':protocol,'data':metadata,'checks':checks|{'saved_checkpoint_restore_equal':True,'candidate_order_mapping_equal':True},'checkpoint':{'path':str(path.relative_to(ROOT)),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'curves':curves,'metrics':reports,'controls':{'cyclic_caption_mismatch_R1':wrong_accuracy,'caption_mismatch_is_evaluation_control_not_extra_training':True},'representative_test_cosine':representative_cos.tolist(),'representative_test_pixels':representative.numpy().tolist(),'resources':{'training_checks_seconds':time.perf_counter()-begin,'including_import_seconds':time.perf_counter()-START,'data_tensor_bytes':sum(x['tensor_bytes'] for x in metadata['splits'].values()),'peak_process_memory':'not measured'}}
    (folder/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from figures import write_figures
    write_figures(result,ROOT/'docs/assets/28-multimodal-representations')
    print(json.dumps({'status':'passed','parameters':result['parameters'],'vision_parameters':result['vision_parameters'],'best_update':best_update,'checks':result['checks'],'metrics':{name:{s:{k:v for k,v in m.items() if k in ['image_to_text_R1','image_to_text_R3','text_to_image_R1_class','image_class_mean_nll']} for s,m in splits.items()} for name,splits in reports.items()},'controls':result['controls'],'checkpoint':result['checkpoint'],'resources':result['resources']},ensure_ascii=True,indent=2))
if __name__=='__main__':main()
