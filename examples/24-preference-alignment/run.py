"""CPU offline DPO from the frozen chapter-22 SFT model, with adverse-label control."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,platform,shutil,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import load_splits,prompt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(1,str(ROOT/'examples/21-tiny-language-model'))
from model import load_checkpoint,inference_checkpoint
from tokenizer import encode,decode

def state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def collate(records,context_length):
    encoded=[]
    for row in records:
        prefix=encode(prompt(row))
        for key in ['chosen','rejected']:
            full=prefix+encode(row[key]+'\n');encoded.append((full,len(prefix)))
    length=max(len(full)-1 for full,p in encoded)
    if length>context_length:raise ValueError('example longer than context; truncation is not permitted')
    ids=torch.zeros((len(encoded),length),dtype=torch.long);targets=torch.zeros_like(ids);valid=torch.zeros_like(ids,dtype=torch.bool);response=torch.zeros_like(ids,dtype=torch.bool)
    for i,(full,prefix_length) in enumerate(encoded):
        n=len(full)-1;ids[i,:n]=torch.tensor(full[:-1]);targets[i,:n]=torch.tensor(full[1:]);valid[i,:n]=True;response[i,prefix_length-1:n]=True
    if not bool((response.sum(-1)==2).all()):raise ValueError('each candidate must have exactly one ASCII letter plus LF')
    return {'ids':ids,'targets':targets,'attention_mask':valid,'response_mask':response}

def response_logprobs(model,batch):
    logits=model(batch['ids'],batch['attention_mask']);values=F.log_softmax(logits,dim=-1).gather(-1,batch['targets'].unsqueeze(-1)).squeeze(-1)
    return (values*batch['response_mask']).sum(-1).reshape(-1,2)

@torch.no_grad()
def evaluate(model,records,reference_logs,beta,flip=False):
    model.eval();logs=response_logprobs(model,collate(records,model.config.context_length));delta=(logs[:,0]-logs[:,1])-(reference_logs[:,0]-reference_logs[:,1]);margin=beta*delta
    candidate_logq=F.log_softmax(logs,dim=-1);candidate_logref=F.log_softmax(reference_logs,dim=-1);subset_kl=(candidate_logq.exp()*(candidate_logq-candidate_logref)).sum(-1).mean()
    score=lambda values:float(((values>1e-7).float()+.5*(values.abs()<=1e-7)).mean())
    return {'dpo_loss_true_labels':float(F.softplus(-margin).mean()),'optimization_loss':float(F.softplus(margin if flip else -margin).mean()),'raw_candidate_pair_accuracy':score(logs[:,0]-logs[:,1]),'relative_pair_accuracy':score(delta),'mean_margin':float(margin.mean()),'chosen_mean_logprob':float(logs[:,0].mean()),'rejected_mean_logprob':float(logs[:,1].mean()),'two_candidate_conditional_kl':float(subset_kl),'pair_logprobs':logs.tolist()}

@torch.no_grad()
def generation(model,records):
    model.eval();rows=[]
    for row in records:
        values=encode(prompt(row));new=[];stopped=False
        for _ in range(4):
            if len(values)>model.config.context_length:break
            next_id=int(model(torch.tensor([values],dtype=torch.long))[0,-1].argmax());new.append(next_id);values.append(next_id)
            if next_id==10:stopped=True;break
        body=new[:-1] if stopped else new;expected=encode(row['chosen']);rows.append({'id':row['id'],'instruction':row['instruction'],'chosen':row['chosen'],'rejected':row['rejected'],'generated':decode(body),'generated_ids':new,'stopped_on_lf':stopped,'format_valid':stopped and len(body)==1 and body[0] in range(103,109),'exact_match':stopped and body==expected})
    return {'exact_match':sum(r['exact_match'] for r in rows)/len(rows),'format_valid':sum(r['format_valid'] for r in rows)/len(rows),'examples':rows}

def hand_checks(require):
    beta=.5;chosen=torch.tensor(math.log(.6),dtype=torch.float64,requires_grad=True);rejected=torch.tensor(math.log(.3),dtype=torch.float64,requires_grad=True);margin=beta*((chosen-rejected)-(math.log(.4)-math.log(.4)));loss=F.softplus(-margin);loss.backward();expected=.5*math.log(2);derivative=-beta/(1+math.exp(expected))
    require(abs(float(margin.detach())-expected)<1e-14,'manual_dpo_margin');require(abs(float(loss.detach())-math.log1p(math.exp(-expected)))<1e-14,'stable_manual_dpo_loss');require(abs(float(chosen.grad)-derivative)<1e-14 and abs(float(rejected.grad)+derivative)<1e-14,'manual_dpo_gradients')
    rewards=torch.tensor([1.,0.],dtype=torch.float64);ref=torch.tensor([.5,.5],dtype=torch.float64);optimal=F.softmax(ref.log()+rewards,dim=0);kl=(optimal*(optimal.log()-ref.log())).sum();objective=(optimal*rewards).sum()-kl;log_z=torch.log((ref*rewards.exp()).sum())
    require(abs(float(objective-log_z))<1e-14,'kl_reward_optimum_identity')
    return {'beta':beta,'margin':float(margin.detach()),'loss':float(loss.detach()),'preference_probability':1/(1+math.exp(-expected)),'chosen_logprob_derivative':float(chosen.grad),'rejected_logprob_derivative':float(rejected.grad),'two_response_optimum':optimal.tolist(),'two_response_kl':float(kl),'two_response_objective':float(objective)}

def train(base,reference,splits,reference_logs,beta,flip,folder,name,protocol):
    model=copy.deepcopy(base);optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=0.,foreach=False);rng=torch.Generator().manual_seed(2401);history=[]
    def record(step):
        row={'step':step}
        for split in ['train','validation']:row[split]=evaluate(model,splits[split],reference_logs[split],beta,flip)
        history.append(row)
    record(0)
    for step in range(1,301):
        indexes=torch.randint(len(splits['train']),(12,),generator=rng);records=[splits['train'][i] for i in indexes.tolist()];batch=collate(records,model.config.context_length);model.train();optimizer.zero_grad(set_to_none=True);logs=response_logprobs(model,batch);base_logs=reference_logs['train'][indexes];margin=beta*((logs[:,0]-logs[:,1])-(base_logs[:,0]-base_logs[:,1]));loss=F.softplus(margin if flip else -margin).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
        if step%25==0:record(step)
    target=folder/f'{name}.pt';torch.save(inference_checkpoint(model,{**protocol,'method':name,'beta':beta,'flipped_training_labels':flip}),target)
    return model,{'name':name,'beta':beta,'flipped_training_labels':flip,'history':history,'checkpoint':str(target.relative_to(ROOT))}

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sft-checkpoint',type=Path,default=ROOT/'build/experiments/22-instruction-tuning/response_only.pt');parser.add_argument('--write-figures',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(2402);started=time.perf_counter();folder=ROOT/'build/experiments/24-preference-alignment';folder.mkdir(parents=True,exist_ok=True)
    if not args.sft_checkpoint.exists():raise FileNotFoundError('Run chapters21 and22 first; no substitute random base will be created.')
    frozen=folder/'sft-used.pt';shutil.copyfile(args.sft_checkpoint,frozen);base,metadata=load_checkpoint(frozen);reference=copy.deepcopy(base).eval();reference.requires_grad_(False);initial_hash=state_hash(reference);splits,data=load_splits();reference_logs={}
    with torch.no_grad():
        for split,records in splits.items():reference_logs[split]=response_logprobs(reference,collate(records,reference.config.context_length))
    source_sha=hashlib.sha256(frozen.read_bytes()).hexdigest();protocol={'chapter':24,'sft_file_sha256':source_sha,'sft_state_sha256':initial_hash,'sft_protocol':metadata['protocol'],'data':data,'learning_rate':.0005,'optimizer':'AdamW weight_decay0 foreachFalse','steps':300,'batch_pairs':12,'pairs_processed_per_configuration':3600,'response_logprob':'sum over one answer byte and LF; prompt/padding excluded','reference':'frozen SFT clone, cached logprobs','checkpoint_selection':'fixed step300; validation diagnostic only','test_protocol':'all configurations declared before training; evaluate frozen final models once, no subsequent tuning'}
    checks=[]
    def require(condition,name):
        if not bool(condition):raise RuntimeError(name)
        checks.append(name)
    hand=hand_checks(require);batch=collate(splits['train'][:3],base.config.context_length);require(bool((batch['response_mask'].sum(-1)==2).all()),'exactly_two_response_targets_per_candidate');require(not bool((batch['response_mask']&~batch['attention_mask']).any()),'padding_never_scored');require(reference_logs['train'].requires_grad==False,'reference_cache_detached')
    with torch.no_grad():
        lp=response_logprobs(base,batch);manual=[]
        for row in splits['train'][:3]:
            for key in ['chosen','rejected']:
                p=encode(prompt(row));full=p+encode(row[key]+'\n');total=0.
                for t in range(len(p)-1,len(full)-1):total+=float(F.log_softmax(base(torch.tensor([full[:t+1]],dtype=torch.long))[0,-1],dim=-1)[full[t+1]])
                manual.append(total)
        scoring_error=float((lp.reshape(-1)-torch.tensor(manual)).abs().max())
    require(scoring_error<1e-5,'teacher_forced_response_sum_matches_prefix_scoring')
    padded={key:torch.cat([value,torch.zeros((value.shape[0],2),dtype=value.dtype)],dim=1) for key,value in batch.items()};padded['ids'][:,-2:]=200
    with torch.no_grad():padding_error=float((response_logprobs(base,padded)-lp).abs().max())
    require(padding_error<1e-5,'real_response_scores_invariant_to_actual_right_padding');require(bool(torch.isfinite(F.softplus(torch.tensor([-1000.,1000.],dtype=torch.float64))).all()),'extreme_dpo_margins_remain_finite')
    models=[base];experiments=[{'name':'sft_reference','beta':1.,'flipped_training_labels':False,'history':[{'step':0,'train':evaluate(base,splits['train'],reference_logs['train'],1.),'validation':evaluate(base,splits['validation'],reference_logs['validation'],1.)}]}]
    for name,beta,flip in [('dpo_beta1',1.,False),('dpo_beta0_1',.1,False),('flipped_labels',1.,True)]:
        model,result=train(base,reference,splits,reference_logs,beta,flip,folder,name,protocol);models.append(model);experiments.append(result)
        require(result['history'][-1]['train']['optimization_loss']<result['history'][0]['train']['optimization_loss'],f'own_training_objective_decreases_{name}')
        require(all(math.isfinite(r['train']['optimization_loss']) and math.isfinite(r['validation']['optimization_loss']) for r in result['history']),f'finite_losses_{name}')
    # All final policies and fixed budgets are locked before these held-out evaluations.
    for model,result in zip(models,experiments):
        result['test']=evaluate(model,splits['test'],reference_logs['test'],result['beta'],result['flipped_training_labels']);result['generation']=generation(model,splits['test'])
    require(state_hash(reference)==initial_hash and state_hash(base)==initial_hash,'reference_and_source_parameters_unchanged');require(all(p.grad is None for p in reference.parameters()),'reference_gradients_absent');require(hashlib.sha256(args.sft_checkpoint.read_bytes()).hexdigest()==source_sha,'sft_source_file_unchanged');require(all(abs(r['train']['optimization_loss']-math.log(2))<1e-6 for e in experiments[1:] for r in e['history'][:1]),'initial_equal_policy_reference_loss_log2')
    loaded,_=load_checkpoint(folder/'dpo_beta1.pt');trained=models[1];probe=collate(splits['validation'],loaded.config.context_length)
    with torch.no_grad():reload_error=float((loaded(probe['ids'],probe['attention_mask'])-trained(probe['ids'],probe['attention_mask'])).abs().max())
    require(reload_error==0,'dpo_checkpoint_reload_logits_equal');sft_prompts={json.loads(line)['instruction'] for p in (ROOT/'examples/22-instruction-tuning/data').glob('*.jsonl') for line in p.read_text(encoding='utf-8').splitlines() if line.strip()};require(all(row['instruction'] not in sft_prompts for rows in splits.values() for row in rows),'preference_prompts_disjoint_from_sft_data')
    report={'chapter':'24','collected_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU','threads':1,'dtype':'float32'},'protocol':protocol,'source':{'sft_file':str(args.sft_checkpoint.relative_to(ROOT)),'frozen_copy':str(frozen.relative_to(ROOT)),'file_sha256':source_sha,'state_sha256':initial_hash,'metadata':metadata},'hand':hand,'scoring_check':{'maximum_logprob_difference':scoring_error,'response_targets':2,'right_padding_max_logprob_difference':padding_error,'reload_max_logit_difference':reload_error},'experiments':experiments,'checks':{'status':'passed','count':len(checks),'names':checks},'resources':{'elapsed_seconds_excluding_imports':time.perf_counter()-started,'parameters_per_policy':sum(p.numel() for p in base.parameters()),'parameter_numeric_bytes_per_policy':sum(p.numel()*p.element_size() for p in base.parameters()),'scope':'numeric parameter payload only; no process peak measured'}}
    (folder/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.write_figures:
        from figures import write_figures
        write_figures(report,ROOT/'docs/assets/24-preference-alignment')
    compact={key:report[key] for key in ['collected_at','environment','hand','scoring_check','checks','resources']};compact['sft_file_sha256']=source_sha;compact['final']=[{'name':e['name'],'train':e['history'][-1]['train'],'validation':e['history'][-1]['validation'],'test':e['test'],'generation':{'exact_match':e['generation']['exact_match'],'format_valid':e['generation']['format_valid']}} for e in experiments];print(json.dumps(compact,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
