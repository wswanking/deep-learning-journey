"""Train a 35k-parameter CPU byte LM; verify resume, causality, padding and ablations."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, platform, random, sys, time
from dataclasses import asdict, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import TinyLMConfig,TinyDecoderLM,inference_checkpoint,load_checkpoint
from tokenizer import encode,decode


def read_data():
    path=Path(__file__).with_name('corpus.jsonl'); payload=path.read_bytes(); records=[json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    groups=sorted({r['group_id'] for r in records}); generator=random.Random(42); generator.shuffle(groups)
    split={'train':groups[:16],'validation':groups[16:20],'test':groups[20:]}
    tensors={name:[torch.tensor(encode(r['text']),dtype=torch.long) for r in records if r['group_id'] in ids] for name,ids in split.items()}
    if min(len(x) for values in tensors.values() for x in values)<65:raise ValueError('all documents must contain at least 65 bytes for declared full-window budget')
    if len({hashlib.sha256(r['text'].encode()).hexdigest() for r in records})!=len(records):raise ValueError('duplicate document')
    return records,tensors,split,hashlib.sha256(payload).hexdigest()


def batch(documents,rng,batch_size=8,length=64):
    rows=[]
    for _ in range(batch_size):
        index=int(torch.randint(len(documents),(1,),generator=rng)); doc=documents[index]
        start=int(torch.randint(len(doc)-length,(1,),generator=rng)); rows.append(doc[start:start+length+1])
    sequences=torch.stack(rows)
    return sequences[:,:-1],sequences[:,1:]


@torch.no_grad()
def evaluate(model,documents):
    model.eval(); total_loss=0.; total_tokens=0
    length=model.config.context_length
    for doc in documents:
        for start in range(0,len(doc)-1,length):
            seq=doc[start:start+length+1]; count=len(seq)-1
            inputs=torch.zeros((1,length),dtype=torch.long); targets=torch.full((1,length),-100,dtype=torch.long); valid=torch.zeros((1,length),dtype=torch.bool)
            inputs[0,:count]=seq[:-1]; targets[0,:count]=seq[1:]; valid[0,:count]=True
            logits=model(inputs,valid)
            total_loss+=float(F.cross_entropy(logits.reshape(-1,256),targets.reshape(-1),ignore_index=-100,reduction='sum'));total_tokens+=count
    return {'cross_entropy_nat_per_byte':total_loss/total_tokens,'perplexity_per_byte':math.exp(total_loss/total_tokens),'predicted_bytes':total_tokens}


def run_training(config,documents,validation,protocol,steps=600,stop=None,resume=None,save=None):
    torch.manual_seed(20261115); model=TinyDecoderLM(config); optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01,foreach=False)
    rng=torch.Generator().manual_seed(17); next_step=0
    history=[{'step':0,'train':evaluate(model,documents),'validation':evaluate(model,validation)}]
    if resume is not None:
        state=torch.load(resume,weights_only=True,map_location='cpu')
        if state['config']!=asdict(config) or state['protocol']!=protocol:raise ValueError('resume config/data protocol mismatch')
        model.load_state_dict(state['model_state']); optimizer.load_state_dict(state['optimizer_state']);next_step=state['next_step'];rng.set_state(state['sampling_rng_state']);torch.set_rng_state(state['torch_rng_state']);history=state['history']
    ending=steps if stop is None else min(stop,steps)
    for step in range(next_step,ending):
        model.train(); x,y=batch(documents,rng);optimizer.zero_grad(set_to_none=True);logits=model(x);loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
        if (step+1)%50==0 or step+1==steps:history.append({'step':step+1,'train':evaluate(model,documents),'validation':evaluate(model,validation)})
    if save is not None:
        state=inference_checkpoint(model,protocol);state.update({'optimizer_state':optimizer.state_dict(),'next_step':ending,'sampling_rng_state':rng.get_state(),'torch_rng_state':torch.get_rng_state(),'history':history});torch.save(state,save)
    return model,history,rng.get_state()


@torch.no_grad()
def generate(model,prompt,mode,seed=111,max_new_bytes=96):
    model.eval(); values=encode(prompt); rng=torch.Generator().manual_seed(seed); continuation=[]
    for _ in range(max_new_bytes):
        ids=torch.tensor([values[-model.config.context_length:]],dtype=torch.long); logits=model(ids)[0,-1]
        if mode=='greedy':next_id=int(logits.argmax())
        else:next_id=int(torch.multinomial(torch.softmax(logits/.8,dim=-1),1,generator=rng))
        values.append(next_id);continuation.append(next_id)
        if next_id==10:break
    raw=bytes(continuation)
    try:raw.decode('utf-8'); utf8_valid=True
    except UnicodeDecodeError:utf8_valid=False
    return {'prompt':prompt,'mode':mode,'sampling_seed':seed if mode!='greedy' else None,'new_byte_count':len(continuation),'continuation':decode(continuation),'utf8_valid':utf8_valid,'generated_byte_ids':continuation}


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=Path('build/experiments/21-tiny-language-model'));parser.add_argument('--write-figures',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);started=time.perf_counter();args.output_dir.mkdir(parents=True,exist_ok=True)
    records,datasets,split,fingerprint=read_data();config=TinyLMConfig();protocol={'corpus_sha256':fingerprint,'split_group_ids':split,'source':'original bilingual color/object sentences','seed':20261115,'sampler_seed':17,'batch_size':8,'context_length':64,'steps':600,'predicted_bytes_processed':600*8*64,'objective':'next byte mean CE','tokenizer':'utf8-byte 0..255 no special tokens','checkpoint_selection':'fixed final step, no validation early stopping'}
    main_model,main_history,main_rng=run_training(config,datasets['train'],datasets['validation'],protocol)
    checkpoint=args.output_dir/'base-model.pt';torch.save(inference_checkpoint(main_model,protocol),checkpoint)
    print(json.dumps({'event':'base_checkpoint_ready','path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'parameter_count':sum(p.numel() for p in main_model.parameters()),'final':main_history[-1]},ensure_ascii=False),flush=True)
    resume_path=args.output_dir/'step-237-checkpoint.pt';run_training(config,datasets['train'],datasets['validation'],protocol,stop=237,save=resume_path)
    restored,resumed_history,resumed_rng=run_training(config,datasets['train'],datasets['validation'],protocol,resume=resume_path)
    maximum_error=max(float((a-b).detach().abs().max()) for a,b in zip(main_model.parameters(),restored.parameters()))
    experiments=[{'name':'main_2_layers','config':asdict(config),'parameters':sum(p.numel() for p in main_model.parameters()),'train_document_count':len(datasets['train']),'history':main_history}]
    one_config=replace(config,n_layers=1);one_model,one_history,_=run_training(one_config,datasets['train'],datasets['validation'],{**protocol,'ablation':'one decoder layer'})
    experiments.append({'name':'one_layer','config':asdict(one_config),'parameters':sum(p.numel() for p in one_model.parameters()),'train_document_count':len(datasets['train']),'history':one_history})
    subset=[torch.tensor(encode(row['text']),dtype=torch.long) for row in records if row['group_id'] in split['train'][:2]]
    small_model,small_history,_=run_training(config,subset,datasets['validation'],{**protocol,'ablation':'only first two training groups, four documents'})
    experiments.append({'name':'four_documents','config':asdict(config),'parameters':sum(p.numel() for p in small_model.parameters()),'train_document_count':len(subset),'history':small_history})
    # Test is evaluated once after all configurations and the final main checkpoint are fixed.
    test_result=evaluate(main_model,datasets['test'])
    loaded,metadata=load_checkpoint(checkpoint); probe=torch.tensor([encode('The red square ')],dtype=torch.long)
    with torch.no_grad(): reload_error=float((main_model(probe)-loaded(probe)).abs().max())
    prompt_results=[generate(loaded,prompt,mode) for prompt in ['The red ','红色','计算3+5='] for mode in ['greedy','sample_temperature_0.8']]
    original=torch.tensor([[65,66,67,68,69]],dtype=torch.long);changed=original.clone();changed[:,3:]=torch.tensor([[120,121]])
    with torch.no_grad():
        original_logits=loaded(original);changed_logits=loaded(changed);causal_error=float((original_logits[:,:3]-changed_logits[:,:3]).abs().max())
        padded=torch.tensor([[65,66,0,0]],dtype=torch.long);valid=torch.tensor([[True,True,False,False]]);padded_changed=padded.clone();padded_changed[:,2:]=200
        pad_error=float((loaded(padded,valid)[:,:2]-loaded(padded_changed,valid)[:,:2]).abs().max())
    checks=[]
    def require(condition,name):
        if not bool(condition):raise RuntimeError(name)
        checks.append(name)
    require(sum(p.numel() for p in main_model.parameters())==35328,'main_parameter_count_35328')
    require(maximum_error==0,'resume_final_parameters_exactly_equal')
    require(resumed_history==main_history,'resume_evaluation_history_exactly_equal')
    require(torch.equal(main_rng,resumed_rng),'resume_sampler_state_exactly_equal')
    require(reload_error==0,'base_inference_reload_logits_equal')
    require(causal_error==0,'future_bytes_do_not_change_prefix_logits')
    require(pad_error==0,'padded_byte_values_do_not_change_real_prefix_logits')
    require(not(set(split['train'])&set(split['validation']) or set(split['train'])&set(split['test']) or set(split['validation'])&set(split['test'])),'groups_disjoint')
    require(len(records)==48 and len(set(r['text'] for r in records))==48,'corpus_48_unique_documents')
    require(decode(encode('红色 square'))=='红色 square','utf8_byte_tokenizer_roundtrip')
    for result in experiments:
        require(result['history'][-1]['train']['cross_entropy_nat_per_byte']<result['history'][0]['train']['cross_entropy_nat_per_byte'],f'training_loss_decreases_{result["name"]}')
        require(all(math.isfinite(row['train']['cross_entropy_nat_per_byte']) and math.isfinite(row['validation']['cross_entropy_nat_per_byte']) for row in result['history']),f'finite_metrics_{result["name"]}')
    reject=False
    try:loaded(torch.tensor([[0,65]]),torch.tensor([[False,True]]))
    except ValueError:reject=True
    require(reject,'unsupported_left_padding_explicitly_rejected')
    report={'chapter':'21','collected_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'python_executable':sys.executable,'device':'CPU','dtype':'float32','threads':1},'protocol':protocol,'data':{'documents':48,'groups':24,'train_documents':32,'validation_documents':8,'test_documents':8,'total_corpus_bytes':sum(len(encode(r['text'])) for r in records),'corpus_file':'examples/21-tiny-language-model/corpus.jsonl','test_protocol':'one main-model evaluation after fixed training/ablations; not used to choose checkpoint'},'experiments':experiments,'main_test':test_result,'generation':prompt_results,'resume':{'saved_after_step':237,'total_steps':600,'maximum_parameter_absolute_error':maximum_error,'history_equal':resumed_history==main_history,'sampler_equal':bool(torch.equal(main_rng,resumed_rng)),'state':['model_state','optimizer_state','next_step','sampling_rng_state','torch_rng_state','history','config','protocol']},'checks':{'status':'passed','count':len(checks),'names':checks},'checkpoint':{'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'schema_version':1,'parameter_count':35328,'inference_reload_max_error':reload_error},'causality':{'prefix_max_error_after_future_change':causal_error,'prefix_max_error_after_padding_change':pad_error},'resources':{'elapsed_seconds_excluding_imports':time.perf_counter()-started,'main_parameter_numeric_bytes':35328*4,'estimated_main_parameter_and_adam_moments_numeric_bytes':35328*4*3,'memory_scope':'parameter payload and two optimizer moments only, excludes gradients/activations/framework; no process peak measured'}}
    out=args.output_dir/'results.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.write_figures:
        from figures import write_figures
        write_figures(report,Path(__file__).resolve().parents[2]/'docs/assets/21-tiny-language-model')
    compact={key:report[key] for key in ['collected_at','environment','data','protocol','main_test','generation','resume','checks','checkpoint','causality','resources']};compact['final_configurations']=[{'name':r['name'],'parameters':r['parameters'],'final':r['history'][-1]} for r in experiments]; print(json.dumps(compact,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
