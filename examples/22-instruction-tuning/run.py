"""Response-only SFT versus full-sequence loss, using chapter-21's frozen API."""
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
import torch
from torch.nn import functional as F
from sft_data import load_splits,collate,render_prompt,encode_text,fingerprints,TEMPLATE_VERSION
ROOT=Path(__file__).resolve().parents[2];sys.path.append(str(ROOT/'examples/21-tiny-language-model'))
from model import TinyLMConfig,TinyDecoderLM,load_checkpoint,inference_checkpoint
from tokenizer import decode

def require(test,message):
 if not test:raise AssertionError(message)
def state_hash(model):
 digest=hashlib.sha256()
 for name,value in model.state_dict().items():digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
 return digest.hexdigest()
def masked_loss(logits,labels):
 count=int((labels!=-100).sum())
 if count==0:raise ValueError('no supervised target bytes')
 return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')/count

def check_right_padding(model,batch):
 # Batch/slice kernels can round differently even with one thread and deterministic
 # algorithms. Keep an explicit float32 bound AND an independent strict float64
 # semantic check; do not let a wider float32 tolerance hide a masking defect.
 # https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
 measurements={}
 with torch.no_grad():
  for name,probe,tolerance in [('float32',model,1e-5),('float64',copy.deepcopy(model).double(),1e-11)]:
   probe.eval();valid=batch['attention_mask'];batched=probe(batch['input_ids'],valid);maximum=0.
   for i,e in enumerate(batch['encoded']):
    n=len(e['input_ids']);single=probe(batch['input_ids'][i:i+1,:n]);error=float((single-batched[i:i+1,:n]).abs().max());maximum=max(maximum,error)
    require(math.isfinite(error) and error<tolerance,f'padding changed real-token outputs ({name}, max_error={error}, tolerance={tolerance})')
   # Same tensor shapes isolate padding contents from batch/slice rounding.
   altered=batch['input_ids'].clone();altered[~valid]=(altered[~valid]+137)%probe.config.vocab_size
   changed=probe(altered,valid)
   require(torch.equal(batched[valid],changed[valid]),f'padding content leaked into real tokens ({name})')
   measurements[name]={'max_logit_difference':maximum,'absolute_tolerance':tolerance,'padding_content_bitexact':True}
 return {'right_padding_checks':measurements}

def mechanics(model,records):
 batch=collate(records[:8],model.config.context_length);labels=batch['labels'];valid=batch['attention_mask']
 require(bool((labels[~valid]==-100).all()),'padding targets must be ignored')
 for i,e in enumerate(batch['encoded']):
  p=e['prefix_length'];n=len(e['input_ids']);require(labels[i,p-2]==-100 and labels[i,p-1]==e['full_bytes'][p],'first answer alignment');require(int((labels[i]!=-100).sum())==3,'three supervised bytes including LF')
 # Standalone four-symbol example: Q=0, colon=1, A=2, newline=3.
 probabilities=torch.tensor([[[.25,.25,.25,.25],[.05,.05,.8,.1],[.1,.2,.2,.5]]],dtype=torch.float64);logits=probabilities.log().requires_grad_(True);target=torch.tensor([[-100,2,3]])
 loss=masked_loss(logits,target);loss.backward();expected=(-math.log(.8)-math.log(.5))/2
 require(abs(float(loss.detach())-expected)<1e-14 and float(logits.grad[0,0].abs().max())==0,'masked hand loss')
 require(torch.allclose(logits.grad[0,1],(probabilities[0,1]-F.one_hot(torch.tensor(2),4))/2,atol=1e-14,rtol=0),'softmax gradient')
 rejected=False
 try:masked_loss(logits,torch.full_like(target,-100))
 except ValueError:rejected=True
 require(rejected,'reject zero supervision')
 # Prompt output logits have no local loss gradient, but prompt context still influences answer.
 isolated=copy.deepcopy(model).train();isolated.zero_grad(set_to_none=True);output=isolated(batch['input_ids'],batch['attention_mask']);output.retain_grad();masked_loss(output,labels).backward()
 ignored_gradient=float(output.grad[labels==-100].abs().max());prompt_embedding_gradient=float(isolated.position_embedding.weight.grad[0].norm())
 require(ignored_gradient==0 and prompt_embedding_gradient>0,'loss mask is not context erasure')
 # At least two prompt lengths occur because the long verb has two additional Chinese characters.
 lengths=[len(e['input_ids']) for e in batch['encoded']];require(len(set(lengths))==2 and int((~valid).sum())>0,'actual right padding exercised')
 padding_checks=check_right_padding(model,batch)
 return {'manual_ce':expected,'autograd_ce':float(loss.detach()),'ignored_output_gradient':ignored_gradient,'prompt_position_embedding_gradient_norm':prompt_embedding_gradient,'first_answer_target_aligned':True,'supervised_bytes_per_record':3,'right_padding_input_lengths':lengths,'padding_slots':int((~valid).sum()),'zero_target_rejected':rejected,**padding_checks}

def evaluate_response(model,records):
 batch=collate(records,model.config.context_length);model.eval()
 with torch.no_grad():
  logits=model(batch['input_ids'],batch['attention_mask']);loss=float(masked_loss(logits,batch['labels']));selected=batch['labels']!=-100;correct=int(((logits.argmax(-1)==batch['labels'])&selected).sum());count=int(selected.sum())
 return {'response_mean_nll':loss,'teacher_forced_byte_accuracy':correct/count,'supervised_bytes':count}

def generate(model,record,max_new=8):
 ids=encode_text(render_prompt(record));generated=[];stopped=False;model.eval()
 with torch.no_grad():
  for _ in range(max_new):
   if len(ids)>model.config.context_length:break
   logits=model(torch.tensor([ids],dtype=torch.long));token=int(logits[0,-1].argmax());generated.append(token);ids.append(token)
   if token==10:stopped=True;break
 body=generated[:-1] if stopped else generated
 return {'id':record['id'],'task':record['task'],'instruction':record['instruction'],'expected':record['response'],'generated':decode(body),'generated_ids':generated,'stopped_on_lf':stopped,'format_valid':stopped and len(body)==2 and all(97<=v<=102 for v in body),'exact_match':stopped and body==encode_text(record['response'])}

def evaluate_generation(model,records):
 predictions=[generate(model,r) for r in records];n=len(predictions)
 return {'examples':predictions,'exact_match':sum(p['exact_match'] for p in predictions)/n,'format_valid':sum(p['format_valid'] for p in predictions)/n,'by_task':{task:sum(p['exact_match'] for p in predictions if p['task']==task)/sum(p['task']==task for p in predictions) for task in ['copy','reverse']}}

def train(base,objective,splits,updates,folder):
 model=copy.deepcopy(base);initial_hash=state_hash(model);optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=0.);gen=torch.Generator().manual_seed(2201)
 curves={'update':[],'validation_response_nll':[],'training_objective':[]};best=math.inf;best_state=None;best_update=0;batch_digest=hashlib.sha256()
 def record(update,last_loss):
  nonlocal best,best_state,best_update
  result=evaluate_response(model,splits['validation']);value=result['response_mean_nll'];curves['update'].append(update);curves['validation_response_nll'].append(value);curves['training_objective'].append(last_loss)
  if value<best:best=value;best_state=copy.deepcopy(model.state_dict());best_update=update
 record(0,None)
 for update in range(1,updates+1):
  model.train();indexes=torch.randint(len(splits['train']),(16,),generator=gen);batch_digest.update(indexes.numpy().tobytes());records=[splits['train'][i] for i in indexes.tolist()];batch=collate(records,model.config.context_length,objective)
  optimizer.zero_grad(set_to_none=True);logits=model(batch['input_ids'],batch['attention_mask']);loss=masked_loss(logits,batch['labels']);loss.backward();require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'finite training gradient');optimizer.step()
  if update%50==0 or update==updates:record(update,float(loss.detach()))
 model.load_state_dict(best_state);model.eval();checkpoint=folder/f'{objective}.pt';torch.save(inference_checkpoint(model,{'chapter':22,'objective':objective,'base_state_sha256':initial_hash,'template_version':TEMPLATE_VERSION,'data_sha256':fingerprints(),'best_update':best_update}),checkpoint)
 restored,_=load_checkpoint(checkpoint);restored.eval();probe=collate(splits['validation'],model.config.context_length)
 with torch.no_grad():require(torch.equal(model(probe['input_ids'],probe['attention_mask']),restored(probe['input_ids'],probe['attention_mask'])),'restored predictions')
 return model,{'method':objective,'initial_state_sha256':initial_hash,'batch_order_sha256':batch_digest.hexdigest(),'best_update':best_update,'best_validation_response_nll':best,'curves':curves,'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_restored_exact':True}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base-checkpoint',type=Path,default=ROOT/'build/experiments/21-tiny-language-model/base-model.pt');parser.add_argument('--updates',type=int,default=800);parser.add_argument('--check-data-only',action='store_true');parser.add_argument('--skip-figures',action='store_true');args=parser.parse_args()
 require(args.updates>0,'updates must be positive');torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True);torch.manual_seed(2202);begin=time.perf_counter();splits,metadata=load_splits();folder=ROOT/'build/experiments/22-instruction-tuning';folder.mkdir(parents=True,exist_ok=True)
 if args.check_data_only:
  model=TinyDecoderLM(TinyLMConfig());checked=mechanics(model,splits['train']);result={'status':'passed','training_performed':False,'data':metadata,'mask_checks':checked};(folder/'mask-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2));return
 if not args.base_checkpoint.exists():raise FileNotFoundError('Run chapter 21 first to create its declared base-model.pt checkpoint.')
 base,base_metadata=load_checkpoint(args.base_checkpoint);base.eval();base_hash=state_hash(base);base_file_sha=hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest();checked=mechanics(base,splits['train']);require(state_hash(base)==base_hash,'mechanics changed base parameters')
 models=[base];reports=[{'method':'base','initial_state_sha256':base_hash,'best_update':0,'best_validation_response_nll':evaluate_response(base,splits['validation'])['response_mean_nll']}]
 for objective in ['response_only','all_tokens']:
  model,report=train(base,objective,splits,args.updates,folder);report['checkpoint_file_sha256']=hashlib.sha256((ROOT/report['checkpoint']).read_bytes()).hexdigest();models.append(model);reports.append(report)
 require(reports[1]['initial_state_sha256']==reports[2]['initial_state_sha256']==base_hash,'different initial weights');require(reports[1]['batch_order_sha256']==reports[2]['batch_order_sha256'],'different training batches');require(state_hash(base)==base_hash,'frozen base changed')
 # All configurations and validation-selected models are frozen before this test phase.
 for model,report in zip(models,reports):
  report['train_response_metrics']=evaluate_response(model,splits['train']);report['test_response_metrics']=evaluate_response(model,splits['test']);report['test_generation']=evaluate_generation(model,splits['test']);report['train_generation']=evaluate_generation(model,splits['train'])
 result={'run_at':datetime.now().astimezone().isoformat(timespec='seconds'),'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','dtype':'float32','threads':1},'base':{'path':str(args.base_checkpoint),'file_sha256':base_file_sha,'state_sha256':base_hash,'metadata':base_metadata,'parameters':sum(p.numel() for p in base.parameters())},'protocol':{'updates':args.updates,'batch_size':16,'optimizer':'AdamW','learning_rate':.0015,'weight_decay':0,'betas':[.9,.999],'epsilon':1e-8,'sampling_seed':2201,'validation_interval':50,'selection':'response-only validation mean NLL for both training objectives','greedy_max_new_bytes':8,'stop_byte':10,'template_version':TEMPLATE_VERSION,'test_after_frozen':True,'one_initialization_only':True},'data':metadata,'data_sha256':fingerprints(),'mask_checks':checked,'runs':reports,'resources':{'training_and_checks_seconds':time.perf_counter()-begin,'including_import_seconds':time.perf_counter()-START,'dataset_jsonl_bytes':sum((Path(__file__).parent/'data'/f'{s}.jsonl').stat().st_size for s in ['train','validation','test']),'peak_process_memory':'not measured'}}
 (folder/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if not args.skip_figures:
  from figures import write_figures
  write_figures(result,ROOT/'docs/assets/22-instruction-tuning')
 print(json.dumps({'status':'passed','base_sha256':base_file_sha,'runs':[{'method':r['method'],'best_update':r['best_update'],'test_nll':r['test_response_metrics']['response_mean_nll'],'test_exact':r['test_generation']['exact_match'],'test_format':r['test_generation']['format_valid'],'train_exact':r['train_generation']['exact_match']} for r in reports],'resources':result['resources']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
