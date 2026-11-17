"""Rebuild a CPU text/evidence project in isolated outputs, or replay frozen project models."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,math,os,platform,subprocess,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(ROOT/'examples/21-tiny-language-model'));sys.path.insert(2,str(ROOT/'examples/22-instruction-tuning'))
from model import TinyLMConfig,load_checkpoint,inference_checkpoint
from tokenizer import encode,decode

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def emit(stage,**values):print(json.dumps({'stage':stage,**values},ensure_ascii=False),flush=True)

@torch.no_grad()
def generate(model,text,max_bytes):
    values=encode(text)
    if len(values)>model.config.context_length:raise ValueError('initial prompt would lose evidence; no silent truncation')
    result=[]
    for _ in range(max_bytes):
        token=int(model(torch.tensor([values[-64:]],dtype=torch.long))[0,-1].argmax());result.append(token);values.append(token)
        if token==10:break
    raw=bytes(result)
    try:raw.decode('utf-8');valid=True
    except UnicodeDecodeError:valid=False
    return {'prompt':text,'new_byte_ids':result,'text':decode(result),'utf8_valid':valid,'stopped_on_lf':bool(result and result[-1]==10)}

def source_files():
    values=[]
    for chapter in ['21-tiny-language-model','22-instruction-tuning','26-retrieval-augmented-generation','27-model-evaluation','30-capstone']:
        for path in (ROOT/'examples'/chapter).rglob('*'):
            if path.is_file() and path.suffix in ['.py','.mjs','.json','.jsonl'] and '__pycache__' not in path.parts:values.append(path)
    for name in ['requirements/cpu-lock.txt','package-lock.json','scripts/build.mjs','docs/styles/book.css','docs/book.json']:
        path=ROOT/name
        if path.is_file():values.append(path)
    return sorted(set(values))

def hand_budget(config):
    pre=config['pretraining']['steps']*config['pretraining']['batch_size']*config['model']['context_length'];sft=config['sft']['updates']*config['sft']['batch_size']*3
    return {'pretrain_main_prediction_events':pre,'resume_verification_extra_prediction_events':pre,'sft_answer_events':sft,'all_optimization_target_events':2*pre+sft,'scope':'counts include repeated data; objectives differ, so their total is a work ledger, not a shared quality metric'}

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['rebuild','replay'],default='rebuild');parser.add_argument('--skip-export',action='store_true');parser.add_argument('--require-quality',action='store_true');args=parser.parse_args();config=json.loads((HERE/'config.json').read_text(encoding='utf-8'));folder=ROOT/config['outputs'];folder.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.set_default_dtype(torch.float32);started=time.perf_counter();checks=[]
    def require(test,name):
        if not bool(test):raise RuntimeError(name)
        checks.append(name)
    fixed={'schema_version':1,'model':{'vocab_size':256,'context_length':64,'d_model':32,'n_heads':4,'n_layers':2,'d_ff':64,'dropout':0.0},'pretraining':{'steps':600,'batch_size':8,'learning_rate':.003,'weight_decay':.01,'seed':20261115,'sampler_seed':17,'resume_after_step':237},'sft':{'updates':800,'objective':'response_only','batch_size':16,'learning_rate':.0015,'weight_decay':0.0,'sampler_seed':2201,'validation_interval':50},'evaluation':{'language_max_new_bytes':8,'rag_max_new_bytes':16,'stop_byte':10,'template':'instruction-response-utf8-v1'}}
    if any(config[key]!=value for key,value in fixed.items()) or config['retrieval']['width_candidates']!=[36,72,144] or config['retrieval']['overlap_candidates']!=[0,12] or config['retrieval']['k']!=3 or config['retrieval']['context_slots']!=2 or config['outputs']!='build/experiments/30-capstone':raise ValueError('recorded teaching protocol is fixed; changing it needs implementation and fresh acceptance')
    if any(not 0<=float(value)<=1 for key,value in config['review_thresholds'].items() if key.endswith('_min')):raise ValueError('probability review thresholds must lie in [0,1]')
    frozen_manifest=None
    if args.mode=='replay':
        frozen_manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        require(all(sha(ROOT/path)==digest for key in ['sources','artifacts'] for path,digest in frozen_manifest[key].items()),'cold_rebuild_manifest_matches_before_replay')
    protected=[ROOT/'build/experiments'/relative for relative in ['21-tiny-language-model/base-model.pt','21-tiny-language-model/results.json','22-instruction-tuning/response_only.pt','22-instruction-tuning/results.json','26-retrieval-augmented-generation/results.json','27-model-evaluation/results.json']];protected_before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.exists()}
    pre=import_file('capstone30_pretraining',ROOT/'examples/21-tiny-language-model/run.py');sft=import_file('capstone30_sft',ROOT/'examples/22-instruction-tuning/run.py');sd=import_file('capstone30_rag_data',ROOT/'examples/26-retrieval-augmented-generation/data.py');retrieval=import_file('capstone30_retrieval',ROOT/'examples/26-retrieval-augmented-generation/retrieval.py')
    ed=import_file('capstone30_evaluation_data',ROOT/'examples/27-model-evaluation/data.py');previous_data=sys.modules.get('data');sys.modules['data']=ed;ev=import_file('capstone30_evaluation',ROOT/'examples/27-model-evaluation/run.py')
    if previous_data is None:sys.modules.pop('data',None)
    else:sys.modules['data']=previous_data
    base_path=folder/'base-model.pt';sft_folder=folder/'sft';sft_folder.mkdir(exist_ok=True);training_path=folder/'training-record.json';durations={}
    if args.mode=='rebuild':
        records,datasets,split,fingerprint=pre.read_data();model_config=TinyLMConfig(**config['model']);protocol={'chapter':30,'corpus_sha256':fingerprint,'split_group_ids':split,'source':'chapter21 original bilingual corpus','seed':20261115,'sampler_seed':17,'batch_size':8,'context_length':64,'steps':600,'objective':'next byte mean CE','checkpoint_selection':'fixed final step600; validation diagnostic only'};emit('pretraining_start',updates=600);begin=time.perf_counter();base,history,rng=pre.run_training(model_config,datasets['train'],datasets['validation'],protocol);torch.save(inference_checkpoint(base,protocol),base_path);durations['main_pretraining_seconds']=time.perf_counter()-begin
        emit('resume_verification_start',saved_step=237,final_step=600);begin=time.perf_counter();resume_file=folder/'step-237-training.pt';pre.run_training(model_config,datasets['train'],datasets['validation'],protocol,stop=237,save=resume_file);restored,resumed_history,resumed_rng=pre.run_training(model_config,datasets['train'],datasets['validation'],protocol,resume=resume_file);resume_error=max(float((a-b).detach().abs().max()) for a,b in zip(base.parameters(),restored.parameters()));require(resume_error==0 and history==resumed_history and torch.equal(rng,resumed_rng),'independent_partial_resume_matches_continuous_training');durations['resume_verification_seconds']=time.perf_counter()-begin
        emit('sft_start',updates=800,objective='response_only');splits,sft_data=sft.load_splits();initial_hash=sft.state_hash(base);begin=time.perf_counter();fine,sft_training=sft.train(base,'response_only',splits,800,sft_folder);durations['sft_training_seconds']=time.perf_counter()-begin;require(sft.state_hash(base)==initial_hash,'sft_did_not_change_base_object');training={'pretraining_protocol':protocol,'pretraining_history':history,'resume':{'step':237,'maximum_parameter_absolute_error':resume_error,'history_equal':history==resumed_history,'sampling_state_equal':bool(torch.equal(rng,resumed_rng))},'sft':sft_training,'sft_data':sft_data,'stage_durations':durations,'optimization_event_budget':hand_budget(config)};write_json(training_path,training)
    else:
        for path in [base_path,sft_folder/'response_only.pt',training_path]:
            if not path.exists():raise FileNotFoundError('Run --mode rebuild once before replay: '+str(path))
        training=json.loads(training_path.read_text(encoding='utf-8'));base,_=load_checkpoint(base_path);fine,_=load_checkpoint(sft_folder/'response_only.pt');emit('replay_existing_project',training_performed=False)
    base.eval();fine.eval();require(sum(p.numel() for p in base.parameters())==35328 and sum(p.numel() for p in fine.parameters())==35328,'model_parameter_count');records,evaluation_data=ed.load_records();summaries={};predictions={}
    emit('fixed_language_evaluation',cases=len(records));begin=time.perf_counter()
    for name,model in [('base',base),('sft',fine)]:
        before_hash=sft.state_hash(model);rows={};scores={}
        with torch.no_grad():
            for _ in range(3):model(torch.tensor([[65,66]],dtype=torch.long))
        for row in records:
            start=time.perf_counter();value=ev.predict(model,row);value['latency_ms']=(time.perf_counter()-start)*1000;rows[row['id']]=value;scores[row['id']]=ev.target_nll(model,row)
        summaries[name]=ev.summarize(records,rows,scores);predictions[name]=rows;require(sft.state_hash(model)==before_hash,'evaluation_parameters_unchanged_'+name)
    baseline={}
    for row in records:
        start=time.perf_counter();value=ev.score_output(row,encode(row['instruction'].rsplit(' ',1)[1]+'\n'));value['latency_ms']=(time.perf_counter()-start)*1000;baseline[row['id']]=value
    summaries['echo_baseline']=ev.summarize(records,baseline);predictions['echo_baseline']=baseline;durations['language_evaluation_seconds']=time.perf_counter()-begin
    emit('retrieval_selection_start',candidates=6);begin=time.perf_counter();documents,validation,test=sd.dataset();candidates=[]
    for width in config['retrieval']['width_candidates']:
        for overlap in config['retrieval']['overlap_candidates']:
            chunks=retrieval.split_documents(documents,width,overlap);index=retrieval.ExactIndex(chunks);result=retrieval.evaluate(index,validation,3,limit=2);candidates.append({'width':width,'overlap':overlap,'indexed_characters':sum(len(c['text']) for c in chunks),'chunks':len(chunks),'validation_em':result['answer_em']})
    index_path=folder/('index.npz' if args.mode=='rebuild' else 'replay-index.npz')
    selected=sorted(candidates,key=lambda r:(-r['validation_em'],r['indexed_characters'],r['width'],r['overlap']))[0];chunks=retrieval.split_documents(documents,selected['width'],selected['overlap']);index=retrieval.ExactIndex(chunks);rule_result=retrieval.evaluate(index,test,3,limit=2);index.save(index_path);saved=np.load(index_path,allow_pickle=False);require(np.array_equal(saved['matrix'],index.matrix) and np.array_equal(saved['idf'],index.idf) and saved['vocabulary'].tolist()==index.vocabulary,'retrieval_index_numeric_reload_equal');docmap={r['id']:r['text'] for r in documents}
    for row in rule_result['traces']:
        if row['citation']:
            c=row['citation'];require(docmap[c['doc_id']][c['start']:c['end']]==c['text'],'citation_original_slice_'+row['id'])
    durations['retrieval_seconds']=time.perf_counter()-begin;emit('neural_evidence_probe_start',answerable_questions=12);probes={}
    for name,model in [('base',base),('sft',fine)]:
        rows=[]
        for trace in rule_result['traces']:
            if not trace['answerable']:continue
            sentence=trace['citation']['text'] if trace['citation'] else '';closed=generate(model,'q:'+trace['question']+'\na:',16);grounded=generate(model,'d:'+sentence+'\nq:'+trace['question']+'\na:',16)
            for value in [closed,grounded]:value['answer_em']=value['text'].strip()==trace['answer']
            rows.append({'id':trace['id'],'question':trace['question'],'gold':trace['answer'],'evidence':sentence,'closed_book':closed,'with_evidence':grounded})
        probes[name]={'closed_book_em':float(np.mean([r['closed_book']['answer_em'] for r in rows])),'with_evidence_em':float(np.mean([r['with_evidence']['answer_em'] for r in rows])),'pairs':rows}
    require(all(p['evidence'] and len(p['with_evidence']['prompt'].encode())<=64 for value in probes.values() for p in value['pairs']),'all_neural_probes_receive_untruncated_complete_evidence')
    loaded,_=load_checkpoint(base_path);loaded_sft,_=load_checkpoint(sft_folder/'response_only.pt');probe=torch.tensor([encode('The red ')],dtype=torch.long)
    with torch.no_grad():base_error=float((base(probe)-loaded(probe)).abs().max());sft_error=float((fine(probe)-loaded_sft(probe)).abs().max())
    require(base_error==0 and sft_error==0,'independent_checkpoint_reload_logits_equal');require(all(sha(ROOT/name)==value for name,value in protected_before.items()),'previous_chapter_binary_and_reports_unchanged');require(all(sum(m['failure_categories'].values())==m['count'] for group in summaries.values() for m in group.values()),'failure_counts_cover_all_language_examples')
    gates=[{'name':'SFT existing test EM','value':summaries['sft']['existing_22_test']['exact_match'],'minimum':config['review_thresholds']['sft_existing_em_min']},{'name':'SFT new alphabet EM','value':summaries['sft']['new_alphabet_diagnostic']['exact_match'],'minimum':config['review_thresholds']['sft_new_alphabet_em_min']},{'name':'neural evidence answer EM','value':probes['sft']['with_evidence_em'],'minimum':config['review_thresholds']['neural_rag_em_min']},{'name':'restricted rule evidence EM','value':rule_result['answer_em'],'minimum':config['review_thresholds']['rule_rag_em_min']},{'name':'restricted rule unknown abstention','value':rule_result['abstention_accuracy'],'minimum':config['review_thresholds']['rule_rag_abstention_min']}]
    if args.mode=='replay':
        prior=json.loads((folder/'results.json').read_text(encoding='utf8'))
        require(all(predictions[name][rid]['new_byte_ids']==prior['language']['outputs'][name][rid]['new_byte_ids'] for name in predictions for rid in predictions[name]),'replay_all_language_byte_predictions_equal')
        require(rule_result==prior['retrieval']['rule_result'] and probes==prior['neural_evidence'],'replay_rule_traces_and_neural_evidence_predictions_equal')
        require(all(sha(ROOT/path)==digest for key in ['sources','artifacts'] for path,digest in frozen_manifest[key].items()),'cold_rebuild_artifacts_preserved_after_replay')
    for gate in gates:gate['passed']=gate['value']>=gate['minimum']
    quality=all(g['passed'] for g in gates);report={'schema_version':1,'project':config['project'],'mode':args.mode,'collected_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU','threads':1,'dtype':'float32 model / float64 retrieval'},'config':config,'training':training,'evaluation_data':evaluation_data,'language':{'summaries':summaries,'outputs':predictions,'cases':records},'retrieval':{'selection_candidates':candidates,'selected':selected,'rule_result':rule_result,'matrix_shape':list(index.matrix.shape),'matrix_numeric_bytes':index.matrix.nbytes},'neural_evidence':probes,'checkpoints':{'base':{'path':str(base_path.relative_to(ROOT)),'file_sha256':sha(base_path),'state_sha256':sft.state_hash(base),'reload_max_logit_error':base_error},'sft':{'path':str((sft_folder/'response_only.pt').relative_to(ROOT)),'file_sha256':sha(sft_folder/'response_only.pt'),'state_sha256':sft.state_hash(fine),'reload_max_logit_error':sft_error}},'protected_previous_artifacts':protected_before,'checks':{'status':'passed','count':len(checks),'names':checks},'quality_review':{'threshold_origin':config['review_thresholds']['scope'],'gates':gates,'ready_for_stated_application':quality,'status':'quality_met' if quality else 'quality_not_met'},'resources':{'current_run_seconds_excluding_imports_and_export':time.perf_counter()-started,'timing_scope':'timer begins after numpy/torch imports and thread setup; includes helper imports, checks, training, evaluation and checkpoint I/O; excludes figure/report generation and export','stage_seconds':durations,'model_parameter_numeric_bytes_each':141312,'peak_process_memory':'not measured'}}
    report['publication']={'export_requested':args.mode=='rebuild' and not args.skip_export,'scope':'request only; export completion is separately recorded in report-publication.json for this rebuild; replay preserves the original report'}
    report_path=folder/('results.json' if args.mode=='rebuild' else 'replay-results.json');write_json(report_path,report)
    markdown_path=folder/'report.md'
    if args.mode=='rebuild':
        fg=import_file('capstone30_figures',HERE/'figures.py');fg.write_figures(report,ROOT/'docs/assets/30-capstone')
        text_module=import_file('capstone30_report_text',HERE/'report_text.py');markdown_path.write_text(text_module.report_markdown(report),encoding='utf-8',newline='\n')
        if not args.skip_export:
            env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1'};completed=subprocess.run(['node',str(HERE/'render-report.mjs')],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace');(folder/'export-console.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8',newline='\n')
            if completed.returncode:raise RuntimeError('Standalone HTML/PDF export failed; read '+str(folder/'export-console.log'))
    artifacts=[base_path,sft_folder/'response_only.pt',folder/'step-237-training.pt',training_path,index_path,report_path]
    if args.mode=='rebuild':
        artifacts +=[markdown_path]+sorted((ROOT/'docs/assets/30-capstone').glob('*.svg'))
        if not args.skip_export:artifacts +=[folder/'report.html',folder/'report.pdf',folder/'report-publication.json']
    manifest={'schema_version':1,'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'project':config['project'],'mode':args.mode,'sources':{str(p.relative_to(ROOT)):sha(p) for p in source_files()},'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in artifacts},'status':'engineering_checks_passed','quality_review':report['quality_review'],'hash_scope':'SHA256 identifies exact files, not semantic equivalence or data independence; manifest and verification file exclude themselves'};manifest_path=folder/('manifest.json' if args.mode=='rebuild' else 'replay-manifest.json');write_json(manifest_path,manifest);verified=all(sha(ROOT/p)==value for mapping in [manifest['sources'],manifest['artifacts']] for p,value in mapping.items());write_json(folder/('artifact-verification.json' if args.mode=='rebuild' else 'replay-artifact-verification.json'),{'status':'passed' if verified else 'failed','source_files':len(manifest['sources']),'artifact_files':len(manifest['artifacts']),'checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat()})
    if not verified:raise RuntimeError('manifest_all_recorded_files_match')
    emit('complete',mode=args.mode,checks=report['checks']['count'],quality=report['quality_review']['status'],review_gate_results=gates,project_report=str(markdown_path.relative_to(ROOT)),manifest=str(manifest_path.relative_to(ROOT)))
    if args.require_quality and not quality:raise SystemExit(2)
if __name__=='__main__':main()
