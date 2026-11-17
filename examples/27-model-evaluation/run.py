"""CPU frozen-model evaluation with fixed outputs, grouped comparisons and RAG traces."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,collections,copy,hashlib,importlib.util,json,math,platform,shutil,sys,time
import numpy as np
import torch
from torch.nn import functional as F
from data import load_records
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(1,str(ROOT/'examples/21-tiny-language-model'))
from model import load_checkpoint
from tokenizer import encode,decode

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def prompt(row):return '[指令]\n'+row['instruction']+'\n[回答]\n'
def score_output(row,ids):
    raw=bytes(ids);stopped=bool(ids and ids[-1]==10);body=ids[:-1] if stopped else ids
    try:raw.decode('utf-8');valid=True
    except UnicodeDecodeError:valid=False
    fmt=stopped and len(body)==2 and all(97<=v<=108 for v in body);exact=stopped and body==encode(row['response'])
    category='correct' if exact else 'invalid_utf8' if not valid else 'no_stop' if not stopped else 'format_invalid' if not fmt else 'wrong_answer'
    return {'text':decode(body),'new_byte_ids':ids,'utf8_valid':valid,'stopped_on_lf':stopped,'format_valid':fmt,'exact_match':exact,'category':category}

@torch.no_grad()
def predict(model,row):
    values=encode(prompt(row));new=[]
    if len(values)>model.config.context_length:raise ValueError('prompt exceeds shared context')
    for _ in range(8):
        token=int(model(torch.tensor([values[-64:]],dtype=torch.long))[0,-1].argmax());new.append(token);values.append(token)
        if token==10:break
    return score_output(row,new)

@torch.no_grad()
def target_nll(model,row):
    prefix=encode(prompt(row));full=prefix+encode(row['response']+'\n');inputs=torch.tensor([full[:-1]],dtype=torch.long);target=torch.tensor(full[1:],dtype=torch.long);logits=model(inputs)[0];loss=F.cross_entropy(logits[len(prefix)-1:],target[len(prefix)-1:],reduction='sum')
    return float(loss),len(full)-len(prefix)

def grouped_difference(records,a,b,suite):
    rows=[r for r in records if r['suite']==suite];groups=sorted({r['group_id'] for r in rows});diffs=[]
    for group in groups:
        members=[r for r in rows if r['group_id']==group];diffs.append(np.mean([int(b[r['id']]['exact_match'])-int(a[r['id']]['exact_match']) for r in members]))
    if len({sum(r['group_id']==g for r in rows) for g in groups})!=1:raise ValueError('this teaching estimator assumes equal cluster sizes within each suite')
    rng=np.random.default_rng(2701);resampled=np.array(diffs)[rng.integers(0,len(groups),size=(2000,len(groups)))].mean(1)
    return {'comparison':'SFT minus base','group_count':len(groups),'differences_by_group':dict(zip(groups,map(float,diffs))),'point_difference':float(np.mean(diffs)),'percentile_95_interval':np.quantile(resampled,[.025,.975]).tolist(),'replicates':2000,'seed':2701,'interpretation':'descriptive paired group bootstrap; few synthetic groups, not a population guarantee'}

def summarize(records,predictions,nlls=None):
    result={}
    for suite in sorted({r['suite'] for r in records}):
        selected=[r for r in records if r['suite']==suite];values=[predictions[r['id']] for r in selected];timings=[r['latency_ms'] for r in values]
        item={'count':len(selected),'correct':sum(r['exact_match'] for r in values),'exact_match':float(np.mean([r['exact_match'] for r in values])),'format_valid':float(np.mean([r['format_valid'] for r in values])),'utf8_valid':float(np.mean([r['utf8_valid'] for r in values])),'stopped':float(np.mean([r['stopped_on_lf'] for r in values])),'failure_categories':dict(collections.Counter(r['category'] for r in values)),'latency_ms_median':float(np.median(timings)),'latency_ms_p95':float(np.quantile(timings,.95)),'generated_bytes':sum(len(r['new_byte_ids']) for r in values),'latency_scope':'one warmed CPU run including tensor construction/decoding; no service concurrency or reliable tail guarantee'}
        if nlls is not None:
            total=sum(nlls[r['id']][0] for r in selected);count=sum(nlls[r['id']][1] for r in selected);item['response_nll_nat_per_byte']=total/count;item['response_perplexity_per_byte']=math.exp(total/count);item['scored_response_bytes']=count
        result[suite]=item
    return result

def classify_rag(trace):
    if not trace['answerable']:return 'correct_abstention' if trace['answer'] is None else 'unsupported_answer'
    if not trace['retrieval_hit']:return 'retrieval_miss'
    if not trace['context_hit']:return 'context_lost'
    if not trace['correct']:return 'reader_error'
    if not trace['gold_supported']:return 'unsupported_citation'
    return 'correct_supported'

def binary_metrics(tp,fp,fn,tn):
    return {'accuracy':(tp+tn)/(tp+fp+fn+tn),'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-figures',action='store_true');args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(2700);started=time.perf_counter();folder=ROOT/'build/experiments/27-model-evaluation';folder.mkdir(parents=True,exist_ok=True)
    records,data=load_records();sources={};models={};outputs={};summaries={};nlls={}
    for name,relative in [('base','21-tiny-language-model/base-model.pt'),('sft','22-instruction-tuning/response_only.pt')]:
        source=ROOT/'build/experiments'/relative
        if not source.exists():raise FileNotFoundError('Run declared previous chapters first: '+str(source))
        target=folder/f'{name}-used.pt';shutil.copyfile(source,target);model,metadata=load_checkpoint(target);models[name]=model;sources[name]={'source':str(source.relative_to(ROOT)),'frozen_copy':str(target.relative_to(ROOT)),'file_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'metadata':metadata,'parameter_count':sum(p.numel() for p in model.parameters())}
        with torch.no_grad():
            for _ in range(3):model(torch.tensor([[65,66]],dtype=torch.long))
        predictions={};teacher_scores={}
        for row in records:
            before=time.perf_counter();value=predict(model,row);value['latency_ms']=(time.perf_counter()-before)*1000;predictions[row['id']]=value;teacher_scores[row['id']]=target_nll(model,row)
        outputs[name]=predictions;nlls[name]=teacher_scores;summaries[name]=summarize(records,predictions,teacher_scores)
    baseline={}
    for row in records:
        before=time.perf_counter();payload=row['instruction'].rsplit(' ',1)[1];value=score_output(row,encode(payload+'\n'));value['latency_ms']=(time.perf_counter()-before)*1000;baseline[row['id']]=value
    outputs['echo_baseline']=baseline;summaries['echo_baseline']=summarize(records,baseline)
    differences={suite:grouped_difference(records,outputs['base'],outputs['sft'],suite) for suite in data['counts']}
    rag_path=ROOT/'build/experiments/26-retrieval-augmented-generation/results.json'
    if not rag_path.exists():raise FileNotFoundError('Run chapter26 first to create actual RAG traces')
    rag_copy=folder/'rag-source.json';shutil.copyfile(rag_path,rag_copy);rag=json.loads(rag_copy.read_text(encoding='utf-8'));rd=import_file('chapter27_rag_data',ROOT/'examples/26-retrieval-augmented-generation/data.py');rr=import_file('chapter27_retrieval',ROOT/'examples/26-retrieval-augmented-generation/retrieval.py');documents,validation,test=rd.dataset();config=rag['protocol'];chunks=rr.split_documents(documents,config['width'],config['overlap']);index=rr.ExactIndex(chunks);recomputed=rr.evaluate(index,test,config['k'],limit=config['context_slots']);docmap={r['id']:r['text'] for r in documents};selected=rag['variants']['selected'];checks=[]
    def require(condition,name):
        if not bool(condition):raise RuntimeError(name)
        checks.append(name)
    for metric in ['retrieval_hit_at_k','context_hit','answer_em','abstention_accuracy','citation_precision_on_emitted']:require(recomputed[metric]==selected[metric],'rag_recomputed_'+metric)
    require([(r['id'],r['answer'],r['citation']) for r in recomputed['traces']]==[(r['id'],r['answer'],r['citation']) for r in selected['traces']],'rag_trace_answers_and_citations_reproduced')
    for trace in selected['traces']:
        citation=trace['citation']
        if citation:require(docmap[citation['doc_id']][citation['start']:citation['end']]==citation['text'],'citation_exact_source_slice_'+trace['id'])
    rules=[{**r,'category':classify_rag(r)} for r in selected['traces']];by_id={r['id']:r for r in rules};lm_reader=[]
    require(rag['tiny_lm'] is not None and rag['tiny_lm']['base_sha256']==sources['base']['file_sha256'],'rag_neural_reader_uses_identical_base')
    for pair in rag['tiny_lm']['pairs']:
        trace=copy.deepcopy(by_id[pair['id']]);generated=pair['with_retrieval'];trace.update({'answer':generated['text'].strip(),'correct':generated['answer_em'],'gold_supported':False,'citation':None,'generation_record':generated,'reader':'actual frozen 35k byte decoder from chapter26'});trace['category']=classify_rag(trace);lm_reader.append(trace)
    target_row=test[0];without=rr.ExactIndex(rr.split_documents([d for d in documents if d['id']!=target_row['gold_doc']],config['width'],config['overlap']));missing=rr.evaluate(without,[target_row],config['k'],limit=config['context_slots'])['traces'][0];zero_context=rr.evaluate(index,[target_row],config['k'],limit=0)['traces'][0];injections=[{'intervention':'remove gold source from index, predeclared diagnostic','trace':missing,'category':classify_rag(missing)},{'intervention':'force context budget zero, predeclared diagnostic','trace':zero_context,'category':classify_rag(zero_context)}]
    require(injections[0]['category']=='retrieval_miss' and injections[1]['category']=='context_lost','controlled_rag_fault_categories')
    require(all(r['category']=='reader_error' for r in lm_reader),'neural_reader_failures_with_present_evidence')
    require(all(sum(value['failure_categories'].values())==value['count'] for result in summaries.values() for value in result.values()),'exclusive_failure_counts_cover_every_case')
    require(all(result['exact_match']==.5 for result in summaries['echo_baseline'].values()),'copy_only_baseline_exactly_half_on_balanced_tasks')
    hand=binary_metrics(8,2,4,6);require(abs(hand['f1']-8/11)<1e-14 and hand['accuracy']==.7,'manual_binary_metrics')
    probabilities=np.array([.9,.8,.6,.1]);labels=np.array([1.,0.,1.,0.]);brier=float(np.mean((probabilities-labels)**2));confidence=np.maximum(probabilities,1-probabilities);correct=(probabilities>=.5)==labels;bins=[];ece=0.
    for low,high in [(.5,.75),(.75,1.01)]:
        mask=(confidence>=low)&(confidence<high);size=int(mask.sum());acc=float(correct[mask].mean());conf=float(confidence[mask].mean());ece+=size/len(labels)*abs(acc-conf);bins.append({'lower':low,'upper':min(high,1),'count':size,'accuracy':acc,'confidence':conf})
    require(abs(brier-.205)<1e-14 and abs(ece-.25)<1e-14,'manual_brier_and_binned_ece');require(binary_metrics(0,0,3,1)['precision'] is None,'undefined_precision_is_null')
    require(all(hashlib.sha256((ROOT/source['source']).read_bytes()).hexdigest()==source['file_sha256'] for source in sources.values()),'input_checkpoints_unchanged')
    report={'schema_version':1,'chapter':'27','collected_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'CPU','threads':1,'precision':'float32 models; float64 descriptive statistics'},'data':data,'protocol':{'models_frozen':True,'training_performed':False,'updates':0,'max_new_bytes':8,'stopping':'LF or8 bytes','template':'instruction-response-utf8-v1','sampling':'greedy deterministic, seed2700 for environment','test_policy':'fixed reevaluation, no tuning; existing22 test is not a new blind result','format':'exactly two ASCII a..l letters then LF','failure_priority':['correct','invalid_utf8','no_stop','format_invalid','wrong_answer'],'bootstrap':'paired equal-sized groups within suite, 2000 replicates, seed2701','timing':'one warmed single-thread process, includes decode/tensor construction, no peak memory measured'},'sources':sources,'language_models':{'summaries':summaries,'outputs':outputs,'target_nlls':nlls,'paired_group_comparison':differences},'rag':{'source_sha256':hashlib.sha256(rag_copy.read_bytes()).hexdigest(),'source_copy':str(rag_copy.relative_to(ROOT)),'frozen_protocol':config,'rule_metrics':{k:v for k,v in selected.items() if k!='traces'},'rule_categories':dict(collections.Counter(r['category'] for r in rules)),'rule_traces':rules,'neural_reader_categories':dict(collections.Counter(r['category'] for r in lm_reader)),'neural_reader_traces':lm_reader,'controlled_fault_injections':injections,'distinction':'rule reader and actual neural reader are different components, not interchangeable quality claims'},'hand':{'confusion':{'tp':8,'fp':2,'fn':4,'tn':6},'binary':hand,'calibration_probabilities':probabilities.tolist(),'calibration_labels':labels.tolist(),'brier':brier,'ece':ece,'bins':bins,'source':'original toy calculation, not a measured calibration of either LM'},'checks':{'count':len(checks),'names':checks,'status':'passed'},'resources':{'elapsed_seconds_excluding_imports':time.perf_counter()-started,'model_parameters_each':35328,'parameter_numeric_bytes_each':141312,'memory_scope':'parameter payload only, not process peak'}}
    (folder/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.write_figures:
        from figures import write_figures
        write_figures(report,ROOT/'docs/assets/27-model-evaluation')
    print(json.dumps({'collected_at':report['collected_at'],'checks':len(checks),'summaries':summaries,'paired_differences':differences,'rag_categories':{'rule':report['rag']['rule_categories'],'neural_reader':report['rag']['neural_reader_categories']},'resources':report['resources']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
