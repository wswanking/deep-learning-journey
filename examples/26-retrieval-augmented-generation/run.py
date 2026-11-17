"""Run chapter-26 exact retrieval, controlled QA, and paired tiny-LM failure probe."""
from __future__ import annotations
import time
PROCESS_START=time.perf_counter()
import argparse,hashlib,json,platform,sys,importlib.util
from datetime import datetime
from pathlib import Path
import numpy as np
from data import dataset
from retrieval import ExactIndex,split_documents,rerank,extract,evaluate,contains_gold,features

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'build/experiments/26-retrieval-augmented-generation'
CHECKS=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    CHECKS.append(name)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-lm',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    documents,validation,test=[[json.loads(line) for line in (Path(__file__).parent/name).read_text(encoding='utf-8').splitlines() if line.strip()] for name in ['documents.jsonl','validation.jsonl','test.jsonl']];docmap={d['id']:d['text'] for d in documents}
    check('12 unique original document IDs',len(docmap)==12)
    check('validation/test questions and IDs disjoint',not({r['id'] for r in validation}&{r['id'] for r in test}) and not({r['question'] for r in validation}&{r['question'] for r in test}))
    check('predeclared 6/12/4 evaluation counts',len(validation)==6 and sum(r['answerable'] for r in test)==12 and sum(not r['answerable'] for r in test)==4)
    candidates=[]
    # Fixed k=3, two context slots. Width/overlap are selected using validation answer EM only.
    for width in [36,72,144]:
        for overlap in [0,12]:
            chunks=split_documents(documents,width,overlap);index=ExactIndex(chunks)
            for chunk in chunks:check('source slice '+chunk['id'],chunk['text']==docmap[chunk['doc_id']][chunk['start']:chunk['end']])
            val=evaluate(index,validation,3)
            candidates.append({'width':width,'overlap':overlap,'chunks':len(chunks),'validation':val})
    # Highest validation EM, then smallest total encoded character count, then width/overlap.
    selected=min(candidates,key=lambda c:(-c['validation']['answer_em'],
        sum(len(x['text']) for x in split_documents(documents,c['width'],c['overlap'])),c['width'],c['overlap']))
    chunks=split_documents(documents,selected['width'],selected['overlap']);index=ExactIndex(chunks)
    frozen={'width':selected['width'],'overlap':selected['overlap'],'k':3,'context_slots':2,'selection':'validation EM; fewer indexed characters; width; overlap'}
    # Test consulted only after the selected protocol is fixed.
    final=evaluate(index,test,3)
    no_rerank=evaluate(index,test,3,reranking=False)
    top1=evaluate(index,test,1)
    cut_small=evaluate(ExactIndex(split_documents(documents,36,0)),test,3)
    cut_large=evaluate(ExactIndex(split_documents(documents,144,0)),test,3)
    variants={'selected':final,'no_rerank':no_rerank,'top1':top1,'width36_no_overlap':cut_small,'width144_no_overlap':cut_large}
    norms=np.linalg.norm(index.matrix,axis=1)
    check('nonempty indexed vectors are unit norm',np.max(np.abs(norms-1))<1e-12)
    check('cosine matches direct hand calculation',abs(np.dot(np.array([1.,1.])/np.sqrt(2),np.array([2.,0.])/2)-1/np.sqrt(2))<1e-12)
    check('unknown empty query vector stays zero',np.all(index.encode('')==0))
    index.save(OUT/'index.npz')
    with np.load(OUT/'index.npz',allow_pickle=False) as restored:
        check('index matrix restored exactly',np.array_equal(restored['matrix'],index.matrix))
        check('IDF restored exactly',np.array_equal(restored['idf'],index.idf))
        check('vocabulary restored exactly',restored['vocabulary'].tolist()==index.vocabulary)
    check('stable top-k deterministic',index.search(test[0]['question'],3)==index.search(test[0]['question'],3))
    check('reranking preserves candidate IDs',set(c['id'] for c in index.search(test[0]['question'],3))==set(c['id'] for c in rerank(test[0]['question'],index.search(test[0]['question'],3))))
    for trace in final['traces']:
        citation=trace['citation']
        if citation:
            check('citation original slice '+trace['id'],docmap[citation['doc_id']][citation['start']:citation['end']]==citation['text'])
            check('answer occurs in cited sentence '+trace['id'],trace['answer'] in citation['text'])
    check('missing entity explicitly abstains',all(t['answer'] is None for t in final['traces'] if not t['answerable']))
    check('no retrieval cannot extract evidence',all(extract(r['question'],[])['answer'] is None for r in test))
    # Deliberate failure probes isolate stages without changing selected test configuration.
    positive=test[0]
    correct_doc=next(d for d in documents if d['id']==positive['gold_doc'])
    oracle=split_documents([correct_doc],144,0)
    check('positive oracle evidence can be read',extract(positive['question'],oracle)['answer']==positive['answer'])
    wrong_context=split_documents([d for d in documents if d['id']!=positive['gold_doc']],144,0)
    check('missing gold source cannot produce its matching fact',not any(contains_gold(c,positive) for c in wrong_context))
    mixed=np.array([.7,.3]);generator=np.array([.2,.8])
    check('retrieval mixture hand example',abs(float(mixed@generator)-.38)<1e-12)
    lm=None
    retrieval_seconds=time.perf_counter()-start
    if not args.skip_lm:
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        folder=ROOT/'examples/21-tiny-language-model';sys.path.insert(0,str(folder))
        from model import load_checkpoint
        base=ROOT/'build/experiments/21-tiny-language-model/base-model.pt'
        model,metadata=load_checkpoint(base)
        @torch.no_grad()
        def generate(prompt):
            ids=list(prompt.encode('utf-8'))
            if len(ids)>64:raise ValueError('full evidence prompt exceeds fixed 64-byte context')
            output=[]
            for _ in range(16):
                x=torch.tensor([ids[-64:]],dtype=torch.long)
                nxt=int(model(x)[0,-1].argmax());ids.append(nxt);output.append(nxt)
                if nxt==10:break
            raw=bytes(output)
            return {'prompt':prompt,'new_byte_ids':output,'text':raw.decode('utf-8',errors='replace'),
                    'utf8_valid':_valid_utf8(raw)}
        paired=[]
        probe_start=time.perf_counter()
        for trace in final['traces']:
            if not trace['answerable']:continue
            # The reader-selected sentence is used only if actually present in retrieved context.
            evidence=trace['citation']['text'] if trace['citation'] else ''
            closed=generate('q:'+trace['question']+'\na:')
            grounded=generate('d:'+evidence+'\nq:'+trace['question']+'\na:')
            for row in [closed,grounded]:
                row['answer_em']=row['text'].strip()==trace['answer']
                row['answer_substring']=trace['answer'] in row['text']
            paired.append({'id':trace['id'],'question':trace['question'],'gold':trace['answer'],
                           'evidence':evidence,'closed_book':closed,'with_retrieval':grounded})
        lm={'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'parameters':sum(p.numel() for p in model.parameters()),
            'precision':'float32','new_byte_limit':16,'stopping':'LF byte10 or fixed16',
            'context_limit':64,'cache':False,'generation_seconds':time.perf_counter()-probe_start,
            'closed_book_em':float(np.mean([r['closed_book']['answer_em'] for r in paired])),
            'retrieval_em':float(np.mean([r['with_retrieval']['answer_em'] for r in paired])),
            'pairs':paired}
        check('paired generator evaluated all 12 fixed answerable questions',len(paired)==12)
        check('all initial prompts preserve full evidence and question',all(len(p['with_retrieval']['prompt'].encode())<=64 for p in paired))
    # Stable index/version fingerprint. No evaluation label is an index feature.
    canonical=json.dumps({'documents':documents,'protocol':frozen},sort_keys=True).encode()
    result={'timestamp':datetime.now().astimezone().isoformat(),'environment':{'python':platform.python_version(),'numpy':np.__version__,'device':'CPU','retrieval_precision':'float64'},
            'data':{'documents':12,'validation':6,'test_answerable':12,'test_unanswerable':4,'license':'original fictional data, not real facility advice'},
            'protocol':frozen,'index':{'sha256':hashlib.sha256(canonical).hexdigest(),'chunks':len(chunks),'features':len(index.vocabulary),'matrix_shape':list(index.matrix.shape),'matrix_bytes':index.matrix.nbytes},
            'selection_candidates':candidates,'variants':variants,
            'no_retrieval':{'answerable_em':0.0,'unanswerable_abstention':1.0,'reader':'same restricted extractor with empty context'},
            'tiny_lm':lm,'checks':CHECKS,'checks_passed':len(CHECKS),
            'resources':{'retrieval_and_checks_seconds':retrieval_seconds,'elapsed_including_imports_seconds':time.perf_counter()-PROCESS_START,'peak_memory':'not measured'}}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'checks':len(CHECKS),'protocol':frozen,'selected':{k:v for k,v in final.items() if k!='traces'},'tiny_lm':None if lm is None else {k:lm[k] for k in ['closed_book_em','retrieval_em','generation_seconds']},'index':result['index']},ensure_ascii=False))
def _valid_utf8(raw):
    try:raw.decode('utf-8',errors='strict');return True
    except UnicodeDecodeError:return False
if __name__=='__main__':main()
