"""Original grouped SFT data and byte-aligned response masks, shared read-only by later chapters."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import random
import torch
HERE=Path(__file__).resolve().parent
SPLITS=('train','validation','test')
TEMPLATE_VERSION='instruction-response-utf8-v1'
TASKS={'copy':['复制','原样输出'],'reverse':['反转','倒序输出']}

def render_prompt(record):
    return '[指令]\n'+record['instruction']+'\n[回答]\n'

def render_record(record):return render_prompt(record)+record['response']+'\n'
def encode_text(text):return list(text.encode('utf-8'))

def make_records():
    pairs=list(itertools.combinations('abcdef',2));random.Random(2200).shuffle(pairs)
    groups={'train':pairs[:9],'validation':pairs[9:12],'test':pairs[12:]};out={key:[] for key in SPLITS}
    for split,pairs in groups.items():
        for pair in pairs:
            group='pair:'+''.join(pair)
            for payload in [''.join(pair),''.join(reversed(pair))]:
                for task,verbs in TASKS.items():
                    for alias,verb in enumerate(verbs):
                        out[split].append({'id':f'{group}:{payload}:{task}:{alias}','group_id':group,'task':task,'instruction':verb+' '+payload,'response':payload if task=='copy' else payload[::-1]})
    return out

def validate_splits(splits):
    used_groups=set();used_ids=set();used_prompts={};sizes={};lengths=[]
    for split in SPLITS:
        records=splits[split];current=set();sizes[split]=len(records)
        for record in records:
            if any(not isinstance(record.get(k),str) or not record[k] for k in ['id','group_id','instruction','response','task']):raise ValueError('required nonempty string fields')
            if record['id'] in used_ids:raise ValueError('duplicate record id')
            used_ids.add(record['id']);current.add(record['group_id'])
            if '\n' in record['instruction'] or '\n' in record['response'] or '\0' in render_record(record):raise ValueError('newline or NUL not allowed inside fields')
            verb,payload=record['instruction'].rsplit(' ',1)
            if record['task'] not in TASKS or verb not in TASKS[record['task']]:raise ValueError('unknown task or alias')
            if len(payload)!=2 or len(set(payload))!=2 or not set(payload)<=set('abcdef'):raise ValueError('invalid payload')
            if record['group_id']!='pair:'+''.join(sorted(payload)):raise ValueError('group must bind both payload orders')
            expected=payload if record['task']=='copy' else payload[::-1]
            if record['response']!=expected:raise ValueError('inconsistent response')
            prompt=render_prompt(record)
            if prompt in used_prompts:raise ValueError('duplicate or conflicting prompt')
            used_prompts[prompt]=record['response'];lengths.append(len(encode_text(render_record(record))))
        if current&used_groups:raise ValueError('group leakage across splits')
        used_groups|=current
    return {'sizes':sizes,'groups':len(used_groups),'minimum_full_bytes':min(lengths),'maximum_full_bytes':max(lengths),'group_disjoint':True,'unique_prompts':len(used_prompts),'template_version':TEMPLATE_VERSION}

def load_splits():
    data={}
    for split in SPLITS:
        path=HERE/'data'/f'{split}.jsonl';data[split]=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return data,validate_splits(data)

def encode_record(record,context_length,objective='response_only'):
    if objective not in ['response_only','all_tokens']:raise ValueError('unknown objective')
    prefix=encode_text(render_prompt(record));full=encode_text(render_record(record))
    # No silent truncation: it could delete answer supervision or split a UTF-8 role marker.
    if len(full)-1>context_length:raise ValueError('record exceeds model context; explicitly revise the data')
    ids=full[:-1];labels=full[1:]
    if objective=='response_only':labels[:len(prefix)-1]=[-100]*(len(prefix)-1)
    if sum(v!=-100 for v in labels)!=len(encode_text(record['response']+'\n')) and objective=='response_only':raise AssertionError('answer byte count')
    return {'input_ids':ids,'labels':labels,'prefix_length':len(prefix),'record_id':record['id'],'full_bytes':full}

def collate(records,context_length,objective='response_only'):
    encoded=[encode_record(record,context_length,objective) for record in records];length=max(len(e['input_ids']) for e in encoded)
    ids=torch.zeros(len(records),length,dtype=torch.long);valid=torch.zeros_like(ids,dtype=torch.bool);labels=torch.full_like(ids,-100)
    for i,e in enumerate(encoded):
        n=len(e['input_ids']);ids[i,:n]=torch.tensor(e['input_ids']);valid[i,:n]=True;labels[i,:n]=torch.tensor(e['labels'])
    return {'input_ids':ids,'attention_mask':valid,'labels':labels,'encoded':encoded}

def fingerprints():return {split:hashlib.sha256((HERE/'data'/f'{split}.jsonl').read_bytes()).hexdigest() for split in SPLITS}

if __name__=='__main__':
    proposed=make_records();metadata=validate_splits(proposed)
    for split,records in proposed.items():
        path=HERE/'data'/f'{split}.jsonl';content=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records)
        if path.exists() and path.read_text(encoding='utf-8')!=content:raise ValueError('existing data differs; inspect it before replacing')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
    print(json.dumps({'status':'passed',**metadata,'sha256':fingerprints()},ensure_ascii=False,indent=2))
