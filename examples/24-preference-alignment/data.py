"""Original synthetic first/last-letter preference pairs; no human preference claims."""
from pathlib import Path
import hashlib,itertools,json,random
HERE=Path(__file__).resolve().parent
TEMPLATE_VERSION='instruction-response-utf8-v1'
def prompt(record):return '[指令]\n'+record['instruction']+'\n[回答]\n'
def make_splits():
    groups=list(itertools.combinations('ghijkl',2));random.Random(2400).shuffle(groups);splits={}
    for name,pairs in [('train',groups[:9]),('validation',groups[9:12]),('test',groups[12:])]:
        rows=[]
        for pair in pairs:
            group='pair:'+''.join(pair)
            for payload in [''.join(pair),''.join(reversed(pair))]:
                for task,verb,index in [('first','首字母',0),('last','末字母',1)]:
                    rows.append({'id':f'{group}:{payload}:{task}','group_id':group,'instruction':verb+' '+payload,'task':task,'chosen':payload[index],'rejected':payload[1-index],'label_source':'deterministic synthetic string rule, no human labeler'})
        splits[name]=rows
    return splits

def load_splits():
    splits={name:[json.loads(line) for line in (HERE/'data'/f'{name}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()] for name in ['train','validation','test']}
    groups={name:{r['group_id'] for r in rows} for name,rows in splits.items()};ids=[r['id'] for rows in splits.values() for r in rows];prompts=[prompt(r) for rows in splits.values() for r in rows]
    if len(ids)!=len(set(ids)) or len(prompts)!=len(set(prompts)):raise ValueError('duplicate preference pair')
    if any(groups[a]&groups[b] for a,b in itertools.combinations(groups,2)):raise ValueError('group leakage')
    if splits!=make_splits():raise ValueError('tracked data differs from declared synthetic rule')
    fingerprint={name:hashlib.sha256((HERE/'data'/f'{name}.jsonl').read_bytes()).hexdigest() for name in splits}
    return splits,{'counts':{name:len(rows) for name,rows in splits.items()},'group_ids':{name:sorted(values) for name,values in groups.items()},'sha256':fingerprint,'label_source':'original deterministic synthetic rules; not human evaluation','alphabet':'ghijkl','template_version':TEMPLATE_VERSION}

if __name__=='__main__':
    (HERE/'data').mkdir(exist_ok=True)
    for name,rows in make_splits().items():(HERE/'data'/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
