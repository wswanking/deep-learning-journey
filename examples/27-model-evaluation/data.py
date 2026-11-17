"""Freeze the existing chapter-22 test plus an original out-of-domain diagnostic set."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def records():
    source=[json.loads(line) for line in (ROOT/'examples/22-instruction-tuning/data/test.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    values=[{**r,'id':'id22:'+r['id'],'suite':'existing_22_test','source':'chapter22 frozen test reevaluation, not a new blind benchmark'} for r in source]
    for payload in ['gh','gi','hj','ik','jl','kl']:
        for task,verb in [('copy','复制'),('reverse','反转')]:values.append({'id':f'ood:{payload}:{task}','group_id':'ood:'+payload,'suite':'new_alphabet_diagnostic','source':'original new g..l diagnostic, not used for optimization','task':task,'instruction':verb+' '+payload,'response':payload if task=='copy' else payload[::-1]})
    return values

def load_records():
    path=HERE/'evaluation.jsonl';values=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if values!=records():raise ValueError('frozen evaluation differs from declared source and diagnostic rule')
    prior_train={json.loads(line)['instruction'] for name in ['train','validation'] for line in (ROOT/f'examples/22-instruction-tuning/data/{name}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()}
    if any(r['instruction'] in prior_train for r in values):raise ValueError('evaluation prompt overlaps SFT train/validation')
    if len({r['id'] for r in values})!=len(values):raise ValueError('duplicate evaluation id')
    return values,{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'counts':{name:sum(r['suite']==name for r in values) for name in sorted({r['suite'] for r in values})},'data_policy':'existing test reused only for fixed evaluation, new alphabet diagnostic never used to train; no tuning on these outcomes'}
if __name__=='__main__':(HERE/'evaluation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records()),encoding='utf-8')
