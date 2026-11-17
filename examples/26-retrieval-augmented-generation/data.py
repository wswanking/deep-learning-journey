"""Original fictional facility documents and fixed held-out question protocols."""
from __future__ import annotations
import json
from pathlib import Path

FACILITIES=[
("Sol","08:30","17:10",12,40),("Luna","09:15","18:20",18,24),
("Mira","07:45","16:40",15,32),("Nova","10:00","19:30",21,28),
("Argo","08:10","17:50",16,36),("Vega","09:40","18:05",14,44)]
def dataset():
    documents=[];validation=[];test=[]
    for i,(name,opening,closing,fee,capacity) in enumerate(FACILITIES):
        hours=f"{name} opens at {opening}. {name} closes at {closing}. The opening desk serves booked visitors only. The entrance is on the east side."
        policy=f"{name} fee is {fee} credits. {name} capacity is {capacity} people. Bring a printed booking code to the service desk. The payment counter is on the west side."
        for kind,text in [('hours',hours),('policy',policy)]:
            documents.append({'id':name.lower()+'-'+kind,'text':text,'snapshot':'fictional-v1','license':'original tutorial data'})
        def row(uid,question,answer,doc):
            return {'id':uid,'question':question,'answer':answer,'gold_doc':doc,'answerable':True}
        validation.append(row('v'+str(i),name+' capacity?',str(capacity),name.lower()+'-policy'))
        test.append(row('t'+str(2*i),name+' opening time?',opening,name.lower()+'-hours'))
        # Predeclared lexical mismatch: "price" versus "fee", no per-question tuning.
        test.append(row('t'+str(2*i+1),name+(' price?' if i%2 else ' fee?'),str(fee),name.lower()+'-policy'))
    for i,name in enumerate(['Orion','Atlas','Nix','Ceres']):
        test.append({'id':'u'+str(i),'question':name+' opening time?','answer':None,'gold_doc':None,'answerable':False})
    return documents,validation,test
def write_originals(directory):
    directory=Path(directory)
    for filename,rows in zip(['documents.jsonl','validation.jsonl','test.jsonl'],dataset()):
        (directory/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
if __name__=='__main__':write_originals(Path(__file__).parent)
