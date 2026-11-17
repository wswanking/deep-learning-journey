"""Exact NumPy lexical retrieval and an explicit controlled extractive reader."""
from __future__ import annotations
from collections import Counter
import re
import numpy as np

def features(text):
    text=re.sub(r'\s+',' ',text.lower()).strip()
    return Counter(text[i:i+3] for i in range(max(0,len(text)-2)))

def split_documents(documents,width,overlap):
    if not 0<=overlap<width:raise ValueError('require 0 <= overlap < width')
    chunks=[]
    for doc in documents:
        for start in range(0,len(doc['text']),width-overlap):
            end=min(start+width,len(doc['text']))
            chunks.append({'id':f"{doc['id']}:{start}:{end}",'doc_id':doc['id'],
                           'start':start,'end':end,'text':doc['text'][start:end]})
            if end==len(doc['text']):break
    return chunks

class ExactIndex:
    def __init__(self,chunks):
        if not chunks:raise ValueError('nonempty index required')
        self.chunks=chunks;counts=[features(c['text']) for c in chunks]
        self.vocabulary=sorted(set().union(*(set(c) for c in counts)))
        self.ids={term:i for i,term in enumerate(self.vocabulary)}
        df=np.array([sum(t in c for c in counts) for t in self.vocabulary],dtype=np.float64)
        self.idf=np.log((1+len(chunks))/(1+df))+1
        self.matrix=np.stack([self.encode(c['text']) for c in chunks])
    def encode(self,text):
        values=np.zeros(len(self.vocabulary),dtype=np.float64)
        for term,count in features(text).items():
            if term in self.ids:values[self.ids[term]]=(1+np.log(count))*self.idf[self.ids[term]]
        norm=np.linalg.norm(values)
        return values/norm if norm else values
    def search(self,question,k):
        scores=self.matrix@self.encode(question)
        # Stable original chunk order breaks equal-score ties.
        order=np.argsort(-scores,kind='stable')[:k]
        return [{**self.chunks[i],'score':float(scores[i])} for i in order]
    def save(self,path):
        np.savez(path,matrix=self.matrix,idf=self.idf,vocabulary=np.array(self.vocabulary))

def question_fields(question):
    # Restricted grammar is deliberately documented, not a learned reranker.
    entity=question.split()[0]
    lower=question.lower()
    field='opens' if 'open' in lower else 'capacity' if 'capacity' in lower else 'fee' if ('fee' in lower or 'price' in lower) else None
    return entity,field

def eligible_sentences(candidate):
    # Discard fragments at chunk boundaries; offsets remain original Python str indices.
    text=candidate['text'];cursor=0;sentences=[]
    for match in re.finditer(r'[^.]+\.',text):
        raw=match.group();left=len(raw)-len(raw.lstrip())
        sentence=raw.strip();start=candidate['start']+match.start()+left
        # A trailing partial sentence is excluded by the terminating full stop rule.
        sentences.append({'text':sentence,'start':start,'end':start+len(sentence),
                          'doc_id':candidate['doc_id'],'chunk_id':candidate['id']})
    return sentences

def rerank(question,candidates):
    entity,field=question_fields(question);ranked=[]
    for candidate in candidates:
        best=0
        for sentence in eligible_sentences(candidate):
            entity_match=bool(re.search(r'\b'+re.escape(entity)+r'\b',sentence['text'],re.I))
            field_match=field is not None and bool(re.search(r'\b'+field+r'\b',sentence['text'],re.I))
            best=max(best,2*int(entity_match)+int(field_match))
        ranked.append({**candidate,'rerank_score':best})
    return sorted(ranked,key=lambda c:(-c['rerank_score'],-c['score']))

def extract(question,candidates):
    entity,field=question_fields(question)
    if field is None:return {'answer':None,'citation':None,'reason':'unsupported question grammar'}
    for candidate in candidates:
        for sentence in eligible_sentences(candidate):
            pattern=r'\b'+re.escape(entity)+r'\s+'+field+r'(?: at| is)\s+([0-9:]+)'
            match=re.search(pattern,sentence['text'],re.I)
            if match:
                return {'answer':match.group(1),'citation':sentence,
                        'reason':'restricted literal extraction'}
    return {'answer':None,'citation':None,'reason':'no complete matching sentence in context'}

def contains_gold(chunk,row):
    if not row['answerable'] or chunk['doc_id']!=row['gold_doc']:return False
    entity,field=question_fields(row['question'])
    pattern=r'\b'+entity+r'\s+'+field+r'(?: at| is)\s+'+re.escape(row['answer'])+r'\b'
    return bool(re.search(pattern,chunk['text'],re.I))

def evaluate(index,rows,k,reranking=True,limit=2):
    traces=[]
    for row in rows:
        retrieved=index.search(row['question'],k)
        ranked=rerank(row['question'],retrieved) if reranking else retrieved
        context=ranked[:limit]
        answer=extract(row['question'],context)
        # Correctness uses held-out gold only after retrieval/reading, never to build context.
        hit=any(contains_gold(c,row) for c in retrieved)
        context_hit=any(contains_gold(c,row) for c in context)
        correct=answer['answer']==row['answer']
        citation=answer['citation']
        supported=bool(citation and row['answerable'] and citation['doc_id']==row['gold_doc']
                       and answer['answer']==row['answer'] and row['answer'] in citation['text'])
        traces.append({**row,'retrieved':retrieved,'context':context,'retrieval_hit':hit,
                       'context_hit':context_hit,**answer,'correct':correct,'gold_supported':supported})
    positive=[t for t in traces if t['answerable']];negative=[t for t in traces if not t['answerable']]
    emitted=[t for t in traces if t['answer'] is not None]
    def mean(values):return float(np.mean(values)) if values else None
    return {'retrieval_hit_at_k':mean([t['retrieval_hit'] for t in positive]),
            'context_hit':mean([t['context_hit'] for t in positive]),
            'answer_em':mean([t['correct'] for t in positive]),
            'abstention_accuracy':mean([t['answer'] is None for t in negative]),
            'citation_precision_on_emitted':mean([t['gold_supported'] for t in emitted]),
            'answered':len(emitted),'traces':traces}
