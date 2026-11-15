"""Chapter-20 original corpus, fixed cleaning/group splitting and right-padded shifted batches."""
from __future__ import annotations
import hashlib,json,unicodedata
from pathlib import Path
import torch
DATA=Path(__file__).parent/"data"/"raw.jsonl"
def canonical(text):
    text=unicodedata.normalize("NFC",text.replace("\r\n","\n").replace("\r","\n"))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()
def clean_records(raw):
    unique={};duplicates=[];dropped=[]
    for row in raw:
        text=canonical(row["text"])
        if len(text.encode("utf8"))<2:
            dropped.append(row["id"]);continue
        key=hashlib.sha256(text.encode("utf8")).hexdigest()
        if key in unique:
            if unique[key]["group_id"]!=row["group_id"]:
                raise ValueError("Exact duplicate crosses declared groups; reconcile group identities before splitting")
            duplicates.append({"removed":row["id"],"kept":unique[key]["id"]});continue
        unique[key]={"id":row["id"],"group_id":row["group_id"],"text":text,"sha256":key}
    return list(unique.values()),{"duplicates":duplicates,"dropped":dropped}
def load_splits(path=DATA):
    raw_bytes=Path(path).read_bytes()
    raw=[json.loads(line) for line in raw_bytes.decode("utf8").splitlines() if line.strip()]
    records,audit=clean_records(raw)
    groups=sorted({r["group_id"] for r in records},key=lambda g:hashlib.sha256(("20261115|"+g).encode()).hexdigest())
    if len(groups)!=12:raise ValueError("This fixed teaching protocol expects 12 original groups")
    assignments={"train":groups[:8],"validation":groups[8:10],"test":groups[10:]}
    splits={s:[r for r in records if r["group_id"] in gs] for s,gs in assignments.items()}
    metadata={"raw_records":len(raw),"clean_records":len(records),**audit,"split_groups":assignments,
              "split_records":{s:len(v) for s,v in splits.items()},"source_sha256":hashlib.sha256(raw_bytes).hexdigest(),
              "tokenizer":{"type":"utf8-byte","vocab_size":256,"special_tokens":None},
              "normalization":"NFC; CRLF/CR -> LF; strip line-end whitespace and outer whitespace",
              "split":"ascending SHA256 of 20261115|group_id; 8/2/2 groups; never fit on validation/test",
              "document_policy":"no cross-document pairs; first byte is given; no BOS/EOS or termination probability"}
    return splits,metadata
def chunks(records,context_length=64):
    if context_length<1:raise ValueError("context_length must be positive")
    result=[]
    for row in records:
        ids=list(row["text"].encode("utf8"))
        for start in range(0,len(ids)-1,context_length):
            part=ids[start:start+context_length+1]
            result.append({"document_id":row["id"],"group_id":row["group_id"],"offset":start,
                           "input_ids":part[:-1],"labels":part[1:]})
    return result
def collate(parts,context_length=64):
    if not parts:raise ValueError("nonempty batch required")
    x=torch.zeros((len(parts),context_length),dtype=torch.long)
    y=torch.full_like(x,-100);mask=torch.zeros_like(x,dtype=torch.bool)
    for i,row in enumerate(parts):
        n=len(row["input_ids"])
        if not 1<=n<=context_length:raise ValueError("parts must have a nonempty real prefix within context")
        x[i,:n]=torch.tensor(row["input_ids"]);y[i,:n]=torch.tensor(row["labels"]);mask[i,:n]=True
    return {"input_ids":x,"attention_mask":mask,"labels":y}
