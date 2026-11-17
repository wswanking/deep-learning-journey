"""Chapter 25: actual CPU cache equivalence, bounded sampling, timing and weight storage experiment."""
from __future__ import annotations
import argparse,copy,hashlib,json,platform,statistics,sys,time
from datetime import datetime
from pathlib import Path
START=time.perf_counter()
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.append(str(ROOT/'examples/21-tiny-language-model'))
from model import load_checkpoint
from tokenizer import encode,decode
from inference import CachedDecoder,sampling_distribution,quantize_linears,tensor_storage_bytes,RowInt8Linear

def require(ok,message):
    if not bool(ok):raise AssertionError(message)

def rejected(fn):
    try:fn()
    except ValueError:return True
    raise AssertionError('invalid case not rejected')

@torch.inference_mode()
def trajectory(model,prompt,new_tokens,use_cache,seed=2501,settings=None):
    ids=prompt.clone();cache=None;out=[];gen=torch.Generator().manual_seed(seed);adapter=CachedDecoder(model)
    for step in range(new_tokens):
        if use_cache:logits,cache=adapter(ids if step==0 else ids[:,-1:],cache)
        else:logits=model(ids)
        z=logits[0,-1]
        token=int(z.argmax()) if settings is None else int(torch.multinomial(sampling_distribution(z,**settings),1,generator=gen))
        out.append(token);ids=torch.cat([ids,torch.tensor([[token]])],1)
    return out

@torch.inference_mode()
def timed_workload(model,ids,prompt_length,generated,use_cache):
    adapter=CachedDecoder(model);start=time.perf_counter()
    if use_cache:_,cache=adapter(ids[:,:prompt_length])
    else:model(ids[:,:prompt_length])
    split=time.perf_counter()
    for step in range(1,generated):
        if use_cache:_,cache=adapter(ids[:,prompt_length+step-1:prompt_length+step],cache)
        else:model(ids[:,:prompt_length+step])
    end=time.perf_counter()
    return [(split-start)*1000,(end-split)*1000,(end-start)*1000]

@torch.inference_mode()
def benchmark(model,repeats):
    generator=torch.Generator().manual_seed(2503);rows=[]
    for batch,prompt in [(1,8),(1,24),(1,40),(4,24)]:
        generated=16;ids=torch.randint(0,256,(batch,prompt+generated-1),generator=generator)
        for _ in range(5):
            for flag in [False,True]:timed_workload(model,ids,prompt,generated,flag)
        measurements={False:[],True:[]}
        for repeat in range(repeats):
            for flag in ([False,True] if repeat%2==0 else [True,False]):measurements[flag].append(timed_workload(model,ids,prompt,generated,flag))
        for flag in [False,True]:
            raw=np.array(measurements[flag]);median=np.median(raw,axis=0)
            rows.append({'batch':batch,'prompt_bytes':prompt,'output_distributions':generated,'decode_calls':generated-1,'cache':flag,'prefill_median_ms':float(median[0]),'decode15_median_ms':float(median[1]),'total_median_ms':float(median[2]),'total_p10_ms':float(np.quantile(raw[:,2],.1)),'total_p90_ms':float(np.quantile(raw[:,2],.9)),'decode_bytes_per_second':float(batch*(generated-1)/(median[1]/1000)),'cache_tensor_bytes_end':2*model.config.n_layers*batch*(prompt+generated-1)*model.config.d_model*4 if flag else 0,'projected_positions_per_sequence':prompt+generated-1 if flag else sum(prompt+s for s in range(generated)),'measurements_ms':raw.tolist()})
    return rows

@torch.inference_mode()
def matched_quantization_benchmark(model,repeats,checkpoint):
    quantized=quantize_linears(copy.deepcopy(model)).eval()
    generator=torch.Generator().manual_seed(2506)
    ids=torch.randint(0,256,(1,39),generator=generator)
    before={n:t.clone() for n,t in model.state_dict().items()}
    for _ in range(5):
        for current in [model,quantized]:timed_workload(current,ids,24,16,False)
    raw={'float32':[],'int8_storage_float32_compute':[]}
    models={'float32':model,'int8_storage_float32_compute':quantized}
    for repeat in range(repeats):
        names=list(models) if repeat%2==0 else list(reversed(models))
        for name in names:raw[name].append(timed_workload(models[name],ids,24,16,False))
    require(all(torch.equal(before[n],t) for n,t in model.state_dict().items()),'matched benchmark changed source')
    rows={}
    for name,values in raw.items():
        arr=np.array(values)
        rows[name]={'prefill_median_ms':float(np.median(arr[:,0])),'decode15_median_ms':float(np.median(arr[:,1])),
                    'total_median_ms':float(np.median(arr[:,2])),'total_p10_ms':float(np.quantile(arr[:,2],.1)),
                    'total_p90_ms':float(np.quantile(arr[:,2],.9)),'measurements_ms':values}
    result={'run_at':datetime.now().astimezone().isoformat(timespec='seconds'),
            'base_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'protocol':{'seed':2506,'batch':1,'prompt_bytes':24,'output_distributions':16,'decode_calls':15,
                        'warmups':5,'repeats':repeats,'alternating_order':True,'identical_ids':ids.tolist(),
                        'scope':'model calls only; no cache, sampling or early stop','threads':torch.get_num_threads()},
            'measurements':rows,'source_unchanged':True}
    folder=ROOT/'build/experiments/25-language-model-inference';folder.mkdir(parents=True,exist_ok=True)
    (folder/'matched-quantization.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{n:v for n,v in row.items() if n!='measurements_ms'} for k,row in rows.items()},indent=2))

@torch.inference_mode()
def mechanics(model):
    g=torch.Generator().manual_seed(2500);ids=torch.randint(0,256,(2,47),generator=g);adapter=CachedDecoder(model);full=model(ids);cache=None;pieces=[];start=0
    for length in [7,1,5,3,31]:
        z,cache=adapter(ids[:,start:start+length],cache);pieces.append(z);start+=length
    chunk_error=float((torch.cat(pieces,1)-full).abs().max());require(chunk_error<1e-5,'chunk logits mismatch')
    cache=None;errors=[]
    for end in range(1,48):
        z,cache=adapter(ids[:,end-1:end],cache);errors.append(float((z[:,-1]-model(ids[:,:end])[:,-1]).abs().max()))
    require(max(errors)<1e-5,'incremental logits mismatch');prompt=torch.tensor([encode('The red box is ')]);a=trajectory(model,prompt,16,False);b=trajectory(model,prompt,16,True);require(a==b,'greedy mismatch')
    # Deliberate position reset is a real counterexample: cached keys alone cannot repair positions.
    prefix=torch.tensor([encode('The red ')]);suffix=torch.tensor([encode('box')]);_,prefix_cache=adapter(prefix);correct,_=adapter(suffix,prefix_cache)
    positions=torch.arange(suffix.shape[1]);wrong_embedding=model.token_embedding(suffix)+model.position_embedding(positions)[None]
    right_embedding=model.token_embedding(suffix)+model.position_embedding(torch.arange(prefix.shape[1],prefix.shape[1]+suffix.shape[1]))[None]
    position_reset_embedding_error=float((wrong_embedding-right_embedding).abs().max())
    q=torch.tensor([1.,0.]);k=torch.tensor([[1.,0.],[0.,1.],[1.,1.]]);v=torch.tensor([[1.,0.],[0.,2.],[3.,1.]])
    weights=torch.softmax(k@q/(2**.5),0);out=weights@v
    require(rejected(lambda:adapter(torch.zeros((1,65),dtype=torch.long))),'limit guard')
    p=torch.tensor([.6,.25,.1,.05],dtype=torch.float64);logp=p.log();t=sampling_distribution(logp,temperature=.5);topk=sampling_distribution(logp,top_k=2);topp=sampling_distribution(logp,top_p=.8)
    require(torch.allclose(t,p.square()/p.square().sum()),'temperature mismatch');require(torch.allclose(topk,topp),'top-p crossing omitted')
    exact=sampling_distribution(torch.tensor([.5,.25,.125,.125],dtype=torch.float64).log(),top_p=.5);require(int((exact>0).sum())==1,'top-p equality boundary')
    for kwargs in [{'temperature':0},{'temperature':-1},{'top_k':0},{'top_p':0}]:rejected(lambda:sampling_distribution(logp,**kwargs))
    empirical=torch.bincount(torch.multinomial(t,20000,replacement=True,generator=torch.Generator().manual_seed(2504)),minlength=4).double()/20000
    require(float((empirical-t).abs().max())<.012,'sample frequencies too far')
    sample_configs=[('temperature .7',{'temperature':.7}),('top-k 5',{'top_k':5}),('top-p .9',{'top_p':.9})];samples=[]
    for name,settings in sample_configs:
        output=trajectory(model,prompt,16,True,2501,settings);again=trajectory(model,prompt,16,True,2501,settings);require(output==again,'seeded sampling mismatch');samples.append({'setting':name,'seed':2501,'output_bytes':output,'decoded_with_replacement':decode(output)})
    return {'chunk_lengths':[7,1,5,3,31],'chunk_logits_max_error':chunk_error,'incremental_logits_max_error':max(errors),'greedy_equal':True,'greedy_output_bytes':a,'greedy_decoded':decode(a),'position_reset_embedding_max_error':position_reset_embedding_error,'context_limit_rejected':True,'cache_batch_at_final':list(cache[0][0].shape),'hand_attention_weights':weights.tolist(),'hand_attention_output':out.tolist(),'sampling':{'base':p.tolist(),'temperature_half':t.tolist(),'top_k_2':topk.tolist(),'top_p_08':topp.tolist(),'empirical_20000':empirical.tolist(),'empirical_max_error':float((empirical-t).abs().max()),'zero_temperature_rejected':True,'top_p_boundary_passed':True},'model_samples':samples}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,default=ROOT/'build/experiments/21-tiny-language-model/base-model.pt');parser.add_argument('--repeats',type=int,default=31);parser.add_argument('--matched-quantization-only',action='store_true');args=parser.parse_args();require(args.repeats>=5,'at least 5 repeats')
    torch.set_num_threads(1);torch.manual_seed(2500);torch.use_deterministic_algorithms(True);begin=time.perf_counter();model,metadata=load_checkpoint(args.checkpoint);model.eval()
    if args.matched_quantization_only:
        matched_quantization_benchmark(model,args.repeats,args.checkpoint)
        return
    before={n:t.detach().clone() for n,t in model.state_dict().items()};checks=mechanics(model);timings=benchmark(model,args.repeats)
    quantized=quantize_linears(copy.deepcopy(model)).eval();ids=torch.tensor([encode('The red box is on the table. A blue box is here.')])
    with torch.inference_mode():
        full=model(ids);reduced=quantized(ids);labels=ids[:,1:];ce0=float(F.cross_entropy(full[:,:-1].reshape(-1,256),labels.reshape(-1)));ceq=float(F.cross_entropy(reduced[:,:-1].reshape(-1,256),labels.reshape(-1)))
        errors=[float((child.qweight.float()*child.scale-original.weight).abs().max()) for (name,original) in model.named_modules() if isinstance(original,torch.nn.Linear) for child in [quantized.get_submodule(name)]]
    require(all(torch.equal(before[n],t) for n,t in model.state_dict().items()),'source weights changed')
    restored=copy.deepcopy(quantized);restored.load_state_dict(quantized.state_dict())
    with torch.inference_mode():require(torch.equal(restored(ids),reduced),'quantized reload mismatch')
    qtimes=[]
    for _ in range(5):timed_workload(quantized,ids,24,16,False)
    for _ in range(args.repeats):qtimes.append(timed_workload(quantized,ids,24,16,False))
    w=torch.tensor([[-1.,-.2,0.,.7,1.]]);scale=w.abs().amax(1,keepdim=True)/127;q=torch.round(w/scale).to(torch.int8);deq=q.float()*scale
    result={'run_at':datetime.now().astimezone().isoformat(timespec='seconds'),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float32','threads':1},'checkpoint':{'path':str(args.checkpoint),'file_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'metadata':metadata,'parameters':sum(p.numel() for p in model.parameters())},'protocol':{'new_training':False,'seed':2500,'benchmark_ids_seed':2503,'warmups':5,'repeats':args.repeats,'alternating_cache_order':True,'fixed_continuation':'same random bytes; no early stop or sampling in timing','timing_scope':'model calls only; excludes tokenizer/load/argmax/network/queue','quantized_kernel':'int8 stored, float32 dequantize and matmul each call'},'checks':checks,'timings':timings,'quantization':{'fp32_tensor_bytes':tensor_storage_bytes(model),'int8_storage_model_tensor_bytes':tensor_storage_bytes(quantized),'linear_count':sum(isinstance(x,RowInt8Linear) for x in quantized.modules()),'max_weight_abs_error':max(errors),'max_logit_abs_error':float((full-reduced).abs().max()),'argmax_agreement':float((full.argmax(-1)==reduced.argmax(-1)).float().mean()),'probe_bytes':ids.tolist()[0],'probe_byte_ce_fp32':ce0,'probe_byte_ce_int8_storage':ceq,'probe_is_not_test_benchmark':True,'restored_equal':True,'uncached_prompt24_decode15_median_ms':np.median(qtimes,axis=0).tolist(),'hand_scale':float(scale),'hand_q':q.tolist(),'hand_reconstructed':deq.tolist()},'resources':{'checks_and_benchmarks_seconds':time.perf_counter()-begin,'including_import_seconds':time.perf_counter()-START,'peak_process_memory':'not measured'}}
    folder=ROOT/'build/experiments/25-language-model-inference';folder.mkdir(parents=True,exist_ok=True);(folder/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from figures import write_figures
    write_figures(result,ROOT/'docs/assets/25-language-model-inference')
    print(json.dumps({'status':'passed','checks':checks,'timings':[{k:v for k,v in row.items() if k!='measurements_ms'} for row in timings],'quantization':result['quantization'],'resources':result['resources']},ensure_ascii=True,indent=2))
if __name__=='__main__':main()
