"""Chapter 15: transparent single-head attention, masks, scaling, and counterexamples on CPU."""
from __future__ import annotations
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from torch.nn import functional as F


def attention_numpy(query,key,value,allow=None):
    q,k,v=[np.asarray(x,dtype=np.float64) for x in (query,key,value)]
    if q.ndim<2 or k.ndim!=q.ndim or v.ndim!=q.ndim: raise ValueError('Q/K/V must have matching ranks >=2')
    if q.shape[:-2]!=k.shape[:-2] or k.shape[:-2]!=v.shape[:-2]: raise ValueError('matching batch dimensions required')
    if q.shape[-1]!=k.shape[-1] or k.shape[-2]!=v.shape[-2] or q.shape[-1]<1: raise ValueError('Q/K dimensions and K/V lengths must match')
    if not all(np.isfinite(x).all() for x in (q,k,v)): raise ValueError('finite input values required')
    scores=(q@np.swapaxes(k,-2,-1))/math.sqrt(q.shape[-1])
    if not np.isfinite(scores).all(): raise ValueError('nonfinite scores')
    mask=np.ones(scores.shape,dtype=bool) if allow is None else np.broadcast_to(np.asarray(allow,dtype=bool),scores.shape)
    if not mask.any(-1).all(): raise ValueError('every computed query row requires at least one allowed key')
    masked=np.where(mask,scores,-np.inf)
    shifted=masked-masked.max(axis=-1,keepdims=True)
    numerator=np.exp(shifted); weights=numerator/numerator.sum(axis=-1,keepdims=True)
    return weights@v,weights,scores


def attention_torch(q,k,v,allow=None):
    scores=q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])
    mask=torch.ones_like(scores,dtype=torch.bool) if allow is None else torch.broadcast_to(allow,scores.shape)
    if not bool(mask.any(-1).all()): raise ValueError('every computed query row requires at least one allowed key')
    weights=torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1)
    return weights@v,weights


def scaling_experiment(rng):
    results=[]
    for dimension in [2,8,32,128,256]:
        q=rng.standard_normal((512,dimension)); k=rng.standard_normal((512,8,dimension))
        raw=np.einsum('nd,nkd->nk',q,k); scaled=raw/math.sqrt(dimension)
        stats={}
        for label,scores in [('raw',raw),('scaled',scaled)]:
            shifted=scores-scores.max(-1,keepdims=True); p=np.exp(shifted); p/=p.sum(-1,keepdims=True)
            entropy=-np.sum(np.where(p>0,p*np.log(np.maximum(p,np.finfo(float).tiny)),0),axis=-1)
            stats[label]={'score_variance':float(scores.var()),'mean_entropy_nat':float(entropy.mean()),'mean_max_weight':float(p.max(-1).mean())}
        results.append({'key_dimension':dimension,'queries':512,'keys_per_query':8,**stats})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,default=Path('build/experiments/15-attention')); parser.add_argument('--write-figures',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    started=time.perf_counter(); rng=np.random.default_rng(20261115); checks=[]
    def require(condition,name):
        if not bool(condition): raise RuntimeError(name)
        checks.append(name)
    q=np.array([[math.sqrt(2),0],[0,math.sqrt(2)]])
    k=np.array([[math.log(2),0],[0,math.log(2)],[0,0]])
    v=np.array([[2,0],[0,4],[2,4]],dtype=np.float64)
    hand_out,hand_weights,hand_scores=attention_numpy(q,k,v)
    require(np.allclose(hand_weights,[[.5,.25,.25],[.25,.5,.25]]),'hand_attention_weights')
    require(np.allclose(hand_out,[[1.5,2],[1,3]]),'hand_weighted_outputs')
    tq,tk,tv=[torch.tensor(a,dtype=torch.float64) for a in (q,k,v)]
    torch_out,_=attention_torch(tq,tk,tv)
    library_out=F.scaled_dot_product_attention(tq,tk,tv,dropout_p=0.0)
    hand_library_error=float(np.max(np.abs(library_out.numpy()-hand_out)))
    require(np.allclose(torch_out.numpy(),hand_out,atol=1e-14),'numpy_matches_torch_hand')
    require(hand_library_error<1e-14,'hand_matches_pytorch_sdpa')
    batched_q=rng.standard_normal((2,4,3)); batched_k=rng.standard_normal((2,4,3)); batched_v=rng.standard_normal((2,4,2))
    lengths=np.array([4,2]); positions=np.arange(4); key_valid=positions[None,:]<lengths[:,None]; query_valid=key_valid.copy()
    causal=positions[None,:]<=positions[:,None]
    allow=causal[None,:,:]&key_valid[:,None,:]
    masked_out,masked_weights,_=attention_numpy(batched_q,batched_k,batched_v,allow)
    plain_out,_,_=attention_numpy(batched_q,batched_k,batched_v)
    ts=[torch.tensor(a,dtype=torch.float64) for a in (batched_q,batched_k,batched_v)]; ta=torch.tensor(allow)
    reference=F.scaled_dot_product_attention(*ts,attn_mask=ta,dropout_p=0.0)
    manual,manual_weights=attention_torch(*ts,ta)
    require(np.max(np.abs(reference.numpy()-masked_out))<1e-14,'batched_mask_matches_sdpa_true_is_allowed')
    require(np.max(np.abs(manual.numpy()-masked_out))<1e-14,'batched_numpy_matches_manual_torch')
    require(np.allclose(masked_weights.sum(-1),1),'weights_sum_to_one_along_keys')
    require(np.count_nonzero(masked_weights[~allow])==0,'all_forbidden_weights_are_exactly_zero')
    changed_k=batched_k.copy(); changed_v=batched_v.copy(); changed_k[:,2:]+=1000; changed_v[:,2:]-=1000
    changed_out,_,_=attention_numpy(batched_q,changed_k,changed_v,allow)
    future_error=float(np.max(np.abs(changed_out[:,:2]-masked_out[:,:2])))
    changed_unmasked,_,_=attention_numpy(batched_q,changed_k,changed_v)
    leakage=float(np.max(np.abs(changed_unmasked[:,:2]-plain_out[:,:2])))
    require(future_error==0,'future_and_padded_values_do_not_change_valid_prefix')
    require(leakage>1,'unmasked_attention_leaks_future_values')
    pad_k=batched_k.copy(); pad_v=batched_v.copy(); pad_k[1,2:]+=1000; pad_v[1,2:]-=1000
    pad_out,_,_=attention_numpy(batched_q,pad_k,pad_v,allow)
    padding_error=float(np.max(np.abs(pad_out[1,:2]-masked_out[1,:2])))
    require(padding_error==0,'padding_keys_have_no_effect_on_real_queries')
    query_zeroed=masked_out*query_valid[:,:,None]
    require(np.max(np.abs(query_zeroed[1,2:]))==0,'query_padding_requires_explicit_output_mask')
    gradient_inputs=[torch.tensor(a,dtype=torch.float64,requires_grad=True) for a in (batched_q,batched_k,batched_v)]
    gradient_output,_=attention_torch(*gradient_inputs,ta); gradient_output[:,:2].sum().backward()
    future_key_gradient=float(gradient_inputs[1].grad[:,2:].abs().max()); future_value_gradient=float(gradient_inputs[2].grad[:,2:].abs().max())
    require(future_key_gradient==0 and future_value_gradient==0,'valid_prefix_gradient_to_future_keys_values_is_zero')
    all_masked_raises=False
    try: attention_numpy(q,k,v,np.zeros((2,3),dtype=bool))
    except ValueError: all_masked_raises=True
    require(all_masked_raises,'all_masked_rows_explicitly_rejected')
    invalid_rejected=False
    try: attention_numpy(q,np.zeros((3,4)),v)
    except ValueError: invalid_rejected=True
    require(invalid_rejected,'incompatible_dimensions_rejected')
    nan_rejected=False
    try: attention_numpy(q,k,np.full_like(v,np.nan))
    except ValueError: nan_rejected=True
    require(nan_rejected,'nonfinite_values_rejected')
    scalar_mask=np.array([[True,False,True],[True,False,True]])
    kept_out,kept_weights,_=attention_numpy(q,k,v,scalar_mask)
    require(np.allclose(kept_weights[:,1],0),'padding_mask_removes_weight_before_softmax')
    require(np.allclose(kept_weights.sum(-1),1),'remaining_weights_are_renormalized')
    permutation=np.array([2,0,1]); perm_out,_,_=attention_numpy(q,k[permutation],v[permutation])
    require(np.allclose(perm_out,hand_out),'joint_key_value_permutation_preserves_output_without_position')
    uniform_a=np.array([[.9,.1]]); uniform_b=np.array([[.1,.9]]); duplicate_values=np.array([[2.,3.],[2.,3.]])
    duplicate_first=uniform_a@duplicate_values; duplicate_second=uniform_b@duplicate_values
    require(np.allclose(duplicate_first,duplicate_second),'different_attention_weights_can_give_identical_output')
    scale_results=scaling_experiment(rng)
    require(all(.7<r['scaled']['score_variance']<1.3 for r in scale_results),'scaled_score_variance_stays_near_one_in_synthetic_experiment')
    require(scale_results[-1]['raw']['mean_entropy_nat']<scale_results[-1]['scaled']['mean_entropy_nat'],'large_unscaled_scores_are_more_concentrated_in_this_experiment')
    wrong_axis=np.exp(hand_scores-hand_scores.max(0,keepdims=True)); wrong_axis/=wrong_axis.sum(0,keepdims=True)
    require(not np.allclose(wrong_axis.sum(-1),1),'softmax_wrong_axis_breaks_per_query_probability')
    report={'chapter':'15','collected_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'python_executable':sys.executable,'device':'CPU','dtype':'float64','threads':1},
            'protocol':{'data':'constructed small matrices and seeded synthetic standard-normal vectors','seed':20261115,'trainable_parameters':0,'training_steps':None,'optimizer':None,'dataset_split':None,'dropout_p':0.0,'all_masked_policy':'reject every all-masked computed row','mask_convention':'True means allowed; applies before key-axis softmax'},
            'hand':{'query':q.tolist(),'key':k.tolist(),'value':v.tolist(),'scores':hand_scores.tolist(),'weights':hand_weights.tolist(),'output':hand_out.tolist(),'sdpa_max_error':hand_library_error,'masked_middle_weights':kept_weights.tolist(),'masked_middle_output':kept_out.tolist(),'wrong_axis_row_sums':wrong_axis.sum(-1).tolist()},
            'masks':{'lengths':lengths.tolist(),'causal':causal.tolist(),'allow':allow.tolist(),'weights':masked_weights.tolist(),'raw_query_outputs':masked_out.tolist(),'query_zeroed_outputs':query_zeroed.tolist(),'maximum_future_perturbation_prefix_error':future_error,'maximum_padding_perturbation_error':padding_error,'unmasked_prefix_change':leakage,'future_key_gradient_max':future_key_gradient,'future_value_gradient_max':future_value_gradient,'sdpa_max_error':float(np.max(np.abs(reference.numpy()-masked_out)))},
            'scaling':scale_results,'attention_explanation_counterexample':{'first_weights':uniform_a.tolist(),'second_weights':uniform_b.tolist(),'values':duplicate_values.tolist(),'first_output':duplicate_first.tolist(),'second_output':duplicate_second.tolist()},
            'resources':{'elapsed_seconds_excluding_imports':time.perf_counter()-started,'hand_qkv_numeric_payload_bytes':sum(a.nbytes for a in (q,k,v)),'memory_scope':'small hand-example inputs only; not total process or scaling-simulation peak'},
            'checks':{'status':'passed','count':len(checks),'names':checks}}
    args.output_dir.mkdir(parents=True,exist_ok=True); output=args.output_dir/'results.json'; output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.write_figures:
        from figures import write_figures
        write_figures(report,Path(__file__).resolve().parents[2]/'docs/assets/15-attention')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
