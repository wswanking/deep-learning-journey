"""Small CPU regression: baseline, AdamW weight decay, dropout, and early stopping."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timedelta,timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
STARTED=time.perf_counter()
import numpy as np
import torch
from figures import write_figures
ROOT=Path(__file__).resolve().parents[2]
METHODS={'baseline':{'dropout':0.0,'weight_decay':0.0},'weight_decay':{'dropout':0.0,'weight_decay':0.05},'dropout':{'dropout':0.2,'weight_decay':0.0}}
SEEDS=[11,22,33]


class Regressor(torch.nn.Module):
    def __init__(self,p):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(1,48),torch.nn.ReLU(),torch.nn.Dropout(p),
            torch.nn.Linear(48,48),torch.nn.ReLU(),torch.nn.Dropout(p),torch.nn.Linear(48,1))
        self.double()
    def forward(self,x):return self.net(x)


def initial_state(seed):
    torch.manual_seed(seed)
    model=Regressor(0)
    for layer in [model.net[0],model.net[3]]:
        torch.nn.init.kaiming_normal_(layer.weight,mode='fan_in',nonlinearity='relu')
        torch.nn.init.zeros_(layer.bias)
    torch.nn.init.xavier_uniform_(model.net[6].weight)
    torch.nn.init.zeros_(model.net[6].bias)
    return copy.deepcopy(model.state_dict())


def state_hash(state):
    digest=hashlib.sha256()
    for key,value in state.items():digest.update(key.encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():return float(torch.nn.functional.mse_loss(model(x),y))


def train(method,seed,state,train_x,train_y,val_x,val_y,folder,*,replay=False):
    config=METHODS[method]
    model=Regressor(config['dropout']);model.load_state_dict(state)
    assert sum(p.numel() for p in model.parameters())==2497
    weights=[p for p in model.parameters() if p.ndim>1]
    biases=[p for p in model.parameters() if p.ndim==1]
    optimizer=torch.optim.AdamW([{'params':weights,'weight_decay':config['weight_decay']},
        {'params':biases,'weight_decay':0.0}],lr=0.01,betas=(0.9,0.999),eps=1e-8)
    torch.manual_seed(seed+100000)
    history_train=[evaluate(model,train_x,train_y)]
    history_val=[evaluate(model,val_x,val_y)]
    best_value=history_val[0];best_epoch=0;best_state=copy.deepcopy(model.state_dict());bad=0
    for epoch in range(1,801):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.mse_loss(model(train_x),train_y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
        loss.backward();optimizer.step()
        train_error=evaluate(model,train_x,train_y);val_error=evaluate(model,val_x,val_y)
        history_train.append(train_error);history_val.append(val_error)
        if val_error<best_value-1e-5:
            best_value=val_error;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        if bad>=120:break
    final_train=history_train[-1];final_val=history_val[-1]
    model.load_state_dict(best_state);model.eval()
    restored_val=evaluate(model,val_x,val_y)
    np.testing.assert_allclose(restored_val,best_value,rtol=1e-13,atol=1e-13)
    assert restored_val<=final_val+1e-5
    checkpoint=folder/f'{"replay-" if replay else ""}{method}-seed{seed}.pt'
    torch.save({'state_dict':best_state,'method':method,'dropout':config['dropout'],'seed':seed,
        'best_epoch':best_epoch,'best_val':best_value},checkpoint)
    reloaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    restored=Regressor(config['dropout']);restored.load_state_dict(reloaded['state_dict']);restored.eval()
    with torch.no_grad():torch.testing.assert_close(restored(val_x),model(val_x),rtol=0,atol=0)
    grid=np.linspace(-1,1,101,dtype=np.float64).reshape(-1,1)
    with torch.no_grad():fitted=model(torch.from_numpy(grid)).numpy()[:,0]
    return {'method':method,'seed':seed,'initial_sha256':state_hash(state),'parameter_count':2497,
        'stop_epoch':len(history_train)-1,'best_epoch':best_epoch,'best_val_mse':best_value,
        'best_train_mse':evaluate(model,train_x,train_y),'final_train_mse':final_train,'final_val_mse':final_val,
        'train_eval_mse':history_train,'validation_mse':history_val,'grid':grid[:,0].tolist(),
        'best_prediction':fitted.tolist(),'checkpoint':str(checkpoint.relative_to(ROOT)),
        'checkpoint_prediction_exact':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'build/experiments/10-generalization/results.json')
    parser.add_argument('--skip-figures',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    begun=time.perf_counter()
    rng=np.random.default_rng(20261115)
    datasets={}
    digest=hashlib.sha256()
    for name,n in [('train',20),('validation',128),('test',256)]:
        x=rng.uniform(-1,1,size=(n,1)).astype(np.float64)
        y=np.sin(3*x)+rng.normal(0,0.2,size=(n,1))
        datasets[name]=(x,y);digest.update(name.encode());digest.update(x.tobytes());digest.update(y.tobytes())
    assert len(np.unique(np.concatenate([d[0][:,0] for d in datasets.values()])))==404
    train_x,train_y=[torch.from_numpy(v) for v in datasets['train']]
    val_x,val_y=[torch.from_numpy(v) for v in datasets['validation']]
    folder=ROOT/'build/experiments/10-generalization/checkpoints';folder.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in SEEDS:
        state=initial_state(seed)
        for method in METHODS:
            result=train(method,seed,state,train_x,train_y,val_x,val_y,folder)
            runs.append(result)
            print(f"{method} seed={seed}: stopped={result['stop_epoch']} best={result['best_epoch']} val={result['best_val_mse']:.6f}",flush=True)
    for seed in SEEDS:assert len({r['initial_sha256'] for r in runs if r['seed']==seed})==1
    replay=train('baseline',11,initial_state(11),train_x,train_y,val_x,val_y,folder,replay=True)
    first=runs[0]
    np.testing.assert_array_equal(replay['train_eval_mse'],first['train_eval_mse'])
    np.testing.assert_array_equal(replay['validation_mse'],first['validation_mse'])
    np.testing.assert_array_equal(replay['best_prediction'],first['best_prediction'])
    # Final evaluation phase begins only after the predeclared runs and replay are fixed.
    test_x,test_y=[torch.from_numpy(v) for v in datasets['test']]
    for run in runs:
        model=Regressor(METHODS[run['method']]['dropout'])
        checkpoint=torch.load(ROOT/run['checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict']);run['test_mse']=evaluate(model,test_x,test_y)
    design=np.column_stack((datasets['train'][0][:,0],np.ones(20)))
    coefficients=np.linalg.lstsq(design,datasets['train'][1][:,0],rcond=None)[0]
    linear={name:float(np.mean((np.column_stack((x[:,0],np.ones(x.shape[0])))@coefficients-y[:,0])**2)) for name,(x,y) in datasets.items()}
    h=np.array([2.,4.,6.,8.]);masks=np.array(list(itertools.product([0.,1.],repeat=4)))
    expected_dropout=(masks*h/0.5).mean(axis=0)
    np.testing.assert_array_equal(expected_dropout,h)
    torch.manual_seed(999);drop=torch.nn.Dropout(0.5);drop.train()
    example=drop(torch.from_numpy(h)).numpy()
    drop.eval();np.testing.assert_array_equal(drop(torch.from_numpy(h)).numpy(),h)
    summaries={}
    for method in METHODS:
        group=[r for r in runs if r['method']==method]
        summaries[method]={key:{'mean':float(np.mean([r[key] for r in group])),
            'sample_std':float(np.std([r[key] for r in group],ddof=1))}
            for key in ['best_val_mse','test_mse','best_train_mse']}
    report={'chapter':'10','passed':True,'run_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'os':platform.platform(),'device':'cpu','dtype':'float64','threads':1,'deterministic_algorithms':True},
        'configuration':{'data_seed':20261115,'initialization_seeds':SEEDS,'training_seed_offset':100000,
            'methods':METHODS,'max_epochs':800,'patience':120,'minimum_improvement':1e-5,
            'optimizer':'AdamW','learning_rate':0.01,'betas':[0.9,0.999],'eps':1e-8,
            'weight_decay_scope':'Weights only; biases excluded. AdamW is decoupled, not an L2 term in MSE.',
            'training_batch_size':20,'metric':'Mean squared error, all logged in eval mode without regularization terms.',
            'test_protocol':'One final evaluation phase after all fixed method/seed runs; no test-driven retuning.'},
        'data':{'sha256':digest.hexdigest(),'recipe':'x~Uniform[-1,1], y=sin(3*x)+Normal(0,0.2^2)',
            'shapes':{name:list(pair[0].shape) for name,pair in datasets.items()},
            'train_x':datasets['train'][0][:,0].tolist(),'train_y':datasets['train'][1][:,0].tolist()},
        'runs':runs,'summary':summaries,'linear_reference':{'coefficients':coefficients.tolist(),'mse':linear},
        'dropout_check':{'p':0.5,'input':h.tolist(),'one_sample':example.tolist(),
            'all_16_masks_expectation':expected_dropout.tolist(),'eval_output':h.tolist()},
        'replay':{'method':'baseline','seed':11,'curves_and_predictions_exact':True},
        'resources':{'training_and_checks_seconds':time.perf_counter()-begun,
            'including_imports_seconds':time.perf_counter()-STARTED,
            'tensor_data_bytes':sum(x.nbytes+y.nbytes for x,y in datasets.values()),
            'memory_scope':'Array payload only; process peak not measured. SVG generation excluded.'}}
    if not args.skip_figures:write_figures(report,ROOT/'docs/assets/10-generalization')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2));print('PASS; report='+str(args.output.resolve()))


if __name__=='__main__':main()
