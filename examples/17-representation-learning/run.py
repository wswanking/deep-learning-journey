"""Chapter 17: PCA and nonlinear bottlenecks on a synthetic 3D curve."""
from __future__ import annotations
import argparse,copy,hashlib,json,platform,time
from datetime import datetime
from pathlib import Path
START=time.perf_counter()
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
DT=torch.float64

class Autoencoder(nn.Module):
    def __init__(self,latent):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,latent))
        self.decoder=nn.Sequential(nn.Linear(latent,16),nn.Tanh(),nn.Linear(16,3))
    def forward(self,x):return self.decoder(self.encoder(x))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--skip-figures',action='store_true')
    args=parser.parse_args()
    if args.steps<1:raise ValueError('positive steps required')
    torch.set_num_threads(1);torch.set_default_dtype(DT);torch.use_deterministic_algorithms(True)
    begin=time.perf_counter();folder=ROOT/'build/experiments/17-representation-learning'
    folder.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    # Exact centered hand example, covariance diag(4,.25), PCA-1 keeps x1.
    hand=np.array([[2.,.5],[2.,-.5],[-2.,.5],[-2.,-.5]])
    cov=hand.T@hand/4
    check('hand_covariance',np.array_equal(cov,np.diag([4.,.25])))
    projection=np.array([[1.,0.],[0.,0.]])
    check('hand_per_coordinate_MSE',float(np.mean((hand-hand@projection)**2))==.125)
    # The AE code is not unique: change latent coordinates and invert in decoder.
    E=np.array([[1.,2.],[-1.,1.]]);D=np.array([[2.,0.],[1.,1.]])
    A=np.array([[2.,1.],[0.,3.]]);x=np.array([.3,-.7])
    check('invertible_latent_reparameterization',np.allclose(D@E@x,(D@np.linalg.inv(A))@(A@E)@x))
    rng=np.random.default_rng(20261117);raw=[];times=[]
    for n in [128,64,64]:
        t=rng.uniform(-np.pi,np.pi,n)
        points=np.stack([np.cos(t),np.sin(t),t/np.pi],axis=1)+rng.normal(0,.02,(n,3))
        raw.append(points);times.append(t)
    mean=raw[0].mean(axis=0);centered=[a-mean for a in raw]
    check('training_center_only',np.allclose(centered[0].mean(axis=0),0,atol=1e-14))
    X=[torch.from_numpy(a) for a in centered]
    C=centered[0].T@centered[0]/len(raw[0])
    values,vectors=np.linalg.eigh(C);order=np.argsort(values)[::-1]
    values=values[order];vectors=vectors[:,order]
    check('orthonormal_PCA',np.allclose(vectors.T@vectors,np.eye(3),atol=1e-14))
    check('covariance_eigenpairs',np.allclose(C@vectors,vectors*values,atol=1e-14))
    pca=[];pca_predictions={}
    # Fit all transforms on training only. Test evaluation follows all NN selections.
    for latent in [1,2,3]:
        Q=vectors[:,:latent];train_rec=centered[0]@Q@Q.T
        check('PCA_training_spectral_error_'+str(latent),
              np.isclose(np.mean((train_rec-centered[0])**2),sum(values[latent:])/3,atol=1e-14))
        pca.append({'latent':latent,'train_mse':float(np.mean((train_rec-centered[0])**2)),
                    'validation_mse':float(np.mean((centered[1]@Q@Q.T-centered[1])**2))})
        pca_predictions[latent]=Q
    def train(latent,seed):
        torch.manual_seed(seed);model=Autoencoder(latent).to(dtype=DT)
        check('parameter_count_'+str(latent)+'_'+str(seed),sum(p.numel() for p in model.parameters())==131+33*latent)
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        curves=[];best=float('inf');state=None;best_step=0
        for step in range(args.steps+1):
            if step%20==0 or step==args.steps:
                model.eval()
                with torch.no_grad():
                    tr=float((model(X[0])-X[0]).square().mean())
                    va=float((model(X[1])-X[1]).square().mean())
                curves.append({'step':step,'train_mse':tr,'validation_mse':va})
                if va<best:best=va;state=copy.deepcopy(model.state_dict());best_step=step
            if step==args.steps:break
            model.train();opt.zero_grad(set_to_none=True)
            loss=(model(X[0])-X[0]).square().mean();loss.backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise AssertionError('nonfinite gradient')
            opt.step()
        check('training_reduces_loss_'+str(latent)+'_'+str(seed),curves[-1]['train_mse']<curves[0]['train_mse'])
        model.load_state_dict(state);model.eval()
        checkpoint=folder/'checkpoints'/f'ae-r{latent}-seed{seed}.pt'
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state':state,'mean':torch.from_numpy(mean),'latent':latent},checkpoint)
        restored=Autoencoder(latent).to(dtype=DT)
        saved=torch.load(checkpoint,weights_only=True);restored.load_state_dict(saved['state']);restored.eval()
        with torch.no_grad():
            check('restore_exact_'+str(latent)+'_'+str(seed),torch.equal(model(X[1]),restored(X[1])))
        return model,{'latent':latent,'seed':seed,'parameters':131+33*latent,'best_step':best_step,
                      'validation_mse':best,'curves':curves,'restored_exact':True}
    models=[];runs=[]
    for latent in [1,2,3]:
        for seed in [11,22,33]:
            model,report=train(latent,seed);models.append(model);runs.append(report)
    repeated,replay=train(1,11)
    check('same_seed_curves_exact',replay['curves']==runs[0]['curves'])
    with torch.no_grad():
        check('same_seed_validation_predictions_exact',torch.equal(repeated(X[1]),models[0](X[1])))
        for model,report in zip(models,runs):
            report['train_mse']=float((model(X[0])-X[0]).square().mean())
            prediction=model(X[2]);report['test_mse']=float((prediction-X[2]).square().mean())
            if report['seed']==11:
                report['test_reconstruction']=(prediction+torch.from_numpy(mean)).tolist()
                report['test_latent']=model.encoder(X[2]).tolist()
        for report in pca:
            Q=pca_predictions[report['latent']]
            pred=centered[2]@Q@Q.T
            report['test_mse']=float(np.mean((pred-centered[2])**2))
            report['test_reconstruction']=(pred+mean).tolist()
    # Construct tied linear encoder/decoder from PCA; exact reconstruction identity.
    Q=pca_predictions[1]
    linear=nn.Sequential(nn.Linear(3,1,bias=False),nn.Linear(1,3,bias=False))
    with torch.no_grad():
        linear[0].weight.copy_(torch.from_numpy(Q.T));linear[1].weight.copy_(torch.from_numpy(Q))
        check('tied_linear_AE_equals_PCA',torch.allclose(linear(X[2]),torch.from_numpy(centered[2]@Q@Q.T),rtol=0,atol=1e-15))
    check('PCA_full_dimension_reconstructs',pca[-1]['test_mse']<1e-28)
    summaries=[]
    for latent in [1,2,3]:
        metrics=np.array([r['test_mse'] for r in runs if r['latent']==latent])
        summaries.append({'latent':latent,'test_mean':float(metrics.mean()),'test_sample_std':float(metrics.std(ddof=1))})
    report={'chapter':'17','status':'passed','run_at':datetime.now().astimezone().isoformat(timespec='seconds'),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':1},
        'protocol':{'data_seed':20261117,'counts':[128,64,64],'data':'(cos(t),sin(t),t/pi) plus iid N(0,.02^2); t Uniform[-pi,pi]',
                    'center':'training mean only; no variance scaling','latent_sizes':[1,2,3],'seeds':[11,22,33],
                    'architecture':'3->16 Tanh->r->16 Tanh->3','steps':args.steps,'batch_size':128,
                    'optimizer':'Adam lr .01 betas .9/.999 epsilon 1e-8, no decay','validation_interval':20,
                    'selection':'minimum validation MSE among logged steps including zero; no early stopping',
                    'test':'all fixed runs selected before final evaluation; no test-driven choice','MSE':'mean over samples and all 3 coordinates'},
        'data':{'mean':mean.tolist(),'sha256':[hashlib.sha256(a.tobytes()).hexdigest() for a in raw],
                'array_payload_bytes':sum(a.nbytes for a in raw),'test':raw[2].tolist(),'test_t':times[2].tolist()},
        'pca_eigenvalues':values.tolist(),'pca':pca,'runs':runs,'summary':summaries,
        'checks':checks,'check_count':len(checks),
        'resources':{'training_checks_seconds':time.perf_counter()-begin,'including_import_seconds':time.perf_counter()-START,'process_peak_memory':'not measured'}}
    (folder/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not args.skip_figures:
        from figures import write_figures
        write_figures(report,ROOT/'docs/assets/17-representation-learning')
    print(json.dumps({'status':'passed','checks':len(checks),'pca':[{k:r[k] for k in ['latent','train_mse','validation_mse','test_mse']} for r in pca],'AE':summaries,'resources':report['resources']},indent=2))
if __name__=='__main__':main()
