"""Chapter 18: tiny probabilistic VAE and adversarial GAN on synthetic 2D mixtures, CPU only."""
from __future__ import annotations
import argparse, copy, json, math, platform, sys, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from torch import nn
OBS_STD=.25
CENTERS=torch.tensor([[-2.,0.],[2.,0.]],dtype=torch.float64)


def data(count,seed):
    rng=torch.Generator().manual_seed(seed)
    choices=torch.randint(0,2,(count,),generator=rng)
    return CENTERS[choices]+.2*torch.randn((count,2),generator=rng,dtype=torch.float64)


class VAE(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(2,16),nn.Tanh()); self.mu=nn.Linear(16,1); self.logvar=nn.Linear(16,1); self.decoder=nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,2))
    def encode(self,x):
        h=self.encoder(x); return self.mu(h),self.logvar(h)
    def forward(self,x,epsilon):
        mu,logvar=self.encode(x); z=mu+torch.exp(.5*logvar)*epsilon
        return self.decoder(z),mu,logvar


def vae_terms(x,reconstruction,mu,logvar):
    nll=((x-reconstruction)**2).sum(-1)/(2*OBS_STD**2)+math.log(2*math.pi*OBS_STD**2)
    kl=.5*(mu.square()+logvar.exp()-1-logvar).sum(-1)
    return nll.mean(),kl.mean()


@torch.no_grad()
def vae_evaluate(model,x,eps):
    reconstruction,mu,logvar=model(x,eps); nll,kl=vae_terms(x,reconstruction,mu,logvar)
    mean_reconstruction=model.decoder(mu)
    return {'negative_elbo_mc':float(nll+kl),'reconstruction_nll_mc':float(nll),'kl':float(kl),'posterior_mean_decode_mse':float((mean_reconstruction-x).square().mean()),'mean_abs_encoder_mu':float(mu.abs().mean())}


def coverage(samples):
    distance=torch.cdist(samples,CENTERS); nearest=distance.argmin(-1); close=distance.min(-1).values<=.75
    counts=[int(((nearest==index)&close).sum()) for index in range(2)]
    return {'samples':len(samples),'left_close_count':counts[0],'right_close_count':counts[1],'within_radius_075_fraction':float(close.double().mean()),'covered_modes_at_5_percent':sum(value>=.05*len(samples) for value in counts),'sample_coordinate_std':samples.std(0,correction=0).tolist()}


def train_vae(train,validation,seed,beta):
    torch.manual_seed(seed); model=VAE().double(); rng=torch.Generator().manual_seed(seed+1000); optimizer=torch.optim.Adam(model.parameters(),lr=.01,foreach=False)
    eval_rng=torch.Generator().manual_seed(550); evaluation_eps=torch.randn((len(validation),1),generator=eval_rng,dtype=torch.float64)
    history=[{'step':0,**vae_evaluate(model,validation,evaluation_eps)}]
    for step in range(600):
        indices=torch.randint(len(train),(64,),generator=rng); x=train[indices]; eps=torch.randn((64,1),generator=rng,dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True); reconstruction,mu,logvar=model(x,eps); nll,kl=vae_terms(x,reconstruction,mu,logvar); loss=nll+beta*kl; loss.backward(); optimizer.step()
        if (step+1)%20==0: history.append({'step':step+1,**vae_evaluate(model,validation,evaluation_eps)})
    sampler=torch.Generator().manual_seed(650); z=torch.randn((256,1),generator=sampler,dtype=torch.float64)
    with torch.no_grad():
        means=model.decoder(z); generated=means+OBS_STD*torch.randn((256,2),generator=sampler,dtype=torch.float64); mu,_=model.encode(validation[:64]); reconstruction=model.decoder(mu)
    return model,{'seed':seed,'beta':beta,'parameters':sum(p.numel() for p in model.parameters()),'history':history,'final':history[-1], 'generated_means':means.tolist(),'generated_samples':generated.tolist(),'generation':coverage(generated),'reconstruction_inputs':validation[:64].tolist(),'posterior_mean_reconstructions':reconstruction.tolist()}


class Generator(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,2))
    def forward(self,z): return self.net(z)


class ConstantGenerator(nn.Module):
    def __init__(self): super().__init__(); self.location=nn.Parameter(torch.zeros(2,dtype=torch.float64))
    def forward(self,z): return self.location.expand(len(z),2)


def discriminator(): return nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,1)).double()


def train_gan(train,seed,constant=False):
    torch.manual_seed(seed); g=ConstantGenerator() if constant else Generator().double(); d=discriminator(); rng=torch.Generator().manual_seed(seed+2000)
    opt_g=torch.optim.Adam(g.parameters(),lr=.003,betas=(.5,.999),foreach=False); opt_d=torch.optim.Adam(d.parameters(),lr=.003,betas=(.5,.999),foreach=False); bce=nn.BCEWithLogitsLoss(); history=[]
    first_checks={}
    for step in range(1000):
        real=train[torch.randint(len(train),(64,),generator=rng)]; noise=torch.randn((64,2),generator=rng,dtype=torch.float64)
        opt_d.zero_grad(set_to_none=True); opt_g.zero_grad(set_to_none=True)
        loss_d=bce(d(real),torch.ones((64,1),dtype=torch.float64))+bce(d(g(noise).detach()),torch.zeros((64,1),dtype=torch.float64)); loss_d.backward()
        if step==0: first_checks['generator_has_no_gradient_during_detached_d_step']=all(p.grad is None for p in g.parameters())
        opt_d.step(); opt_d.zero_grad(set_to_none=True)
        for parameter in d.parameters(): parameter.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True); fresh_noise=torch.randn((64,2),generator=rng,dtype=torch.float64)
        loss_g=bce(d(g(fresh_noise)),torch.ones((64,1),dtype=torch.float64)); loss_g.backward()
        if step==0:
            first_checks['discriminator_has_no_parameter_gradient_during_g_step']=all(p.grad is None for p in d.parameters())
            first_checks['generator_receives_gradient_through_frozen_discriminator']=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in g.parameters())
        opt_g.step()
        for parameter in d.parameters():parameter.requires_grad_(True)
        if step==0 or (step+1)%20==0:
            history.append({'step':step+1,'discriminator_loss':float(loss_d.detach()),'generator_loss':float(loss_g.detach())})
    sampler=torch.Generator().manual_seed(750)
    with torch.no_grad(): samples=g(torch.randn((256,2),generator=sampler,dtype=torch.float64))
    return g,{'seed':seed,'constant_generator':constant,'generator_parameters':sum(p.numel() for p in g.parameters()),'discriminator_parameters':sum(p.numel() for p in d.parameters()),'history':history,'generated_samples':samples.tolist(),'generation':coverage(samples),'gradient_policy_checks':first_checks}


def hand_demo():
    # A finite latent example makes the ELBO identity exactly checkable.
    prior=np.array([.5,.5]); likelihood=np.array([.8,.2]); q=np.array([.5,.5]); evidence=float(prior@likelihood); posterior=prior*likelihood/evidence
    elbo=float(np.sum(q*(np.log(prior)+np.log(likelihood)-np.log(q)))); gap=float(np.sum(q*np.log(q/posterior)))
    mu=torch.tensor(1.,dtype=torch.float64,requires_grad=True); logvar=torch.tensor(math.log(4),dtype=torch.float64,requires_grad=True); eps=torch.tensor(-.5,dtype=torch.float64)
    z=mu+torch.exp(.5*logvar)*eps; z.backward()
    kl=.5*(1+4-1-math.log(4))
    # Direct analytic real/fake classification losses.
    d_real=.8; d_fake=.2
    return {'discrete_prior':prior.tolist(),'discrete_likelihood':likelihood.tolist(),'discrete_q':q.tolist(),'evidence':evidence,'posterior':posterior.tolist(),'log_evidence':math.log(evidence),'elbo':elbo,'kl_q_posterior':gap,'identity_residual':math.log(evidence)-elbo-gap,
            'gaussian_kl_mu1_var4_to_standard':kl,'reparameterized_z':float(z.detach()),'dz_dmu':float(mu.grad),'dz_dlogvar':float(logvar.grad),'gan_discriminator_loss':-math.log(d_real)-math.log(1-d_fake),'gan_nonsaturating_generator_loss':-math.log(d_fake)}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,default=Path('build/experiments/18-vae-and-gan')); parser.add_argument('--write-figures',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); started=time.perf_counter(); args.output_dir.mkdir(parents=True,exist_ok=True)
    train=data(512,101); validation=data(256,102); vaes=[]; gans=[]; checks=[]
    def require(condition,name):
        if not bool(condition): raise RuntimeError(name)
        checks.append(name)
    for seed in [11,22,33]:
        for beta in [1.,100.]:
            model,result=train_vae(train,validation,seed,beta); vaes.append(result)
            require(all(math.isfinite(row['negative_elbo_mc']) and row['kl']>=-1e-12 for row in result['history']),f'vae_finite_nonnegative_kl_{seed}_{beta}')
            require(result['final']['reconstruction_nll_mc']+beta*result['final']['kl']<result['history'][0]['reconstruction_nll_mc']+beta*result['history'][0]['kl'],f'vae_weighted_objective_decreases_{seed}_{beta}')
            require(result['parameters']==148,f'vae_148_parameters_{seed}_{beta}')
            if seed==11 and beta==1:
                checkpoint=args.output_dir/'vae-seed11.pt'; torch.save(model.state_dict(),checkpoint); loaded=VAE().double(); loaded.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
                with torch.no_grad(): reload_error=float((model.decoder(torch.zeros((1,1),dtype=torch.float64))-loaded.decoder(torch.zeros((1,1),dtype=torch.float64))).abs().max())
                require(reload_error==0,'vae_inference_weights_reload_identically')
        for constant in [False,True]:
            _,result=train_gan(train,seed,constant); gans.append(result)
            require(all(math.isfinite(row['generator_loss']) and math.isfinite(row['discriminator_loss']) for row in result['history']),f'gan_losses_finite_{seed}_{constant}')
            for name,value in result['gradient_policy_checks'].items(): require(value,f'{name}_{seed}_{constant}')
            if constant: require(max(result['generation']['sample_coordinate_std'])<1e-12,f'constant_generator_has_zero_diversity_{seed}')
    demo=hand_demo()
    require(abs(demo['identity_residual'])<1e-14,'finite_elbo_identity_exact')
    require(abs(demo['elbo']+math.log(2.5))<1e-14,'hand_elbo_matches')
    require(abs(demo['reparameterized_z'])<1e-14 and demo['dz_dmu']==1 and abs(demo['dz_dlogvar']+.5)<1e-14,'reparameterization_hand_gradient')
    require(abs(demo['gaussian_kl_mu1_var4_to_standard']-1.3068528194400546)<1e-14,'diagonal_gaussian_kl_hand')
    require(all(r['final']['kl']<.01 for r in vaes if r['beta']==100),'high_kl_weight_posterior_collapse_observed')
    report={'chapter':'18','collected_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'python_executable':sys.executable,'device':'CPU','dtype':'float64','threads':1},
            'data':{'source':'equal 2D Gaussian mixture centered at (-2,0),(2,0), std .2, self generated','train':512,'validation':256,'seeds':[101,102],'test':None},
            'configuration':{'model_seeds':[11,22,33],'vae_steps':600,'vae_batch':64,'vae_optimizer':'Adam lr=.01','vae_latent_dimension':1,'vae_observation_std':OBS_STD,'vae_validation_mc_samples_per_point':1,'vae_validation_noise_seed':550,'vae_betas':[1,100],'gan_steps':1000,'gan_batch':64,'gan_optimizer':'Adam lr=.003 betas(.5,.999), 1D+1G each iteration','gan_latent_dimension':2,'weight_decay':0,'selection':'fixed final step, no validation selection','checkpoint':'VAE inference weights only, no training resume'},
            'vae':vaes,'gan':gans,'hand':demo,'inference_reload_error':reload_error,'resources':{'elapsed_seconds_excluding_imports':time.perf_counter()-started,'vae_parameter_numeric_bytes':148*8,'gan_normal_parameter_numeric_bytes':(82+65)*8,'memory_scope':'parameters only; process peak not measured'},'checks':{'status':'passed','count':len(checks),'names':checks}}
    path=args.output_dir/'results.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.write_figures:
        from figures import write_figures
        write_figures(report,Path(__file__).resolve().parents[2]/'docs/assets/18-vae-and-gan')
    compact={k:report[k] for k in ['collected_at','environment','configuration','hand','resources','checks']}; compact['vae_results']=[{k:r[k] for k in ['seed','beta','final','generation']} for r in vaes];compact['gan_results']=[{k:r[k] for k in ['seed','constant_generator','generator_parameters','discriminator_parameters','generation']} for r in gans]
    print(json.dumps(compact,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
