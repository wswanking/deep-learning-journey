"""CPU DDPM on an original 2-D four-mode Gaussian mixture; no downloads."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import time
START=time.perf_counter()
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];DT=torch.float64
CENTERS=torch.tensor([[-1.4,-1.4],[-1.4,1.4],[1.4,-1.4],[1.4,1.4]],dtype=DT);DATA_STD=.2

def check(test,message):
 if not test:raise AssertionError(message)
def noise(shape,generator):return torch.randn(shape,generator=generator,dtype=DT)
def data(n,generator):return CENTERS[torch.randint(4,(n,),generator=generator)]+DATA_STD*noise((n,2),generator)
def schedule(steps):
 beta=torch.zeros(steps+1,dtype=DT);beta[1:]=torch.linspace(.001,.2,steps,dtype=DT);alpha=1-beta;abar=alpha.cumprod(0);prev=torch.cat([torch.ones(1,dtype=DT),abar[:-1]]);variance=torch.zeros_like(beta);variance[1:]=beta[1:]*(1-prev[1:])/(1-abar[1:]);return beta,alpha,abar,variance

def corrupt(x0,t,eps,abar):return abar[t,None].sqrt()*x0+(1-abar[t,None]).sqrt()*eps
class EpsilonNetwork(nn.Module):
 def __init__(self,steps):
  super().__init__();self.steps=steps;self.net=nn.Sequential(nn.Linear(5,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),nn.Linear(48,2))
 def forward(self,x,t):
  u=t.to(dtype=DT).unsqueeze(-1)/self.steps;time_features=torch.cat([u,torch.sin(math.pi*u),torch.cos(math.pi*u)],-1)
  return self.net(torch.cat([x,time_features],-1))

def oracle_epsilon(x,t,abar):
 # Closed-form mixture denoiser; knows the synthetic density, not any sample's original x0.
 a=abar[t,None];variance=a*DATA_STD**2+1-a;means=a.sqrt()[:,None,:]*CENTERS[None,:,:]
 log_weights=-(x[:,None,:]-means).square().sum(-1)/(2*variance);responsibilities=torch.softmax(log_weights,-1);expected_mean=(responsibilities[:,:,None]*means).sum(1)
 return (1-a).sqrt()/variance*(x-expected_mean)

def sample(predictor,n,sched,seed,keep=False):
 beta,alpha,abar,posterior_variance=sched;steps=len(beta)-1;gen=torch.Generator().manual_seed(seed);x=noise((n,2),gen);snapshots={steps:x[:160].tolist()} if keep else {}
 with torch.no_grad():
  for step in range(steps,0,-1):
   t=torch.full((n,),step,dtype=torch.long);eps=predictor(x,t)
   mean=(x-beta[step]/torch.sqrt(1-abar[step])*eps)/torch.sqrt(alpha[step])
   x=mean+torch.sqrt(posterior_variance[step])*noise((n,2),gen) if step>1 else mean
   if keep and step-1 in [48,32,16,0]:snapshots[step-1]=x[:160].tolist()
 check(bool(torch.isfinite(x).all()),'nonfinite reverse samples');return x,snapshots

def sliced_distance(a,b):
 # Same 32 fixed unit directions and sample counts for all comparisons.
 angle=torch.arange(32,dtype=DT)*math.pi/32;directions=torch.stack([angle.cos(),angle.sin()],-1)
 pa=(a@directions.T).sort(0).values;pb=(b@directions.T).sort(0).values
 return float((pa-pb).abs().mean())
def summarize(x,reference):
 nearest=torch.cdist(x,CENTERS).argmin(-1);counts=torch.bincount(nearest,minlength=4)
 return {'sliced_empirical_w1':sliced_distance(x,reference),'mean':x.mean(0).tolist(),'standard_deviation':x.std(0,unbiased=False).tolist(),'mode_counts':counts.tolist(),'mean_distance_to_nearest_center':float(torch.cdist(x,CENTERS).min(-1).values.mean())}

def mechanics(sched):
 beta,alpha,abar,var=sched;check(abar[0]==1 and var[1]==0,'schedule boundary');check(bool((abar[1:]<abar[:-1]).all()),'cumulative schedule')
 # Marginal moment propagation: mean coefficient and variance independently by recurrence.
 variance=0.;mean_coefficient=1.
 for t in range(1,len(beta)):
  mean_coefficient*=float(alpha[t].sqrt());variance=float(alpha[t])*variance+float(beta[t])
  check(abs(mean_coefficient-float(abar[t].sqrt()))<1e-14 and abs(variance-float(1-abar[t]))<1e-14,'marginal recursion')
 # Given x0 and the actual sampled epsilon, x0 reconstruction and posterior forms agree.
 x0=torch.tensor([[.8,-.6]],dtype=DT);eps=torch.tensor([[.3,-.2]],dtype=DT);step=23;t=torch.tensor([step]);xt=corrupt(x0,t,eps,abar);recover=(xt-torch.sqrt(1-abar[step])*eps)/torch.sqrt(abar[step])
 check(torch.allclose(recover,x0,atol=1e-14,rtol=0),'known-noise reconstruction')
 c0=torch.sqrt(abar[step-1])*beta[step]/(1-abar[step]);ct=torch.sqrt(alpha[step])*(1-abar[step-1])/(1-abar[step])
 posterior=c0*x0+ct*xt;epsilon_mean=(xt-beta[step]/torch.sqrt(1-abar[step])*eps)/torch.sqrt(alpha[step]);check(torch.allclose(posterior,epsilon_mean,rtol=0,atol=1e-14),'posterior mean parameterization')
 # Textbook small scalar hand example, independent of default long schedule.
 b=.1;a=.9;a1=.9;a2=.81;z0=1.;e=.5;zt=math.sqrt(a2)*z0+math.sqrt(1-a2)*e
 posterior_var=b*(1-a1)/(1-a2);mu=(zt-b/math.sqrt(1-a2)*e)/math.sqrt(a)
 # MSE optimum has unavoidable noise uncertainty: oracle knows density but not sample epsilon.
 return {'marginal_recursion':True,'terminal_abar':float(abar[-1]),'known_noise_reconstruction_error':float((recover-x0).abs().max()),'posterior_mean_error':float((posterior-epsilon_mean).abs().max()),'hand':{'beta1':b,'beta2':b,'alpha_bar2':a2,'x0':z0,'epsilon':e,'x2':zt,'reverse_mean':mu,'posterior_variance':posterior_var},'boundary_variance1':float(var[1])}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=2500);parser.add_argument('--skip-figures',action='store_true');args=parser.parse_args();check(args.updates>0,'positive updates')
 torch.set_num_threads(1);torch.set_default_dtype(DT);torch.use_deterministic_algorithms(True);begin=time.perf_counter();steps=64;sched=schedule(steps);beta,alpha,abar,variance=sched;checked=mechanics(sched)
 data_gen=torch.Generator().manual_seed(1900);train=data(2048,data_gen);validation=data(512,data_gen);test=data(1024,data_gen)
 eval_gen=torch.Generator().manual_seed(1902);val_t=torch.randint(1,steps+1,(len(validation),),generator=eval_gen);val_eps=noise(validation.shape,eval_gen);val_xt=corrupt(validation,val_t,val_eps,abar)
 torch.manual_seed(1901);model=EpsilonNetwork(steps).to(dtype=DT);opt=torch.optim.Adam(model.parameters(),lr=.001);train_gen=torch.Generator().manual_seed(1903)
 curves={'updates':[],'validation_noise_mse':[],'training_batch_noise_mse':[]};best=float('inf');best_state=None;best_update=0
 def evaluate(update,last_loss):
  nonlocal best,best_state,best_update
  model.eval()
  with torch.no_grad():value=float((model(val_xt,val_t)-val_eps).square().mean())
  curves['updates'].append(update);curves['validation_noise_mse'].append(value);curves['training_batch_noise_mse'].append(last_loss)
  if value<best:best=value;best_state=copy.deepcopy(model.state_dict());best_update=update
 evaluate(0,None)
 for update in range(1,args.updates+1):
  model.train();index=torch.randint(len(train),(128,),generator=train_gen);x0=train[index];t=torch.randint(1,steps+1,(128,),generator=train_gen);eps=noise(x0.shape,train_gen);xt=corrupt(x0,t,eps,abar)
  opt.zero_grad(set_to_none=True);loss=(model(xt,t)-eps).square().mean();loss.backward();check(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'finite gradients');opt.step()
  if update%100==0 or update==args.updates:evaluate(update,float(loss.detach()))
 model.load_state_dict(best_state);model.eval();training_seconds=time.perf_counter()-begin
 folder=ROOT/'build/experiments/19-diffusion-models';folder.mkdir(parents=True,exist_ok=True);torch.save({'model_state':best_state,'steps':steps,'beta':beta,'protocol':'four-mode-2d-v1'},folder/'denoiser.pt');loaded=EpsilonNetwork(steps).to(dtype=DT);loaded.load_state_dict(torch.load(folder/'denoiser.pt',weights_only=True)['model_state']);loaded.eval()
 with torch.no_grad():check(torch.equal(model(val_xt,val_t),loaded(val_xt,val_t)),'checkpoint restore')
 # Every candidate is fixed before the test sample distribution is used.
 sample_start=time.perf_counter();generated,snapshots=sample(model,1024,sched,1904,True);reverse_seconds=time.perf_counter()-sample_start
 repeated,_=sample(loaded,1024,sched,1904,False);check(torch.equal(generated,repeated),'sampling fixed-seed restoration')
 oracle_samples,_=sample(lambda x,t:oracle_epsilon(x,t,abar),1024,sched,1904)
 prior,_=sample(lambda x,t:torch.zeros_like(x),1024,sched,1904) # Intentionally missing denoising; clear failure control.
 baseline=noise((1024,2),torch.Generator().manual_seed(1904))
 with torch.no_grad():
  oracle_val=float((oracle_epsilon(val_xt,val_t,abar)-val_eps).square().mean());zero_val=float(val_eps.square().mean())
 # Actual forward Markov chain: fresh independent Gaussian increments at every step.
 forward_gen=torch.Generator().manual_seed(1905);forward_x=train[:160].clone();forward={0:forward_x.tolist()}
 for t in range(1,steps+1):
  forward_x=alpha[t].sqrt()*forward_x+beta[t].sqrt()*noise(forward_x.shape,forward_gen)
  if t in [16,32,48,64]:forward[t]=forward_x.tolist()
 reference_gen=torch.Generator().manual_seed(1906);reference_again=data(1024,reference_gen)
 report={'run_at':datetime.now().astimezone().isoformat(timespec='seconds'),'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':1},'protocol':{'data_seed':1900,'model_seed':1901,'validation_corruption_seed':1902,'training_sampling_seed':1903,'generation_seed':1904,'steps':steps,'beta_first':float(beta[1]),'beta_last':float(beta[-1]),'updates':args.updates,'batch_size':128,'optimizer':'Adam','learning_rate':.001,'betas':[.9,.999],'epsilon':1e-8,'loss':'mean squared epsilon error over B*2 coordinates','evaluation_interval':100,'parameters':sum(p.numel() for p in model.parameters()),'data_counts':[2048,512,1024],'centers':CENTERS.tolist(),'component_std':DATA_STD,'sliced_directions':32},'mechanics':checked,'data_hashes':[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in [train,validation,test]],'curves':curves,'best_update':best_update,'best_validation_noise_mse':best,'oracle_validation_noise_mse':oracle_val,'zero_validation_noise_mse':zero_val,'distribution_metrics':{'trained':summarize(generated,test),'analytic_density_denoiser':summarize(oracle_samples,test),'standard_normal':summarize(baseline,test),'zero_denoiser_chain':summarize(prior,test),'independent_real_sample':summarize(reference_again,test)},'forward_snapshots':forward,'reverse_snapshots':snapshots,'test_points':test[:160].tolist(),'generated_points':generated[:160].tolist(),'standard_normal_points':baseline[:160].tolist(),'resources':{'training_and_mechanics_seconds':training_seconds,'one_1024_sample_reverse_seconds':reverse_seconds,'total_checks_seconds':time.perf_counter()-begin,'including_import_seconds':time.perf_counter()-START,'dataset_tensor_bytes':sum(x.numel()*x.element_size() for x in [train,validation,test]),'process_peak_memory':'not measured'},'checks':{'checkpoint_exact':True,'generation_repeated_exact':True,'finite_gradients':True}}
 (folder/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if not args.skip_figures:
  from figures import write_figures
  write_figures(report,ROOT/'docs/assets/19-diffusion-models')
 print(json.dumps({'status':'passed','best_update':best_update,'validation_noise_mse':best,'oracle_validation_noise_mse':oracle_val,'zero_noise_mse':zero_val,'distribution_metrics':report['distribution_metrics'],'mechanics':checked,'resources':report['resources']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
