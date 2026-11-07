"""Physical MPC with factorized uncertainty and causal online actuator identification.

The angular observer stays on its original .4-second history. Only the actuator
identifier receives the explicitly declared two-second position/command memory.
No neural weights or plant ground-truth parameters enter adaptation.
"""
import itertools
import numpy as np
import torch
from scripts.angular_control import (prepare_history as _prepare,expand_plan,AngularModel)


def prepare_history(theta,x,length,calibration):
    batch,initial=_prepare(theta,x,length,calibration)
    batch['length_m']=torch.tensor(np.asarray(length).copy(),dtype=torch.float32)
    return batch,initial


class AdaptiveModel:
    def __init__(self,name,config=None):
        import json
        from pathlib import Path
        self.config=config or json.loads(Path('configs/adaptive_actuator_20261107.json').read_text())
        self.name=name;self.spec=self.config['variants'][name]
        base=AngularModel('physical');self.calibration=base.calibration;self.neural=None
        frequencies=self.config['frequency_grid'] if self.spec.get('frequency') else [1.]
        actuators=self.config['actuator_grid'] if self.spec.get('actuator') else [[.1,1.]]
        laws=['effective','geometric'] if self.spec.get('force') else ['effective']
        self.scenarios=list(itertools.product(frequencies,actuators,laws));self.tau=None;self.gain=None
        self.ever_valid=None;self.observed=None;self.observed_valid=None

    def update(self,position,commands,tick):
        from scripts.actuator_identification import identify,fit_with_parameters
        n=len(position);c=self.config['identification']
        if self.tau is None:self.tau=np.full(n,.1);self.gain=np.ones(n);self.ever_valid=np.zeros(n,bool)
        accepted=np.zeros(n,bool);information=np.zeros(n);rmse=np.zeros(n)
        if self.spec.get('adaptive') and tick%c['update_every_ticks']==0:
            fit=identify(position,commands,c);accepted=fit['valid'];alpha=c['ema']
            self.tau=np.where(accepted,(1-alpha)*self.tau+alpha*fit['tau'],self.tau)
            self.gain=np.where(accepted,(1-alpha)*self.gain+alpha*fit['gain'],self.gain)
            self.ever_valid|=accepted;information=fit['information'];rmse=fit['rmse']
        self.observed=np.zeros((n,2));self.observed_valid=np.zeros(n,bool)
        if self.spec.get('observer') and position.shape[1]>=c['minimum_samples']:
            v,a,error=fit_with_parameters(position,commands,self.tau,self.gain)
            self.observed=np.column_stack([np.clip(v,-.6,.6),np.clip(a,-1.5,1.5)])
            self.observed_valid=self.ever_valid&(error<=c['maximum_fit_rmse_m'])
        return dict(tau=self.tau.copy(),gain=self.gain.copy(),accepted=accepted,ever_valid=self.ever_valid.copy(),information=information,fit_rmse=rmse,
                    observed=self.observed.copy(),observed_valid=self.observed_valid.copy())

    def used_initial(self,initial):
        if not self.spec.get('observer') or self.observed is None:return initial
        result=initial.clone();valid=torch.tensor(self.observed_valid)
        result[:,1:]=torch.where(valid[:,None],torch.tensor(self.observed,dtype=initial.dtype),result[:,1:])
        return result

    @torch.no_grad()
    def context(self,batch,initial,warm_commands):
        initial=self.used_initial(initial)
        n=len(initial);k=len(self.scenarios)
        ratio=torch.tensor([s[0] for s in self.scenarios]);omega=batch['omega'][:,None]*ratio
        force=torch.stack([torch.full((n,),1/9.81) if s[2]=='effective' else 1/(batch['length_m']*omega[:,j].square())
                           for j,s in enumerate(self.scenarios)],1)
        ctx=dict(initial=initial,omega=omega,q=batch['initial_q'][:,None].expand(-1,k),
            p=batch['initial_p'][:,None]/ratio,force=force,offset=torch.zeros(n,30))
        if self.spec.get('adaptive'):
            ctx['tau']=torch.tensor(self.tau,dtype=initial.dtype)[:,None].expand(-1,k)
            ctx['gain']=torch.tensor(self.gain,dtype=initial.dtype)[:,None].expand(-1,k)
        else:
            ctx['tau']=initial.new_tensor([s[1][0] for s in self.scenarios])[None].expand(n,-1)
            ctx['gain']=initial.new_tensor([s[1][1] for s in self.scenarios])[None].expand(n,-1)
        return ctx

    @torch.no_grad()
    def evaluate(self,ctx,commands,target):
        n=len(commands);k=len(self.scenarios)
        tau=ctx['tau'];gain=ctx['gain']
        decay=torch.exp(-.05/tau)
        x,v,a=[ctx['initial'][:,j,None].expand(-1,k).clone() for j in range(3)]
        q,p=ctx['q'].clone(),ctx['p'].clone();phase=ctx['omega']*.05
        cosine,sine=torch.cos(phase),torch.sin(phase);omc=2*torch.sin(phase/2).square()
        xs=[];vs=[];acc=[];angles=[]
        for j in range(30):
            c=commands[:,j,None]*gain
            dv=c*.05+(a-c)*tau*(1-decay)
            x=x+v*.05+.5*c*.05**2+(a-c)*tau*(.05-tau*(1-decay));v=v+dv;a=c+(a-c)*decay
            average=dv/.05;f=-ctx['force']*average
            q,p=cosine*q+sine*p+omc*f,-sine*q+cosine*p+sine*f
            xs.append(x);vs.append(v);acc.append(average);angles.append(q+ctx['offset'][:,j,None])
        x,v,a,theta=[torch.stack(y,2) for y in [xs,vs,acc,angles]]
        features=torch.cat([(x-target[:,None,None])/.05/np.sqrt(30),theta/.035/np.sqrt(30),
            np.sqrt(.02/30)*v/.2,np.sqrt(.05/30)*a/.5,
            np.sqrt(.1/29)*torch.diff(commands,dim=1)[:,None].expand(-1,k,-1)/.2,
            np.sqrt(5)*(x[:,:,-1:]-target[:,None,None])/.05,np.sqrt(5)*theta[:,:,-1:]/.035,v[:,:,-1:]/.1],2)
        features=features.reshape(n,-1)/np.sqrt(k)
        spread=self.spec.get('spread',0.)
        if spread:
            disagreement=theta-theta.mean(1,keepdim=True)
            features=torch.cat([features,np.sqrt(spread/(k*30))*disagreement.reshape(n,-1)/.035],1)
        cost=features.square().sum(1)
        return cost,dict(x=x.mean(1),v=v.mean(1),acceleration=a.mean(1),theta=theta.mean(1),features=features)


@torch.no_grad()
def solve(model,batch,initial,committed,warm,target,cfg):
    n=len(warm);d=6
    context=model.context(batch,initial,expand_plan(warm,committed))
    # Features are AFFINE: identified parameters and state are fixed within this decision.
    # Unit central probes recover that map without small-epsilon cancellation.
    offsets=torch.cat([torch.zeros(1,d),torch.eye(d),-torch.eye(d)])
    def evaluate(levels,copies):
        ctx={key:value.repeat((copies,)+(1,)*(value.ndim-1)) for key,value in context.items()}
        return model.evaluate(ctx,expand_plan(levels,committed.repeat(copies)),target.repeat(copies))
    _,pred=evaluate((warm[None]+offsets[:,None]).reshape(-1,d),13)
    features=pred['features'].reshape(13,n,-1)
    jac=(features[1:7]-features[7:13]).permute(1,2,0)/2
    hessian=2*jac.transpose(1,2)@jac;linear=2*(jac.transpose(1,2)@features[0][:,:,None])[:,:,0]
    lip=torch.linalg.eigvalsh(hessian)[:,-1].clamp_min(1e-6)
    low=torch.maximum(-cfg['command_bound']-warm,torch.full_like(warm,-cfg['trust_region']))
    high=torch.minimum(cfg['command_bound']-warm,torch.full_like(warm,cfg['trust_region']))
    delta=torch.zeros_like(warm);y=delta.clone();momentum=1.
    for _ in range(cfg['qp_iterations']):
        gradient=(hessian@y[:,:,None])[:,:,0]+linear
        new=torch.maximum(low,torch.minimum(high,y-gradient/lip[:,None]))
        next_momentum=(1+np.sqrt(1+4*momentum**2))/2
        y=new+(momentum-1)/next_momentum*(new-delta);delta=new;momentum=next_momentum
    proposals=(warm[None]+torch.tensor([0.,.25,.5,1.])[:,None,None]*delta[None]).reshape(-1,d)
    values,pred=evaluate(proposals,4);values=values.reshape(4,n)
    finite=torch.isfinite(values);index=torch.where(finite,values,torch.full_like(values,float('inf'))).argmin(0);rows=torch.arange(n)
    if (~finite.any(0)).any():raise FloatingPointError('No finite scenario plan')
    best=proposals.reshape(4,n,d)[index,rows];plan=expand_plan(best,committed)
    predicted={key:value.reshape(4,n,*value.shape[1:])[index,rows] for key,value in pred.items() if key!='features'}
    shifted=torch.cat([plan[:,2:],plan[:,-1:]],1)[:,::5]
    return plan,shifted,predicted,dict(cost=values[index,rows],warm_cost=values[0],nonfinite=~finite.all(0),fallback=torch.zeros(n,dtype=torch.bool))
