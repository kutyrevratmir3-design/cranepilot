"""Offline acceleration-command MPC ablation with causal measured history only."""
import copy
import numpy as np
import torch
from scripts.predict_cranepilot import build_packet
from scripts.run_broad_search import inputs
from scripts.predict_research_bundle import ResearchBundle

DT=.05


def prepare_history(theta,x,length,calibration):
    """Native-reference features are explicitly zero in this NEW contract."""
    raw=build_packet(theta,x,np.zeros_like(theta),length,np.zeros((len(theta),30)),calibration)
    batch=inputs(raw)
    t=-np.arange(21)*.01
    quadratic=np.linalg.pinv(np.column_stack([np.ones(21),t,t*t]))[2]*2
    acceleration=np.clip(x[:,:21]@quadratic,-1.5,1.5)
    return batch,torch.as_tensor(np.column_stack([x[:,0],raw['initial_velocity'],acceleration]),dtype=torch.float32)


def nominal_rollout(commands,initial,tau=.1):
    """Exact lagged acceleration/double-integrator ZOH; angular input is step average."""
    x,v,a=initial.unbind(1);decay=np.exp(-DT/tau)
    positions=[];velocities=[];accelerations=[]
    for c in commands.unbind(1):
        dv=c*DT+(a-c)*tau*(1-decay)
        x=x+v*DT+.5*c*DT**2+(a-c)*tau*(DT-tau*(1-decay))
        v=v+dv;a=c+(a-c)*decay
        positions.append(x);velocities.append(v);accelerations.append(dv/DT)
    return torch.stack(positions,1),torch.stack(velocities,1),torch.stack(accelerations,1)


def physical_angle(batch,acceleration):
    q,p=batch['initial_q'],batch['initial_p'];phase=batch['omega']*DT
    c,s=torch.cos(phase),torch.sin(phase);omc=2*torch.sin(phase/2)**2;ys=[]
    for a in acceleration.unbind(1):
        f=-a/9.81;q,p=c*q+s*p+omc*f,-s*q+c*p+s*f;ys.append(q)
    return torch.stack(ys,1)


class AngularModel:
    def __init__(self,name):
        if name not in ('physical','retained','candidate'):raise ValueError(name)
        self.name=name
        bundle=ResearchBundle(fold='primary',profile='candidate' if name=='candidate' else 'retained')
        self.calibration=bundle.calibration;self.models=[] if name=='physical' else bundle.models
        for model in self.models:
            model.eval()
            for p in model.parameters():p.requires_grad_(False)

    def __call__(self,batch,acceleration):
        if self.name=='physical':return physical_angle(batch,acceleration)
        return torch.stack([m(batch,acceleration)[0] for m in self.models]).mean(0)


def expand_plan(levels,committed):
    return torch.cat([committed[:,None],levels.repeat_interleave(5,dim=1)[:,:29]],1)


def objective(model,batch,initial,commands,target,cfg):
    x,v,a=nominal_rollout(commands,initial,cfg['nominal_tau_s'])
    theta=model(batch,a)
    stage=((x-target[:,None])/.05)**2+(theta/.035)**2+.02*(v/.2)**2+.05*(a/.5)**2
    cost=stage.mean(1)+.1*((torch.diff(commands,dim=1)/.2)**2).mean(1)
    cost=cost+5*((x[:,-1]-target)/.05)**2+5*(theta[:,-1]/.035)**2+(v[:,-1]/.1)**2
    features=torch.cat([(x-target[:,None])/.05/np.sqrt(30),theta/.035/np.sqrt(30),
        np.sqrt(.02/30)*v/.2,np.sqrt(.05/30)*a/.5,np.sqrt(.1/29)*torch.diff(commands,dim=1)/.2,
        np.sqrt(5)*(x[:,-1:]-target[:,None])/.05,np.sqrt(5)*theta[:,-1:]/.035,v[:,-1:]/.1],1)
    return cost,dict(x=x,v=v,acceleration=a,theta=theta,features=features)


@torch.no_grad()
def solve(model,batch,initial,committed,warm,target,cfg):
    """One local linearization, box/trust-region QP, exact nonlinear backtracking.

    Central differences perturb each of six levels, never sensor history.
    All episodes are independent; no optimizer gradients or moments cross rows.
    """
    n=len(warm);eps=cfg['jacobian_epsilon'];d=warm.shape[1]
    offsets=torch.cat([torch.zeros(1,d),eps*torch.eye(d),-eps*torch.eye(d)])
    probes=(warm[None]+offsets[:,None]).reshape(-1,d)
    def evaluate(levels,copies):
        local={k:v.repeat((copies,)+(1,)*(v.ndim-1)) for k,v in batch.items()}
        plans=expand_plan(levels,committed.repeat(copies))
        return objective(model,local,initial.repeat(copies,1),plans,target.repeat(copies),cfg)
    _,data=evaluate(probes,1+2*d)
    features=data['features'].reshape(1+2*d,n,-1)
    jac=(features[1:1+d]-features[1+d:]).permute(1,2,0)/(2*eps)
    f=features[0];hessian=2*jac.transpose(1,2)@jac
    linear=2*(jac.transpose(1,2)@f[:,:,None])[:,:,0]
    lip=torch.linalg.eigvalsh(hessian)[:,-1].clamp_min(1e-6)
    low=torch.maximum(-cfg['command_bound']-warm,torch.full_like(warm,-cfg['trust_region']))
    high=torch.minimum(cfg['command_bound']-warm,torch.full_like(warm,cfg['trust_region']))
    delta=torch.zeros_like(warm);y=delta.clone();momentum=1.
    for _ in range(cfg['qp_iterations']):
        grad=(hessian@y[:,:,None])[:,:,0]+linear
        new=torch.maximum(low,torch.minimum(high,y-grad/lip[:,None]))
        next_momentum=(1+np.sqrt(1+4*momentum**2))/2
        y=new+(momentum-1)/next_momentum*(new-delta)
        delta=new;momentum=next_momentum
    fractions=torch.tensor([0.,.25,.5,1.])
    proposals=(warm[None]+fractions[:,None,None]*delta[None]).reshape(-1,d)
    values,pred=evaluate(proposals,4);values=values.reshape(4,n)
    finite=torch.isfinite(values)
    index=torch.where(finite,values,torch.full_like(values,float('inf'))).argmin(0)
    rows=torch.arange(n);best=proposals.reshape(4,n,d)[index,rows]
    fallback=~finite.any(0);best[fallback]=0
    if fallback.any():raise FloatingPointError('No finite plan; preserve failed run')
    plan=expand_plan(best,committed)
    prediction={key:value.reshape(4,n,*value.shape[1:])[index,rows] for key,value in pred.items() if key!='features'}
    shifted=torch.cat([plan[:,2:],plan[:,-1:]],1)[:,::5]
    return plan,shifted,prediction,dict(cost=values[index,rows],warm_cost=values[0],
        nonfinite=~finite.all(0),fallback=fallback)


def plant_rhs(state,command,parameters,disturbance):
    x,v,q,w,a=state.T
    effective=np.sign(command)*np.maximum(np.abs(command)-parameters['deadzone'],0)*parameters['gain']
    return np.column_stack([v,a,w,-parameters['omega']**2*np.sin(q)
        -parameters['forcing']*a*np.cos(q)-parameters['damping']*w+disturbance,
        (effective-a)/parameters['tau']])


def plant_step(state,command,parameters,disturbance,substeps=2):
    h=.01/substeps;out=state.copy()
    for _ in range(substeps):
        k1=plant_rhs(out,command,parameters,disturbance)
        k2=plant_rhs(out+h*k1/2,command,parameters,disturbance)
        k3=plant_rhs(out+h*k2/2,command,parameters,disturbance)
        k4=plant_rhs(out+h*k3,command,parameters,disturbance)
        out+=h*(k1+2*k2+2*k3+k4)/6
    return out
