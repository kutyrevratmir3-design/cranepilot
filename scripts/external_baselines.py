"""Matched predictive references; adapted method families, not exact paper reproductions."""
import copy,time
import numpy as np
import torch
from torch import nn
from scipy.linalg import expm
from scipy.optimize import least_squares
from cranepilot.physics import LengthCalibration
from cranepilot.forced_observer import forced_observer,analytic_rollout
from scripts.predict_cranepilot import build_packet
from scripts.predict_research_bundle import ResearchBundle
from scripts.run_broad_search import inputs,predict


def radius(cfg):return cfg['boom_tip_height_m']*np.tan(cfg['boom_elevation_rad'])


def geometric(p,cfg):
    omega=np.full(len(p['theta']),np.sqrt(9.81/cfg['rope_length_m']))
    q,v,_=forced_observer(p['theta'],p['x'],omega,dt=cfg['dt_s'])
    return analytic_rollout(q,v,omega,radius(cfg)*p['future_acceleration'],dt=cfg['dt_s'])


def oscillator(p,cfg,parameters):
    omega,damping,gain=parameters;A=np.array([[0.,1.],[-omega**2,-damping]])
    T=expm(A*cfg['dt_s']);B=np.linalg.solve(A,T-np.eye(2))@np.array([0.,-gain/cfg['rope_length_m']])
    n=len(p['theta']);past=p['theta'][:,::-1];u=radius(cfg)*p['past_acceleration'][:,::-1]
    basis=np.stack([np.linalg.matrix_power(T,i)[0] for i in range(past.shape[1])]);part=np.zeros((n,2));particular=[part[:,0].copy()]
    for j in range(past.shape[1]-1):part=part@T.T+u[:,j,None]*B;particular.append(part[:,0].copy())
    coef=np.linalg.solve(basis.T@basis+1e-10*np.eye(2),((past-np.stack(particular,1))@basis).T).T
    state=coef@np.linalg.matrix_power(T,past.shape[1]-1).T+part;out=[]
    for a in (radius(cfg)*p['future_acceleration']).T:state=state@T.T+a[:,None]*B;out.append(state[:,0].copy())
    return np.stack(out,1)


def fit_oscillator(train,cfg):
    c=cfg['pem'];omega=np.sqrt(9.81/cfg['rope_length_m']);index=cfg['horizon_index'];weights=np.zeros(len(train['theta']))
    for sid in np.unique(train['scenario']):
        m=train['scenario']==sid;weights[m]=1/np.sqrt(m.sum())
    low=[omega*c['omega_ratio_bounds'][0],c['damping_bounds'][0],c['gain_bounds'][0]];high=[omega*c['omega_ratio_bounds'][1],c['damping_bounds'][1],c['gain_bounds'][1]]
    fits=[]
    def residual(parameters):return ((oscillator(train,cfg,parameters)[:,index]-train['target'][:,index])*weights[:,None]/.1).ravel()
    for ratio in c['start_omega_ratios']:
        fit=least_squares(residual,[omega*ratio,.05,1.],bounds=(low,high),max_nfev=c['max_nfev'])
        fits.append(dict(parameters=fit.x.tolist(),loss=float(np.mean(residual(fit.x)**2)),nfev=int(fit.nfev),success=bool(fit.success)))
    return min(fits,key=lambda f:f['loss'])['parameters'],fits


def history_features(p):return np.column_stack([p['theta']/.1,(p['x']-p['x'][:,:1])/.1,p['past_velocity']/.2])


def design(p,j):
    return np.column_stack([history_features(p),p['future_velocity'][:,:j+1]/.2,p['future_acceleration'][:,:j+1]/.5])


def fit_arx(train,cfg,alpha):
    models=[];base=geometric(train,cfg);weight=np.zeros(len(train['theta']))
    for sid in np.unique(train['scenario']):
        mask=train['scenario']==sid;weight[mask]=len(weight)/(len(np.unique(train['scenario']))*mask.sum())
    for j in range(cfg['steps']):
        X=design(train,j);mean=X.mean(0);scale=X.std(0).clip(1e-6);X=(X-mean)/scale;X=np.column_stack([np.ones(len(X)),X]);y=train['target'][:,j]-base[:,j]
        penalty=np.eye(X.shape[1])*alpha;penalty[0,0]=0
        coef=np.linalg.solve((X.T*weight)@X/len(X)+penalty,X.T@(weight*y)/len(X))
        models.append(dict(mean=mean,scale=scale,coef=coef))
    return models


def predict_arx(models,p,cfg):
    result=geometric(p,cfg)
    for j,m in enumerate(models):result[:,j]+=np.column_stack([np.ones(len(result)),(design(p,j)-m['mean'])/m['scale']])@m['coef']
    return result


def save_arx(path,models):np.savez_compressed(path,**{f'{key}_{j}':value for j,m in enumerate(models) for key,value in m.items()})

def load_arx(path,steps=30):
    with np.load(path) as z:return [{key:z[f'{key}_{j}'] for key in ['mean','scale','coef']} for j in range(steps)]


class ResidualLSTM(nn.Module):
    def __init__(self,hidden=32):
        super().__init__();self.encoder=nn.LSTM(3,hidden,batch_first=True);self.decoder=nn.LSTMCell(5,hidden);self.output=nn.Linear(hidden,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,b):
        _,(h,c)=self.encoder(b['history']);h,c=h[0],c[0];out=[]
        for j in range(b['future'].shape[1]):
            h,c=self.decoder(b['future'][:,j],(h,c));out.append(b['base'][:,j]+.1*self.output(h)[:,0])
        return torch.stack(out,1)


def lstm_inputs(p,cfg):
    history=np.stack([p['theta']/.1,(p['x']-p['x'][:,:1])/.1,p['past_velocity']/.2],2)[:,::-1].copy();base=geometric(p,cfg)
    phase=np.sqrt(9.81/cfg['rope_length_m'])*.05*np.arange(1,31)
    future=np.stack([p['future_velocity']/.2,p['future_acceleration']/.5,base/.1,np.broadcast_to(np.sin(phase),base.shape),np.broadcast_to(np.cos(phase),base.shape)],2)
    return {key:torch.as_tensor(value,dtype=torch.float32) for key,value in dict(history=history,future=future,base=base).items()}


@torch.no_grad()
def predict_lstm(model,p,cfg):
    b=lstm_inputs(p,cfg);model.eval();out=[]
    for start in range(0,len(p['theta']),256):out.append(model({k:v[start:start+256] for k,v in b.items()}).numpy())
    return np.concatenate(out)


def score(y,p,cfg):
    values=[]
    for sid in np.unique(p['scenario']):
        m=p['scenario']==sid;values.append(np.abs(np.rad2deg(y[m][:,cfg['horizon_index']]-p['target'][m][:,cfg['horizon_index']])).mean())
    return float(np.mean(values))


def fit_lstm(train,validation,cfg,seed):
    torch.manual_seed(seed);model=ResidualLSTM(cfg['lstm']['hidden']);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lstm']['learning_rate'],weight_decay=cfg['lstm']['weight_decay'])
    b=lstm_inputs(train,cfg);target=torch.as_tensor(train['target'],dtype=torch.float32);index=cfg['horizon_index'];rng=np.random.default_rng(seed)
    best=score(predict_lstm(model,validation,cfg),validation,cfg);state=copy.deepcopy(model.state_dict());history=[dict(epoch=-1,validation_mae_deg=best)];stale=0;selected=-1
    weights=np.zeros(len(target))
    for sid in np.unique(train['scenario']):
        mask=train['scenario']==sid;weights[mask]=len(target)/(len(np.unique(train['scenario']))*mask.sum())
    weights=torch.as_tensor(weights,dtype=torch.float32)
    for epoch in range(cfg['lstm']['epochs']):
        model.train();losses=[]
        for ids in np.array_split(rng.permutation(len(target)),int(np.ceil(len(target)/cfg['lstm']['batch_size']))):
            y=model({k:v[ids] for k,v in b.items()});loss=(((y[:,index]-target[ids][:,index])/.1)**2).mean(1);loss=(loss*weights[ids]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite LSTM fit')
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();losses.append(loss.item())
        value=score(predict_lstm(model,validation,cfg),validation,cfg);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_mae_deg=value))
        if value<best-1e-7:best=value;state=copy.deepcopy(model.state_dict());stale=0;selected=epoch
        else:stale+=1
        if epoch%10==0:print('LSTM',seed,epoch,'val_mae_deg',round(value,6),flush=True)
        if stale>=cfg['lstm']['patience']:break
    model.load_state_dict(state);model.eval();return model,history,dict(seed=seed,selected_epoch=selected,validation_mae_deg=best)


class FrozenAngular:
    def __init__(self,profile):self.bundle=ResearchBundle(fold='primary',profile=profile)
    def forecast(self,p,cfg):
        # Strict previous-sample hold: at -10ms use the observation from -50ms.
        lag=(np.arange(41)+4)//5;theta=p['theta'][:,lag].copy();x=p['x'][:,lag].copy();zeros=np.zeros_like(theta)
        raw=build_packet(theta,x,zeros,np.full(len(theta),cfg['rope_length_m']),np.zeros((len(theta),30)),LengthCalibration())
        b=inputs(raw);b['acceleration']=torch.as_tensor(radius(cfg)*p['future_acceleration'],dtype=torch.float32)
        return np.stack([predict(m,b) for m in self.bundle.models]).mean(0)
