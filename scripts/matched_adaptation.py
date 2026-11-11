"""Equal-data development: strong ARX, physics-prior NARX and existing angular fine-tuning."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.external_baselines import geometric,oscillator,history_features,design,radius,score
from scripts.external_crane_data import packets,concatenate,load_sequence,train_offset
from scripts.predict_cranepilot import build_packet
from scripts.predict_research_bundle import ResearchBundle,_model
from scripts.run_broad_search import inputs,predict
from cranepilot.physics import LengthCalibration

CONFIG=Path('configs/matched_adaptation_20261107.json')


def features(p,j,basis):
    linear=design(p,j)
    if basis=='linear':return linear
    if basis!='structured':raise ValueError('Unknown fixed basis')
    q=p['theta']/.1;v=p['past_velocity']/.2
    # Small-angle nonlinearity, velocity-coupled history, and known command nonlinearities.
    nonlinear=np.column_stack([(np.sin(p['theta'])-p['theta'])/.0001,q*v,q*v*v,
        p['future_velocity'][:,:j+1]**2/.04,
        p['future_acceleration'][:,:j+1]*np.abs(p['future_acceleration'][:,:j+1])/.25])
    return np.column_stack([linear,nonlinear])


def prior(p,cfg,base):
    if base=='none':return np.zeros_like(p['future_acceleration'])
    if base=='geometric':return geometric(p,cfg)
    if base=='fitted':return oscillator(p,cfg,cfg['oscillator_parameters'])
    raise ValueError('Unknown prior')


def fit_linear(p,cfg,alpha,basis='linear',base='geometric'):
    baseline=prior(p,cfg,base);weight=np.zeros(len(p['theta']));groups=np.unique(p['scenario'])
    for sid in groups:
        mask=p['scenario']==sid;weight[mask]=len(weight)/(len(groups)*mask.sum())
    models=[]
    for j in range(cfg['steps']):
        X=features(p,j,basis);mean=X.mean(0);scale=X.std(0).clip(1e-6);X=np.column_stack([np.ones(len(X)),(X-mean)/scale]);y=p['target'][:,j]-baseline[:,j]
        penalty=np.eye(X.shape[1])*alpha;penalty[0,0]=0
        coef=np.linalg.solve((X.T*weight)@X/len(X)+penalty,X.T@(weight*y)/len(X))
        models.append(dict(mean=mean,scale=scale,coef=coef))
    return dict(basis=basis,base=base,alpha=alpha,models=models)


def linear_predict(m,p,cfg):
    y=prior(p,cfg,m['base'])
    for j,c in enumerate(m['models']):y[:,j]+=np.column_stack([np.ones(len(y)),(features(p,j,m['basis'])-c['mean'])/c['scale']])@c['coef']
    return y


def save_linear(path,m):
    np.savez_compressed(path,basis=np.array(m['basis']),base=np.array(m['base']),alpha=np.array(m['alpha']),
        **{f'{key}_{j}':v for j,c in enumerate(m['models']) for key,v in c.items()})


def load_linear(path):
    with np.load(path) as z:return dict(basis=str(z['basis']),base=str(z['base']),alpha=float(z['alpha']),models=[{key:z[f'{key}_{j}'] for key in ['mean','scale','coef']} for j in range(30)])


def angular_batch(p,cfg):
    omega,damping,gain=cfg['oscillator_parameters'];length=cfg['rope_length_m'];factor=gain*9.81/(length*omega**2)
    # Fixed source-clock frequency estimated on the same TRAIN only as PEM reference.
    calibration=LengthCalibration(a=9.81/(length*omega**2),b=0.)
    lag=(np.arange(41)+4)//5
    # Virtual reference is the dimensionless known velocity, not a native actuator command.
    reference=2*p['past_velocity'][:,lag]/.2;future=2*p['future_velocity']/.2
    raw=build_packet(p['theta'][:,lag].copy(),(factor*p['x'][:,lag]).copy(),reference,
        np.full(len(p['theta']),length),future,calibration)
    b=inputs(raw);b['acceleration']=torch.as_tensor(factor*radius(cfg)*p['future_acceleration'],dtype=torch.float32)
    return b


class AdaptedAngular(nn.Module):
    def __init__(self,base,equilibrium=False):super().__init__();self.base=base;self.equilibrium=equilibrium
    def forward(self,b,acceleration):
        y,diagnostics=self.base(b,acceleration)
        if self.equilibrium:
            zero={key:(value if key=='omega' else torch.zeros_like(value)) for key,value in b.items()}
            z,_=self.base(zero,torch.zeros_like(acceleration));y=y-z
        return y,diagnostics


def make_angular(seed,equilibrium=False):
    bundle=ResearchBundle(fold='primary',profile='retained');base=bundle.models[seed]
    for parameter in base.parameters():parameter.requires_grad_(True)
    return AdaptedAngular(base,equilibrium)


def train_angular(train,validation,cfg,seed,equilibrium):
    torch.manual_seed(seed);m=make_angular(seed,equilibrium);c=cfg['neural'];opt=torch.optim.AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    b=angular_batch(train,cfg);vb=angular_batch(validation,cfg);target=torch.as_tensor(train['target'],dtype=torch.float32);index=cfg['horizon_index'];rng=np.random.default_rng(seed)
    best=score(predict(m,vb),validation,cfg);state=copy.deepcopy(m.state_dict());epoch_selected=-1;stale=0;log=[dict(epoch=-1,validation_mae_deg=best)]
    groups=np.unique(train['scenario']);weights=np.zeros(len(target))
    for sid in groups:
        mask=train['scenario']==sid;weights[mask]=len(weights)/(len(groups)*mask.sum())
    weights=torch.as_tensor(weights,dtype=torch.float32)
    for epoch in range(c['epochs']):
        m.train();losses=[]
        for ids in np.array_split(rng.permutation(len(target)),int(np.ceil(len(target)/c['batch_size']))):
            local={key:value[ids] for key,value in b.items()};y=m(local,local['acceleration'])[0]
            loss=((((y[:,index]-target[ids][:,index])/.1)**2).mean(1)*weights[ids]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite matched fine-tuning')
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),c['gradient_clip']);opt.step();losses.append(loss.item())
        value=score(predict(m,vb),validation,cfg);log.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_mae_deg=value))
        if value<best-1e-7:best=value;state=copy.deepcopy(m.state_dict());epoch_selected=epoch;stale=0
        else:stale+=1
        if epoch%10==0:print('ANGULAR',seed,equilibrium,epoch,'val',round(value,7),flush=True)
        if stale>=c['patience']:break
    m.load_state_dict(state);m.eval();return m,log,dict(seed=seed,equilibrium=equilibrium,selected_epoch=epoch_selected,validation_mae_deg=best)
