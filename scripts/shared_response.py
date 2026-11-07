"""Physics-shaped shared-horizon residual predictor: common state modes and causal FIR."""
import numpy as np
from scripts.external_baselines import history_features,oscillator


def design(p,cfg,h,variant):
    if not 1<=h<=cfg['steps']:raise ValueError('Forecast step out of range')
    if variant not in cfg['variants']:raise ValueError('Unknown response variant')
    omega,damping,_=cfg['oscillator_parameters'];t=h*cfg['dt_s'];decay=np.exp(-damping*t/2)
    basis=[1.,decay*np.cos(omega*t),decay*np.sin(omega*t)]
    if variant=='sensitivity':basis += [t*decay*np.cos(omega*t),t*decay*np.sin(omega*t)]
    H=history_features(p)
    if variant=='nonlinear':
        q=p['theta']/.1;v=p['past_velocity']/.2
        H=np.column_stack([H,q*v,q*v*v,(np.sin(p['theta'])-p['theta'])/.0001])
    H=np.column_stack([np.ones(len(H)),H]);state=np.einsum('ni,j->nij',H,np.asarray(basis)).reshape(len(H),-1)
    # Coefficient for a fixed lag is shared across all target horizons.
    velocity=np.zeros((len(H),cfg['steps']));acceleration=np.zeros_like(velocity)
    velocity[:,:h]=p['future_velocity'][:,:h][:,::-1]/.2
    acceleration[:,:h]=p['future_acceleration'][:,:h][:,::-1]/.5
    return np.column_stack([state,velocity,acceleration])


def sufficient_statistics(p,cfg,variant):
    ybase=oscillator(p,cfg,cfg['oscillator_parameters']);X=np.concatenate([design(p,cfg,h+1,variant) for h in cfg['horizon_index']]);y=np.concatenate([p['target'][:,h]-ybase[:,h] for h in cfg['horizon_index']])
    labels=np.tile(p['scenario'],len(cfg['horizon_index']));weight=np.zeros(len(labels));groups=np.unique(labels)
    for sid in groups:
        m=labels==sid;weight[m]=len(labels)/(len(groups)*m.sum())
    mean=X.mean(0);scale=X.std(0).clip(1e-6);X=np.column_stack([np.ones(len(X)),(X-mean)/scale])
    return dict(mean=mean,scale=scale,gram=(X.T*weight)@X/len(X),rhs=X.T@(weight*y)/len(X),variant=variant)


def fit(stats,alpha):
    penalty=np.eye(len(stats['rhs']))*alpha;penalty[0,0]=0
    coef=np.linalg.solve(stats['gram']+penalty,stats['rhs'])
    return dict(mean=stats['mean'],scale=stats['scale'],coef=coef,variant=stats['variant'],alpha=alpha)


def predict(m,p,cfg):
    y=oscillator(p,cfg,cfg['oscillator_parameters'])
    for j in range(cfg['steps']):
        X=design(p,cfg,j+1,m['variant']);y[:,j]+=np.column_stack([np.ones(len(X)),(X-m['mean'])/m['scale']])@m['coef']
    return y


def save(path,m):np.savez_compressed(path,**{k:np.array(v) for k,v in m.items()})


def load(path):
    with np.load(path) as f:return dict(mean=f['mean'],scale=f['scale'],coef=f['coef'],variant=str(f['variant']),alpha=float(f['alpha']))
