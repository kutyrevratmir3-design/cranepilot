"""Simpler ARX representations to challenge nuisance-level transfer failures."""
import numpy as np
from scripts.modal_history import modal

def features(p,representation):
    theta=p['theta'];w=p['omega'][:,None]
    if representation=='theta_only':return theta
    if representation=='angle_lpv':return np.column_stack([theta,w,theta*w,theta*w*w])
    if representation not in ['difference','difference_lpv']:raise ValueError('Unknown representation')
    q=theta-theta[:,:1];X=np.column_stack([q,p['length'][:,0],w])
    if representation.endswith('_lpv'):X=np.column_stack([X,q*w,q*w*w])
    return X

def base(p,cfg,representation,form):
    if form=='harmonic_residual':return modal(p,cfg,'fixed')[0]
    if form!='direct':raise ValueError('Unknown form')
    if representation.startswith('difference'):return np.repeat(p['theta'][:,:1],len(cfg['horizons_raw']),axis=1)
    return np.zeros((len(p['k']),len(cfg['horizons_raw'])))

def statistics(p,cfg,representation,form):
    X=features(p,representation);mean=X.mean(0);scale=X.std(0).clip(1e-8);X=np.column_stack([np.ones(len(X)),(X-mean)/scale]);y=p['target']-base(p,cfg,representation,form);weight=np.zeros(len(X));ids=np.unique(p['scenario'])
    for sid in ids:
        m=p['scenario']==sid;weight[m]=len(X)/(len(ids)*m.sum())
    return dict(mean=mean,scale=scale,gram=(X.T*weight)@X/len(X),rhs=X.T@(weight[:,None]*y)/len(X),representation=representation,form=form)

def fit(stats,alpha):
    penalty=np.eye(len(stats['rhs']))*alpha;penalty[0,0]=0
    return dict(mean=stats['mean'],scale=stats['scale'],coef=np.linalg.solve(stats['gram']+penalty,stats['rhs']),representation=stats['representation'],form=stats['form'],alpha=alpha)

def predict(m,p,cfg):
    X=features(p,m['representation']);X=np.column_stack([np.ones(len(X)),(X-m['mean'])/m['scale']])
    return base(p,cfg,m['representation'],m['form'])+X@m['coef']

def save(path,m):np.savez_compressed(path,**m)
def load(path):
    with np.load(path,allow_pickle=False) as z:return {key:(str(z[key]) if key in ['form','representation'] else float(z[key]) if key=='alpha' else z[key]) for key in z.files}
