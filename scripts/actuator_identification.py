"""Causal finite-window first-order lag identification; no consistency certificate.

Initial x/v/a are nuisance variables. No measured derivatives or true plant
parameters are used. Deadzone mismatch is deliberately not fitted here.
"""
import numpy as np


def forced_positions(commands,tau,dt=.05):
    commands=np.asarray(commands,dtype=float);tau=np.broadcast_to(np.asarray(tau,dtype=float),(len(commands),))
    decay=np.exp(-dt/tau);x=np.zeros(len(commands));v=x.copy();a=x.copy();xs=[x.copy()]
    for c in commands.T:
        dv=c*dt+(a-c)*tau*(1-decay)
        x=x+v*dt+.5*c*dt**2+(a-c)*tau*(dt-tau*(1-decay));v=v+dv;a=c+(a-c)*decay;xs.append(x.copy())
    return np.stack(xs,1),v,a


def fit_with_parameters(position,commands,tau,gain):
    """Profile nuisance initial state and estimate endpoint v/a at supplied parameters."""
    y=np.asarray(position,dtype=float).copy();commands=np.asarray(commands,dtype=float).copy()
    n,m=y.shape;t=np.arange(m)*.05;tau=np.broadcast_to(np.asarray(tau,dtype=float),(n,));gain=np.broadcast_to(np.asarray(gain,dtype=float),(n,))
    xf,vf,af=forced_positions(commands,tau);decay=np.exp(-t[None]/tau[:,None])
    basis=np.stack([np.ones_like(decay),np.broadcast_to(t,(n,m)),tau[:,None]*(t[None]-tau[:,None]*(1-decay))],2)
    y-=y[:,:1];residual=y-gain[:,None]*xf
    coef=(np.linalg.pinv(basis,rcond=1e-10)@residual[:,:,None])[:,:,0]
    fitted=np.einsum('nmi,ni->nm',basis,coef)+gain[:,None]*xf
    T=t[-1];end_decay=np.exp(-T/tau)
    v=coef[:,1]+coef[:,2]*tau*(1-end_decay)+gain*vf;a=coef[:,2]*end_decay+gain*af
    return v,a,np.sqrt(np.mean((fitted-y)**2,axis=1))


def identify(position,commands,config):
    position=np.asarray(position,dtype=float).copy();commands=np.asarray(commands,dtype=float).copy()
    if position.ndim!=2 or commands.shape!=(len(position),position.shape[1]-1):raise ValueError('Aligned past positions and applied interval commands required')
    if not np.isfinite(position).all() or not np.isfinite(commands).all():raise ValueError('Finite past inputs required')
    n,m=position.shape;y=position-position[:,:1];t=np.arange(m)*.05
    best=np.full(n,np.inf);tau_hat=np.full(n,.1);gain_hat=np.ones(n);info=np.zeros(n);rmse=np.zeros(n)
    if m<config['minimum_samples']:return dict(tau=tau_hat,gain=gain_hat,information=info,rmse=rmse,valid=np.zeros(n,bool))
    for tau in config['tau_grid']:
        basis=np.column_stack([np.ones(m),t,tau*(t-tau*(1-np.exp(-t/tau)))]);pinv=np.linalg.pinv(basis,rcond=1e-10)
        forced,_,_=forced_positions(commands,tau)
        f=forced-(forced@pinv.T)@basis.T;residual=y-(y@pinv.T)@basis.T
        energy=np.sum(f*f,1);gain=(np.sum(f*residual,1)+config['gain_ridge'])/(energy+config['gain_ridge'])
        gain=np.clip(gain,*config['gain_bounds']);errors=residual-gain[:,None]*f
        score=np.sum(errors**2,1)+config['gain_ridge']*(gain-1)**2+config['tau_penalty']*np.log(tau/.1)**2
        information=np.sqrt(energy/m);valid=information>=config['minimum_information_m'];better=valid&(score<best)
        best=np.where(better,score,best);tau_hat=np.where(better,tau,tau_hat);gain_hat=np.where(better,gain,gain_hat)
        info=np.where(better,information,info);rmse=np.where(better,np.sqrt(np.mean(errors**2,1)),rmse)
    valid=np.isfinite(best)&(rmse<=config['maximum_fit_rmse_m'])
    return dict(tau=tau_hat,gain=gain_hat,information=info,rmse=rmse,valid=valid)
