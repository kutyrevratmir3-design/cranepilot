"""Four frozen forecast experts, causal context, and simplex-constrained past fit."""
import itertools
import numpy as np
from cranepilot.data import QUANTIZATION_RAD
from cranepilot.physics import LengthCalibration
from scripts.modal_history import modal
from scripts.modal_reference_challenge import predict as old_predict
from scripts.modal_backtest_gate import sliced,compose
from scripts.active_qualification import frozen_predictions
from scripts import drive_conditioned as dc

def context(history,cutoff):
    positions=[np.asarray(history[k],dtype=float) for k in ['x','J'] if k in history]
    if not positions or cutoff<=0:raise ValueError('Past support histories and positive cutoff required')
    if any(a.ndim!=2 or a.shape[1]!=401 or not np.isfinite(a).all() or a.shape!=positions[0].shape for a in positions):raise ValueError('Context needs aligned finite raw401sample histories')
    return np.maximum.reduce([np.ptp(a,axis=1)/cutoff for a in positions])

def expert_inputs(p,ratio,pilot,name,models):
    if set(p)-{'theta','length','omega','k','scenario','x','xr','J','Jr'}:raise ValueError('Only past packet fields allowed')
    task=pilot.cfg;cal=LengthCalibration(**pilot.cases[name]['calibration']);old,diagnostic=frozen_predictions(p,pilot,name);free=old['frozen_modal'];prefix=sliced(p,20,None,cal);past_free=modal(prefix,task,'two_mode')[0];past_old=old_predict(pilot.short_models[name],prefix,task);choice=models['reference']
    if choice=='old':current_new=old['strong_arx'];past_new=past_old
    else:
        past_new=dc.arx_predict(models['short'],prefix,past_free)
        if choice=='long':current_new=dc.arx_predict(models['long'],p,free)
        elif choice=='short':
            recent=sliced(p,0,61,cal);current_new=dc.arx_predict(models['short'],recent,modal(recent,task,'two_mode')[0])
        else:raise ValueError('Unknown reference')
    prior=models['prior'];physical=dc.prepare(p,task,'linear',prior['bounds']);past_physical=dc.prepare(prefix,task,'linear',prior['bounds']);forced,_=dc.predict(physical,prior,'adapt');past_forced,_=dc.predict(past_physical,prior,'adapt');observed=p['theta'][:,[18,16,13,10,5,0]];em=np.abs(past_forced[:,:6]-observed).mean(1);er=np.abs(past_new[:,:6]-observed).mean(1);active,w=compose(forced,current_new,dict(modal_backtest_mae=em,arx_backtest_mae=er),'soft');wo=diagnostic['modal_weight'];zero=np.zeros_like(w);po=np.column_stack([1-wo,zero,wo,zero]);pn=np.column_stack([zero,1-w,zero,w]);forecasts=np.stack([old['strong_arx'],current_new,free,forced],axis=2);backtests=np.stack([past_old[:,:6],past_new[:,:6],past_free[:,:6],past_forced[:,:6]],axis=2)
    return dict(forecasts=forecasts,errors=backtests-observed[:,:,None],old_prior=po,new_prior=pn,legacy=old['frozen_blend'],active=active,context=np.asarray(ratio,dtype=float))

def simplex_fit(errors,prior,strength):
    n,t,u=errors.shape
    if strength<=0 or prior.shape!=(n,u) or not np.isfinite(errors).all() or not np.isfinite(prior).all():raise ValueError('Finite aligned errors/prior and positive ridge required')
    if (prior<0).any() or not np.allclose(prior.sum(1),1):raise ValueError('Simplex prior required')
    lam=strength*(np.mean(errors**2,axis=(1,2))+QUANTIZATION_RAD**2/12);A=np.einsum('nti,ntj->nij',errors,errors)/t+lam[:,None,None]*np.eye(u)[None];rhs=lam[:,None]*prior;best=np.full(n,np.inf);weights=prior.copy()
    for count in range(1,u+1):
        for subset in itertools.combinations(range(u),count):
            ids=np.array(subset);G=A[:,ids][:,:,ids];b=rhs[:,ids];sol=np.linalg.solve(G,np.stack([b,np.ones_like(b)],axis=2));v=sol[:,:,0];q=sol[:,:,1];w=v+(1-v.sum(1))[:,None]*q/q.sum(1)[:,None];feasible=(w>=-1e-10).all(1);w=np.maximum(w,0);w/=w.sum(1)[:,None];full=np.zeros((n,u));full[:,ids]=w;loss=np.mean(np.einsum('ntu,nu->nt',errors,full)**2,axis=1)+lam*np.sum((full-prior)**2,axis=1);take=feasible&(loss<best);best[take]=loss[take];weights[take]=full[take]
    if not np.isfinite(best).all():raise ValueError('No finite simplex solution')
    return weights

def fuse(e,variant):
    if set(e)!={'forecasts','errors','old_prior','new_prior','legacy','active','context'}:raise ValueError('Only declared past-derived evidence is accepted')
    F=e['forecasts'];E=e['errors'];n,h,u=F.shape
    if u!=4 or E.shape!=(n,6,u) or e['context'].shape!=(n,) or not all(np.isfinite(x).all() for x in e.values()):raise ValueError('Finite aligned four-expert evidence required')
    quiet=e['context']<=1;po=e['old_prior'];pn=e['new_prior'];hard=np.where(quiet[:,None],po,pn);guarded=True
    if variant=='legacy':w=po
    elif variant=='active':w=pn;guarded=False
    elif variant=='hard_context':w=hard
    elif variant=='ramp_context':
        a=np.clip(e['context']-1,0,1)[:,None];w=(1-a)*po+a*pn
    elif variant=='guarded_inverse':
        q=1/(np.abs(E).mean(1)**2+QUANTIZATION_RAD**2/12);w=q/q.sum(1)[:,None]
    elif variant=='guarded_horizon_inverse':
        q=1/(E**2+QUANTIZATION_RAD**2/12);w=q/q.sum(2)[:,:,None];w=np.concatenate([w,w[:,-1:]],axis=1)
    elif variant=='guarded_winner':w=np.eye(u)[np.argmin(np.abs(E).mean(1),axis=1)]
    elif variant.startswith('guarded_simplex_'):w=simplex_fit(E,hard,float(variant.rsplit('_',1)[1]))
    else:raise ValueError('Unknown routing method')
    if w.ndim==2:w=np.repeat(w[:,None,:],h,axis=1)
    if guarded:w[quiet]=po[quiet,None,:]
    y=np.einsum('nhu,nhu->nh',F,w)
    if variant=='active':y=e['active'].copy()
    elif variant=='legacy':y=e['legacy'].copy()
    elif guarded:y[quiet]=e['legacy'][quiet]
    if (w<0).any() or not np.allclose(w.sum(2),1,atol=1e-12) or not np.isfinite(y).all():raise ValueError('Invalid convex forecast')
    return y,w
