"""Known-age alignment: only missing angle head is predicted; never used as truth."""
import numpy as np
from cranepilot.data import QUANTIZATION_RAD
from cranepilot.physics import LengthCalibration
from scripts.modal_history import modal
from scripts.regime_fusion import context,expert_inputs,fuse,simplex_fit
from scripts import drive_conditioned as dc

def validate(history,age,axis):
    if age is None or isinstance(age,bool) or not np.isscalar(age) or not np.isfinite(age) or int(age)!=age or not 0<=age<=15:raise ValueError('Known integer angle age0..15samples required')
    if set(history)-{'theta','beta','L','x','xr','J','Jr'}:raise ValueError('Only timestamped past channels allowed')
    arrays={k:np.asarray(v,dtype=float) for k,v in history.items()}
    if axis not in arrays or 'L' not in arrays:raise ValueError('Angle and length required')
    shape=arrays[axis].shape
    if len(shape)!=2 or not shape[0] or shape[1]!=401 or any(x.shape!=shape or not np.isfinite(x).all() for x in arrays.values()) or (arrays['L']<=0).any():raise ValueError('Finite aligned401sample histories and positive length required')
    return arrays,int(age)

def mode_fit(theta,omega,task,nominal_when_low=False):
    _,d=modal(dict(theta=theta,omega=omega),task,'two_mode');w=np.where(d['modes']==0,omega,d['omega']) if nominal_when_low else d['omega'];damp=np.where(d['modes']==0,0.,d['damping']) if nominal_when_low else d['damping'];two=d['modes']==2
    def basis(times):
        t=np.asarray(times);t=np.broadcast_to(t,(len(theta),t.shape[-1]));phase=w[:,None]*t;e=np.exp(-damp[:,None]*t/2);second=d['second_omega'][:,None]*t;return np.stack([np.ones_like(t),e*np.cos(phase),e*np.sin(phase),np.cos(second)*two[:,None],np.sin(second)*two[:,None]],axis=2)
    H=basis(-np.arange(theta.shape[1])*.05);G=np.einsum('nti,ntj->nij',H,H)+1e-12*np.eye(5)[None];coef=np.linalg.solve(G,np.einsum('nti,nt->ni',H,theta)[...,None])[...,0]
    return basis,H,G,coef,w,damp,d['modes']==0

def dated_modal(theta,omega,ages,task):
    basis,_,_,coef,_,_,low=mode_fit(theta,omega,task);times=np.array(task['horizons_raw'])[None]*.01+np.asarray(ages)[:,None]*.01;y=np.einsum('nti,ni->nt',basis(times),coef);y[low]=theta[low].mean(1)[:,None];return y

def force_basis(history,gap,w,damp,length,kinds):
    initial=history[:,:1].copy();history=history-initial;gap=gap-initial;n,t,u=history.shape;state=np.zeros((n,u,2));F=np.zeros_like(history)
    def out(value,lam):return np.where(kinds[None]==0,lam[:,None]*state[:,:,0]+damp[:,None]*state[:,:,1]-value,state[:,:,0])/length[:,None]
    A,b,c,lam=dc.transition(w,damp,.05)
    for j in range(1,t):
        state=np.einsum('nij,nuj->nui',A,state)+b[:,None]*history[:,j-1,:,None]+c[:,None]*(history[:,j]-history[:,j-1])[:,:,None];F[:,j]=out(history[:,j],lam)
    A,b,c,lam=dc.transition(w,damp,.01);future=[]
    for j in range(1,gap.shape[1]):
        state=np.einsum('nij,nuj->nui',A,state)+b[:,None]*gap[:,j-1,:,None]+c[:,None]*(gap[:,j]-gap[:,j-1])[:,:,None];future.append(out(gap[:,j],lam))
    return F,np.stack(future,axis=1)

def complete_head(raw,age,axis,method,pilot,name):
    raw,age=validate(raw,age,axis);aligned={k:v.copy() for k,v in raw.items()}
    if age==0 or method=='unaware':return aligned
    aligned[axis][:,age:]=raw[axis][:,:401-age];theta=raw[axis];future=np.arange(age,0,-1)*.01;task=pilot.pilot.cfg;cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);omega=cal.omega(raw['L'][:,age])
    if method=='hold':head=np.repeat(theta[:,:1],age,axis=1)
    elif method.startswith(('linear_','quadratic_')):
        degree=2 if method.startswith('quadratic') else 1;window=float(method.split('_')[1]);count=int(round(window/.01))+1;t=-np.arange(count)*.01;B=np.column_stack([t**j for j in range(degree+1)]);coef=np.linalg.lstsq(B,theta[:,:count].T,rcond=None)[0].T;head=coef@np.stack([future**j for j in range(degree+1)])
    elif method=='modal_3s':
        sample=theta[:,:301:5];basis,_,_,coef,_,_,low=mode_fit(sample,omega,task);head=np.einsum('nti,ni->nt',basis(future),coef);head[low]=sample[low].mean(1)[:,None]
    elif method.startswith('forced_3s_'):
        sample=theta[:,:301:5];basis,H,G,coef,w,damp,_=mode_fit(sample,omega,task,True);ids=age+np.arange(61)*5;positions={k:raw[k][:,ids][:,::-1] for k in ['x','J'] if k in raw};uh,names,kinds=dc.drive_features(positions,.05);gap=np.repeat(uh[:,-1:],age+1,axis=1)
        for j,key in enumerate(positions):gap[:,:,j]=raw[key][:,age-np.arange(age+1)]
        model=pilot.models[name]['prior'];assert names==list(model['channels']);F,Fh=force_basis(uh,gap,w,damp,9.81/omega**2,kinds);F=F[:,::-1];Fh=Fh[:,::-1];projection=np.linalg.solve(G,np.einsum('nti,ntu->niu',H,F));residual=sample-np.einsum('nti,ni->nt',H,coef);Q=(F-np.einsum('nti,niu->ntu',H,projection))/model['scale'];gamma=np.broadcast_to(model['gamma'],(len(theta),len(model['gamma'])))
        if method=='forced_3s_adapt':gamma=np.linalg.solve(np.einsum('ntu,ntv->nuv',Q,Q)/61+np.eye(Q.shape[2])[None],(np.einsum('ntu,nt->nu',Q,residual)/61+model['gamma'])[...,None])[...,0]
        Y=basis(future);head=np.einsum('nti,ni->nt',Y,coef)+np.einsum('ntu,nu->nt',(Fh-np.einsum('nti,niu->ntu',Y,projection))/model['scale'],gamma)
    else:raise ValueError('Unknown head completion')
    if not np.isfinite(head).all():raise ValueError('Nonfinite head completion')
    aligned[axis][:,:age]=head;return aligned

def packet(history,axis,cal,age=0):
    return dict(theta=history[axis][:,::5],length=history['L'][:,::5],omega=cal.omega(history['L'][:,0]),k=np.arange(len(history[axis])),**{k:history[k][:,::5] for k in ['x','xr','J','Jr'] if k in history},age_raw=np.full(len(history[axis]),age),omega_sensor=cal.omega(history['L'][:,age]))

def predict(history,age,axis,method,pilot,name):
    aligned=complete_head(history,age,axis,method,pilot,name);cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);p=packet(aligned,axis,cal);del p['age_raw'];del p['omega_sensor'];ratio=context(aligned,pilot.pilot.cfg['past_position_range']);e=expert_inputs(p,ratio,pilot.pilot,name,pilot.models[name]);valid=np.array([90,80,65,50,25,0])>=age
    if method=='unaware' or age==0:y,weights=fuse(e,pilot.manifest['selected']);valid=np.ones(6,dtype=bool)
    else:
        E=e['errors'][:,valid];q=QUANTIZATION_RAD**2/12;er=np.abs(E[:,:,0]).mean(1);em=np.abs(E[:,:,2]).mean(1);wo=er**2/(er**2+em**2+q);er=np.abs(E[:,:,1]).mean(1);em=np.abs(E[:,:,3]).mean(1);wn=er**2/(er**2+em**2+q);zero=np.zeros_like(wo);po=np.column_stack([1-wo,zero,wo,zero]);pn=np.column_stack([zero,1-wn,zero,wn]);prior=np.where((ratio<=1)[:,None],po,pn);w=simplex_fit(E,prior,1.);quiet=ratio<=1;w[quiet]=po[quiet];weights=np.repeat(w[:,None],7,axis=1);y=np.einsum('nhu,nhu->nh',e['forecasts'],weights)
    return y,dict(frozen_arx=e['forecasts'][:,:,0],active_arx=e['forecasts'][:,:,1],weights=weights,valid_heads=valid,head=aligned[axis][:,:age].copy(),context=ratio)

def arx_features(p,representation):
    if set(p)-{'theta','length','omega','k','scenario','x','xr','J','Jr','age_raw','omega_sensor'}:raise ValueError('Only dated past packet fields allowed')
    q=p['theta']-p['theta'][:,:1];age=p['age_raw'][:,None]*.01;phase=p['omega'][:,None]*age;s=np.sin(phase);c=np.cos(phase)-1
    return np.column_stack([dc.io_features(p,representation),age,phase,s,c,q*s,q*c,q*age,p['omega_sensor']])

def fit_stats(X,target,base,weights,representation,form):
    mean=np.average(X,axis=0,weights=weights);scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=weights)).clip(1e-8);Z=np.column_stack([np.ones(len(X)),(X-mean)/scale]);return dict(mean=mean,scale=scale,gram=(Z.T*weights)@Z/len(Z),rhs=Z.T@(weights[:,None]*(target-base))/len(Z),representation=representation,form=form)

def arx_prediction(model,p,modal_age):
    X=np.column_stack([np.ones(len(p['theta'])),(arx_features(p,model['representation'])-model['mean'])/model['scale']]);base=np.repeat(p['theta'][:,:1],7,axis=1) if model['form']=='direct' else modal_age;return base+X@model['coef']
