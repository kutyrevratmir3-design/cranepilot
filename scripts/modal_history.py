"""Restricted-record causal inputs, modal projection, and equal-history references."""
import json
from pathlib import Path
import numpy as np
from cranepilot.data import Scenario,QUANTIZATION_RAD

def load_allowed(cfg,ids,role):
    allowed=set(v for f in cfg['families'].values() for v in f[role])
    if not ids or not set(ids)<=allowed:raise ValueError('Record outside explicit role')
    if role=='confirmation':raise ValueError('Confirmation loader disabled in development')
    root=Path(cfg['data']);manifest=json.loads((root/'manifest.json').read_text());meta={m['sid']:m for m in manifest['scenarios']}
    # NPZ is lazy: never enumerate/read the unrequested signal arrays.
    with np.load(root/'dataset.npz',allow_pickle=False) as z:
        time=z['time'].copy()
        return {sid:Scenario(sid,time.copy(),{name:z[f's{sid}_{name}'].copy() for name in meta[sid]['signals']},meta[sid]) for sid in ids}

def packets(s,axis,cfg,calibration,include_targets=True):
    theta=s.signals[axis];spacing=cfg['history_spacing_raw'];history=cfg['history_samples'];h=np.array(cfg['horizons_raw']);lo=max((history-1)*spacing,int(round(cfg['origin_start_s']/.01)))
    k=np.arange(lo,len(theta)-max(h),cfg['origin_stride_raw']);allk=k.copy();raw=k[:,None]-np.arange((history-1)*spacing+1)
    length=np.ptp(s.signals['L'][raw],axis=1)<=cfg['past_length_range_m'];drive=np.ones(len(k),dtype=bool);reference=drive.copy()
    for key in ['x','J']:
        if key in s.signals:drive &= np.ptp(s.signals[key][raw],axis=1)<=cfg['past_position_range']
    for key in ['xr','Jr']:
        if key in s.signals:reference &= np.ptp(s.signals[key][k[:,None]+np.arange(max(h)+1)],axis=1)<=cfg['future_reference_range']
    keep=length&drive&reference;k=k[keep];past=k[:,None]-np.arange(history)*spacing
    p=dict(theta=theta[past],length=s.signals['L'][past],k=k,scenario=np.full(len(k),s.sid,dtype=int),omega=calibration.omega(s.signals['L'][k]))
    for key in ['x','xr','J','Jr']:
        if key in s.signals:p[key]=s.signals[key][past]
    if include_targets:p['target']=theta[k[:,None]+h]
    coverage=dict(scenario=s.sid,axis=axis,origins_considered=len(allk),eligible=len(k),past_length_gate=int(length.sum()),past_drive_gate=int(drive.sum()),planned_reference_gate=int(reference.sum()),eligible_fraction=float(keep.mean()) if len(keep) else 0.)
    return p,coverage

def concatenate(parts):
    if not parts or any(not len(p['k']) for p in parts):raise ValueError('Every development record must have eligible origins')
    return {key:np.concatenate([p[key] for p in parts]) for key in parts[0]}

def basis(time,omega,damping):
    phase=np.asarray(omega)[:,None]*np.asarray(time);e=np.exp(-np.broadcast_to(damping,(len(phase),))[:,None]*np.asarray(time)/2)
    return np.stack([np.ones_like(phase),e*np.cos(phase),e*np.sin(phase)],axis=2)

def project(y,B):
    gram=np.einsum('nti,ntj->nij',B,B);rhs=np.einsum('nti,nt->ni',B,y)
    coef=np.linalg.solve(gram+np.eye(B.shape[-1])[None]*1e-12,rhs[...,None])[...,0]
    fitted=np.einsum('nti,ni->nt',B,coef);sse=((y-fitted)**2).sum(1)
    return coef,sse,fitted

def modal(p,cfg,variant):
    if variant not in cfg['modal_variants']:raise ValueError('Unknown modal variant')
    y=p['theta'];nominal=p['omega'];time=-np.arange(y.shape[1])*cfg['dt_s'];future=np.array(cfg['horizons_raw'])*.01;n=len(y)
    if not n:raise ValueError('Empty packet set')
    if variant=='fixed':
        w=nominal.copy();d=np.zeros(n);coef,sse,fitted=project(y,basis(time,w,d))
    else:
        grid=cfg['frequency_ratios'];ratios=np.linspace(grid['min'],grid['max'],grid['count']);sse=np.full(n,np.inf);w=nominal.copy();d=np.zeros(n);coef=np.zeros((n,3));fitted=np.zeros_like(y)
        for damping in cfg['damping_grid']:
            for ratio in ratios:
                freq=nominal*ratio;c,e,f=project(y,basis(time,freq,damping));take=e<sse;sse[take]=e[take];w[take]=freq[take];d[take]=damping;coef[take]=c[take];fitted[take]=f[take]
        step=nominal*(ratios[1]-ratios[0]);_,left,_=project(y,basis(time,w-step,d));_,right,_=project(y,basis(time,w+step,d))
        curvature=left-2*sse+right;delta=np.divide(.5*(left-right),curvature,out=np.zeros(n),where=curvature>1e-20).clip(-1,1)
        refined=np.clip(w+step*delta,nominal*ratios[0],nominal*ratios[-1]);c,e,f=project(y,basis(time,refined,d));take=e<sse;sse[take]=e[take];w[take]=refined[take];coef[take]=c[take];fitted[take]=f[take]
    forecast=np.einsum('nti,ni->nt',basis(future,w,d),coef);modes=np.ones(n,dtype=int);second=np.zeros(n)
    if variant=='two_mode':
        grid=cfg['secondary_ratios'];ratios=np.linspace(grid['min'],grid['max'],grid['count']);B=basis(time,w,d);best=sse.copy();bestcoef=np.zeros((n,5));bestw=np.zeros(n)
        for ratio in ratios:
            freq=nominal*ratio;valid=np.abs(freq-w)>=cfg['minimum_mode_separation_ratio']*nominal
            C=np.concatenate([B,basis(time,freq,0)[:,:,1:]],axis=2);c,e,_=project(y,C);take=valid&(e<best);best[take]=e[take];bestcoef[take]=c[take];bestw[take]=freq[take]
        variance=np.maximum(QUANTIZATION_RAD**2/12,sse/(y.shape[1]-3));penalty=variance*(3*np.log(y.shape[1])+2*np.log(len(ratios)));take=(sse-best)>penalty
        C=np.concatenate([basis(future,w,d),basis(future,bestw,0)[:,:,1:]],axis=2);two=np.einsum('nti,ni->nt',C,bestcoef);forecast[take]=two[take];sse[take]=best[take];second[take]=bestw[take];modes[take]=2
    low=np.hypot(coef[:,1],coef[:,2])<cfg['minimum_amplitude_bins']*QUANTIZATION_RAD
    forecast[low]=y[low].mean(1)[:,None];modes[low]=0
    if not np.isfinite(forecast).all():raise ValueError('Nonfinite modal forecast')
    return forecast,dict(omega=w,damping=d,modes=modes,second_omega=second,history_sse=sse,low_amplitude=low)

def features(p,form):
    keys=['theta','length']+[k for k in ['x','xr','J','Jr'] if k in p];X=np.column_stack([p[k] for k in keys]+[p['omega']])
    if form.startswith('lpv_'):X=np.column_stack([X,p['theta']*p['omega'][:,None],p['theta']*p['omega'][:,None]**2])
    return X

def arx_statistics(p,cfg,form):
    X=features(p,form);mean=X.mean(0);scale=X.std(0).clip(1e-8);X=np.column_stack([np.ones(len(X)),(X-mean)/scale]);y=p['target'].copy()
    if form.endswith('harmonic_residual'):y-=modal(p,cfg,'fixed')[0]
    weight=np.zeros(len(X));ids=np.unique(p['scenario'])
    for sid in ids:
        m=p['scenario']==sid;weight[m]=len(X)/(len(ids)*m.sum())
    return dict(mean=mean,scale=scale,gram=(X.T*weight)@X/len(X),rhs=X.T@(weight[:,None]*y)/len(X),form=form)

def fit_arx(stats,alpha):
    P=np.eye(len(stats['rhs']))*alpha;P[0,0]=0
    return dict(mean=stats['mean'],scale=stats['scale'],coef=np.linalg.solve(stats['gram']+P,stats['rhs']),form=stats['form'],alpha=alpha)

def predict_arx(m,p,cfg):
    X=np.column_stack([np.ones(len(p['k'])),(features(p,m['form'])-m['mean'])/m['scale']]);y=X@m['coef']
    if m['form'].endswith('harmonic_residual'):y+=modal(p,cfg,'fixed')[0]
    return y

def save_arx(path,m):np.savez_compressed(path,**m)
def load_arx(path):
    with np.load(path,allow_pickle=False) as z:return dict(mean=z['mean'],scale=z['scale'],coef=z['coef'],form=str(z['form']),alpha=float(z['alpha']))

def local_ar(p,cfg,order=2,alpha=1e-4,tls=False):
    raw=p['theta'];mean=raw.mean(1);scale=raw.std(1).clip(QUANTIZATION_RAD);y=(raw-mean[:,None])/scale[:,None];n,t=y.shape
    X=np.stack([y[:,i+1:t-order+i+1] for i in range(order)],axis=2);target=y[:,:t-order]
    if tls:
        if order!=2:raise ValueError('TLS reference is AR2')
        Z=np.concatenate([X,target[:,:,None]],axis=2);_,_,vh=np.linalg.svd(Z,full_matrices=False);v=vh[:,-1];coef=np.divide(-v[:,:order],v[:,order,None],out=np.zeros((n,order)),where=np.abs(v[:,order,None])>1e-10)
    else:
        gram=np.einsum('nti,ntj->nij',X,X)/X.shape[1];rhs=np.einsum('nti,nt->ni',X,target)/X.shape[1];coef=np.linalg.solve(gram+alpha*np.eye(order)[None],rhs[...,None])[...,0]
    for i in range(n):
        roots=np.roots(np.r_[1.,-coef[i]]);roots/=np.maximum(1.,np.abs(roots));coef[i]=-np.poly(roots)[1:].real
    state=y[:,:order].copy();steps=max(cfg['horizons_raw'])//cfg['history_spacing_raw'];out=[]
    for _ in range(steps):
        value=(coef*state).sum(1);out.append(value*scale+mean);state=np.column_stack([value,state[:,:-1]])
    forecast=np.column_stack(out)[:,np.array(cfg['horizons_raw'])//cfg['history_spacing_raw']-1]
    if not np.isfinite(forecast).all():raise ValueError('Nonfinite local AR forecast')
    return forecast

def score(y,p):
    errors=np.abs(np.rad2deg(y-p['target'])).mean(1)
    return float(np.mean([errors[p['scenario']==sid].mean() for sid in np.unique(p['scenario'])]))
