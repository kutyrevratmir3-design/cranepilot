"""Dated-angle model bank: measured historical gap, assumed future, no head truth."""
import numpy as np
from cranepilot.physics import LengthCalibration
from cranepilot.data import QUANTIZATION_RAD
from scripts import latency_observer as lo
from scripts import drive_conditioned as dc
from scripts.regime_fusion import context,expert_inputs,simplex_fit
from scripts.modal_history import modal
from scripts.modal_backtest_gate import sliced
from scripts.modal_reference_challenge import predict as old_predict

def validate(history,age,axis,shadow=False):
    if isinstance(age,(bool,np.bool_)) or not isinstance(age,(int,np.integer)) or not 0<=age<=25:raise ValueError('Known integer age0..25required')
    if set(history)-{'theta','beta','L','x','xr','J','Jr'}:raise ValueError('Only past channels permitted')
    raw={k:np.asarray(v,dtype=float) for k,v in history.items()}
    if axis not in raw or 'L' not in raw:raise ValueError('Angle and length required')
    shape=raw[axis].shape;allowed=[301,401] if shadow else [401]
    if len(shape)!=2 or not shape[0] or shape[1] not in allowed or any(v.shape!=shape or not np.isfinite(v).all() for v in raw.values()) or (raw['L']<=0).any():raise ValueError('Finite aligned current-support/dated-angle histories required')
    return raw

def extended_head(raw,age,axis,method,pilot,name):
    raw=validate(raw,age,axis)
    if age<=15:return lo.complete_head(raw,age,axis,method,pilot,name)
    aligned={k:v.copy() for k,v in raw.items()}
    if method=='unaware':return aligned
    aligned[axis][:,age:]=raw[axis][:,:401-age];theta=raw[axis];future=np.arange(age,0,-1)*.01;task=pilot.pilot.cfg;cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);omega=cal.omega(raw['L'][:,age])
    if method=='hold':head=np.repeat(theta[:,:1],age,axis=1)
    elif method.startswith(('linear_','quadratic_')):
        degree=2 if method.startswith('quadratic') else 1;count=int(round(float(method.split('_')[1])/.01))+1;t=-np.arange(count)*.01;B=np.column_stack([t**j for j in range(degree+1)]);coef=np.linalg.lstsq(B,theta[:,:count].T,rcond=None)[0].T;head=coef@np.stack([future**j for j in range(degree+1)])
    elif method=='modal_3s':
        sample=theta[:,:301:5];basis,_,_,coef,_,_,low=lo.mode_fit(sample,omega,task);head=np.einsum('nti,ni->nt',basis(future),coef);head[low]=sample[low].mean(1)[:,None]
    elif method.startswith('forced_3s_'):
        sample=theta[:,:301:5];basis,H,G,coef,w,damp,_=lo.mode_fit(sample,omega,task,True);ids=age+np.arange(61)*5;positions={k:raw[k][:,ids][:,::-1] for k in ['x','J'] if k in raw};uh,names,kinds=dc.drive_features(positions,.05);gap=np.repeat(uh[:,-1:],age+1,axis=1)
        for j,key in enumerate(positions):gap[:,:,j]=raw[key][:,age-np.arange(age+1)]
        prior=pilot.models[name]['prior'];assert names==list(prior['channels']);F,Fh=lo.force_basis(uh,gap,w,damp,9.81/omega**2,kinds);F=F[:,::-1];Fh=Fh[:,::-1];projection=np.linalg.solve(G,np.einsum('nti,ntu->niu',H,F));residual=sample-np.einsum('nti,ni->nt',H,coef);Q=(F-np.einsum('nti,niu->ntu',H,projection))/prior['scale'];gamma=coupling(Q,residual,prior,method.endswith('adapt'));Y=basis(future);head=np.einsum('nti,ni->nt',Y,coef)+np.einsum('ntu,nu->nt',(Fh-np.einsum('nti,niu->ntu',Y,projection))/prior['scale'],gamma)
    else:raise ValueError('Unknown extended head method')
    aligned[axis][:,:age]=head
    if not np.isfinite(head).all():raise ValueError('Nonfinite head')
    return aligned

def coupling(Q,residual,prior,adapt):
    if not adapt:return np.broadcast_to(prior['gamma'],(len(Q),len(prior['gamma'])))
    return np.linalg.solve(np.einsum('ntu,ntv->nuv',Q,Q)/Q.shape[1]+np.eye(Q.shape[2])[None],(np.einsum('ntu,nt->nu',Q,residual)/Q.shape[1]+prior['gamma'])[...,None])[...,0]

def parent_quad(raw,age,axis,pilot,name):
    if age<=15:return lo.predict(raw,age,axis,'quadratic_0.25',pilot,name)[0]
    aligned=extended_head(raw,age,axis,'quadratic_0.25',pilot,name);cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);p=lo.packet(aligned,axis,cal);p.pop('age_raw');p.pop('omega_sensor');ratio=context(raw,pilot.pilot.cfg['past_position_range']);e=expert_inputs(p,ratio,pilot.pilot,name,pilot.models[name]);valid=np.array([90,80,65,50,25,0])>=age;E=e['errors'][:,valid];q=QUANTIZATION_RAD**2/12;er=np.abs(E[:,:,0]).mean(1);em=np.abs(E[:,:,2]).mean(1);wo=er**2/(er**2+em**2+q);er=np.abs(E[:,:,1]).mean(1);em=np.abs(E[:,:,3]).mean(1);wn=er**2/(er**2+em**2+q);z=np.zeros_like(wo);po=np.column_stack([1-wo,z,wo,z]);pn=np.column_stack([z,1-wn,z,wn]);prior=np.where((ratio<=1)[:,None],po,pn);weights=simplex_fit(E,prior,1);weights[ratio<=1]=po[ratio<=1];return np.einsum('nhu,nu->nh',e['forecasts'],weights)

def positions_at(raw,key,indices,bounds,continuation):
    """Position at integer times relative to CURRENT clock; positive times assumed."""
    times=np.asarray(indices);n=len(raw[key]);past=np.maximum(-times,0);values=raw[key][:,past].copy();future=times>0
    if future.any():
        current=raw[key][:,0];pred=np.repeat(current[:,None],future.sum(),axis=1)
        if continuation=='linear':
            t=-np.arange(5)*.05;t-=t.mean();v=raw[key][:,np.arange(5)*5]@t/(t@t);pred+=v[:,None]*times[future][None]*.01
            if bounds is not None:pred=np.clip(pred,np.minimum(bounds[0],current)[:,None],np.maximum(bounds[1],current)[:,None])
        elif continuation!='hold':raise ValueError('Unknown continuation')
        values[:,future]=pred
    return values

def future_drivers(raw,age,steps,positions,uh,bounds,continuation):
    times=np.arange(steps+1)-age;pos={};velocity={}
    for j,key in enumerate(positions):
        b=bounds[j];pos[key]=positions_at(raw,key,times,b,continuation);left=positions_at(raw,key,times-5,b,continuation);older=positions_at(raw,key,times-10,b,continuation);velocity[key]=(3*pos[key]-4*left+older)/.1
    values=[pos[k] for k in positions]
    if 'J' in pos:
        values.append(velocity['J']**2)
        if 'x' in pos:values.extend([pos['x']*velocity['J']**2,velocity['x']*velocity['J']])
    future=np.stack(values,axis=2);future[:,0]=uh[:,-1];return future

def variable_length_free(raw,age,horizons,cal,coef,w,damp,second,low,free):
    n=len(raw['L']);steps=age+int(max(horizons));ids=np.maximum(age-np.arange(steps+1),0);length=np.maximum(cal.a*raw['L'][:,ids]+cal.b,.01);Ls=length[:,0];state=np.stack([np.stack([coef[:,1],-.5*damp*coef[:,1]+w*coef[:,2]],axis=1),np.stack([coef[:,3],second*coef[:,4]],axis=1)],axis=1);lam=np.column_stack([w*w+damp*damp/4,second*second]);base_d=np.column_stack([damp,np.zeros(n)]);values=[]
    for j in range(steps):
        L=(length[:,j]+length[:,j+1])/2;rate=(length[:,j+1]-length[:,j])/.01;a=lam*(Ls/L)[:,None];d=base_d+2*(rate/L)[:,None]
        def f(z):return np.stack([z[:,:,1],-a*z[:,:,0]-d*z[:,:,1]],axis=2)
        k1=f(state);k2=f(state+.005*k1);k3=f(state+.005*k2);k4=f(state+.01*k3);state+=.01*(k1+2*k2+2*k3+k4)/6;values.append(coef[:,0]+state[:,:,0].sum(1))
    out=np.stack(values,axis=1)[:,age+np.array(horizons)-1];constant=np.ptp(length,axis=1)==0;out[constant|low]=free[constant|low];return out

def direct_bank(raw,age,axis,pilot,name,horizons):
    validate(raw,age,axis,True);cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);task=pilot.pilot.cfg;omega=cal.omega(raw['L'][:,age]);theta=raw[axis][:,:251:5];basis,H,G,coef,w,damp,low=lo.mode_fit(theta,omega,task,True);times=(np.asarray(horizons)+age)*.01;Y=basis(times);free=np.einsum('nti,ni->nt',Y,coef);free[low]=theta[low].mean(1)[:,None];ids=age+np.arange(51)*5;positions={k:raw[k][:,ids][:,::-1] for k in ['x','J'] if k in raw};uh,names,kinds=dc.drive_features(positions,.05);prior=pilot.models[name]['prior'];assert names==list(prior['channels']);out=[free]
    for continuation in ['hold','linear']:
        uf=future_drivers(raw,age,age+int(max(horizons)),list(positions),uh,prior['bounds'],continuation);F,Ff=lo.force_basis(uh,uf,w,damp,9.81/omega**2,kinds);F=F[:,::-1];Ff=Ff[:,age+np.asarray(horizons)-1];P=np.linalg.solve(G,np.einsum('nti,ntu->niu',H,F));Q=(F-np.einsum('nti,niu->ntu',H,P))/prior['scale'];residual=theta-np.einsum('nti,ni->nt',H,coef);Ffuture=(Ff-np.einsum('nti,niu->ntu',Y,P))/prior['scale']
        for adapt in [False,True]:out.append(np.einsum('nti,ni->nt',Y,coef)+np.einsum('ntu,nu->nt',Ffuture,coupling(Q,residual,prior,adapt)))
    # The secondary frequency is available from the fitted basis derivative.
    # Reuse the frozen search diagnostic explicitly, not a numerical derivative.
    from scripts.modal_history import modal as _modal
    second=_modal(dict(theta=theta,omega=omega),task,'two_mode')[1]['second_omega'];out.append(variable_length_free(raw,age,horizons,cal,coef,w,damp,second,low,free));return np.stack(out,axis=2)

def predictions(raw,age,axis,pilot,name,methods):
    raw=validate(raw,age,axis);parent=parent_quad(raw,age,axis,pilot,name);n=len(parent);h=np.array(pilot.pilot.cfg['horizons_raw']);cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);bare=lo.dated_modal(raw[axis][:,::5],cal.omega(raw['L'][:,age]),np.full(n,age),pilot.pilot.cfg)
    if age==0:return {key:parent.copy() for key in methods},dict(bare_dated_modal=bare,bank_bypassed=True)
    current=direct_bank(raw,age,axis,pilot,name,h);shadow={k:v[:,100:] for k,v in raw.items()};heads=np.array([10,20,35,50,70,100-age]);back=direct_bank(shadow,age,axis,pilot,name,heads);indices=100-age-heads;truth=raw[axis][:,indices];E=back-truth[:,:,None];ratio=context(raw,pilot.pilot.cfg['past_position_range']);quiet=ratio<=1;prior=np.tile([.25,.125,.125,.125,.125,.25],(n,1));prior[quiet]=[.5,0,0,0,0,.5];q=1/(np.abs(E).mean(1)**2+QUANTIZATION_RAD**2/12);wi=q/q.sum(1)[:,None];out=dict(parent_quad=parent,direct_free4=bare,direct_free2p5=current[:,:,0],direct_hold_fixed=current[:,:,1],direct_hold_adapt=current[:,:,2],direct_linear_fixed=current[:,:,3],direct_linear_adapt=current[:,:,4],direct_variable_length=current[:,:,5],bank_inverse=np.einsum('nhu,nu->nh',current,wi));weights={}
    for strength in [.1,1,10]:
        w=simplex_fit(E,prior,strength);key=f'bank_simplex_{strength:g}';out[key]=np.einsum('nhu,nu->nh',current,w);weights[key]=w
    for tau in [5,10,20]:
        a=age/(age+tau);out[f'mix_free_{tau}']=(1-a)*parent+a*bare;out[f'mix_bank_{tau}']=(1-a)*parent+a*out['bank_simplex_1']
    for tau in [10,20]:
        a=(age/(age+tau))*h/(h+50);out[f'mix_horizon_{tau}']=(1-a[None])*parent+a[None]*out['bank_simplex_1']
    a=(age/(age+10))*quiet*(np.ptp(raw['L'],axis=1)<=pilot.pilot.cfg['past_length_range_m']);out['mix_quiet_10']=(1-a[:,None])*parent+a[:,None]*bare
    if set(out)!=set(methods) or not all(np.isfinite(y).all() for y in out.values()):raise ValueError('Unexpected/nonfinite direct forecast')
    return out,dict(bare_dated_modal=bare,bank_bypassed=False,bank_forecasts=current,past_errors=E,shadow_heads=heads,truth_indices=indices,**{'weight_'+k:v for k,v in weights.items()})

def arx(raw,age,axis,pilot,name,method,expert):
    aligned=extended_head(raw,age,axis,method,pilot,name);cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);p=lo.packet(aligned,axis,cal);p.pop('age_raw');p.pop('omega_sensor');task=pilot.pilot.cfg
    if expert=='frozen_arx' or pilot.models[name]['reference']=='old':
        c=pilot.pilot.reliability_manifest['cases'][name]
        return old_predict(pilot.pilot.short_models[name],sliced(p,0,61,cal),task) if c['reference']=='short' else old_predict(pilot.pilot.cases[name]['reference'],p,task)
    short=pilot.models[name]['reference']=='short';m=pilot.models[name]['short'] if short else pilot.models[name]['long'];part=sliced(p,0,61,cal) if short else p;free=modal(part,task,'two_mode')[0] if m['form']=='modal_residual' else np.zeros((len(p['theta']),7));return dc.arx_predict(m,part,free)
