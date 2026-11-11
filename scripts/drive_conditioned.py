"""Local modal state fit with measured past support motion and a TRAIN coupling prior."""
import numpy as np
from scripts.modal_history import modal

def transition(omega,damping,dt):
    n=len(omega);lam=omega**2+damping**2/4;c=np.cos(omega*dt);s=np.sin(omega*dt)/omega;e=np.exp(-damping*dt/2);A=np.zeros((n,2,2));A[:,0,1]=1;A[:,1,0]=-lam;A[:,1,1]=-damping
    Ad=e[:,None,None]*np.stack([np.stack([c+damping*s/2,s],axis=1),np.stack([-lam*s,c-damping*s/2],axis=1)],axis=1);b=np.tile([0.,1.],(n,1));B0=np.linalg.solve(A,(Ad-np.eye(2)[None])[:,:,1,None])[...,0];B1=np.linalg.solve(A,(B0-dt*b)[...,None])[...,0]/dt
    return Ad,B0,B1,lam

def drive_features(positions,dt):
    values=[];names=[];kinds=[]
    for key in ['x','J']:
        if key in positions:values.append(positions[key]);names.append(key+'_support');kinds.append(0)
    if 'J' in positions:
        rate=np.gradient(positions['J'],dt,axis=1,edge_order=2);values.append(rate**2);names.append('J_rate_squared');kinds.append(1)
        if 'x' in positions:
            xr=np.gradient(positions['x'],dt,axis=1,edge_order=2);values.extend([positions['x']*rate**2,xr*rate]);names.extend(['x_J_rate_squared','x_rate_J_rate']);kinds.extend([1,1])
    return np.stack(values,axis=2),names,np.array(kinds,dtype=int)

def future_positions(p,dt,continuation,bounds=None,given=None):
    positions={k:p[k][:,::-1] for k in ['x','J'] if k in p};future={};steps=31;time=np.arange(steps)*dt
    for j,key in enumerate(positions):
        current=p[key][:,0]
        if given is not None:
            y=np.asarray(given[key],dtype=float)
            if y.shape!=(len(current),steps) or not np.isfinite(y).all() or not np.allclose(y[:,0],current,rtol=0,atol=1e-12):raise ValueError('Hypothetical support path must be finite, current-first and aligned')
            future[key]=y.copy();continue
        if continuation=='hold':future[key]=np.repeat(current[:,None],steps,axis=1)
        elif continuation=='linear':
            t=np.arange(5)*dt;t-=t.mean();velocity=(positions[key][:,-5:]@t)/(t@t);y=current[:,None]+velocity[:,None]*time
            if bounds is not None:y=np.clip(y,np.minimum(bounds[j,0],current)[:,None],np.maximum(bounds[j,1],current)[:,None])
            future[key]=y
        else:raise ValueError('Unknown support continuation')
    return positions,future

def transfer_history_future(history,future,omega,damping,length,kinds,dt):
    # A first-order hold integrates the measured displacement itself; no second derivative.
    initial=history[:,:1].copy();history=history-initial;future=future-initial
    Ad,B0,B1,lam=transition(omega,damping,dt);n,t,u=history.shape;state=np.zeros((n,u,2));out=np.zeros_like(history)
    def output(state,value):
        support=lam[:,None]*state[:,:,0]+damping[:,None]*state[:,:,1]-value
        return np.where(kinds[None,:]==0,support,state[:,:,0])/length[:,None]
    for j in range(1,t):
        state=np.einsum('nij,nuj->nui',Ad,state)+B0[:,None,:]*history[:,j-1,:,None]+B1[:,None,:]*(history[:,j]-history[:,j-1])[:,:,None];out[:,j]=output(state,history[:,j])
    values=[]
    for j in range(1,future.shape[1]):
        state=np.einsum('nij,nuj->nui',Ad,state)+B0[:,None,:]*future[:,j-1,:,None]+B1[:,None,:]*(future[:,j]-future[:,j-1])[:,:,None];values.append(output(state,future[:,j]))
    return out,np.stack(values,axis=1)

def prepare(p,task,continuation='hold',bounds=None,given=None):
    if 'target' in p:raise ValueError('Keep targets separate')
    free,diag=modal(p,task,'two_mode');n,t=p['theta'].shape;low=diag['modes']==0;w=np.where(low,p['omega'],diag['omega']);d=np.where(low,0.,diag['damping']);w2=diag['second_omega'];two=diag['modes']==2;past=-np.arange(t)[::-1]*task['dt_s'];future=np.array(task['horizons_raw'])*.01
    def B(time):
        phase=w[:,None]*time;e=np.exp(-d[:,None]*time/2);second=w2[:,None]*time
        return np.stack([np.ones((n,len(time))),e*np.cos(phase),e*np.sin(phase),np.cos(second)*two[:,None],np.sin(second)*two[:,None]],axis=2)
    H=B(past);Y=B(future);positions,planned=future_positions(p,task['dt_s'],continuation,bounds,given);uh,names,kinds=drive_features(positions,task['dt_s']);uf,other,other_kinds=drive_features(planned,task['dt_s']);assert names==other and np.array_equal(kinds,other_kinds);uf[:,0]=uh[:,-1];initial=uh[:,:1].copy();uh-=initial;uf-=initial;length=9.81/p['omega']**2;F,Ffuture=transfer_history_future(uh,uf,w,d,length,kinds,task['dt_s']);Ffuture=Ffuture[:,np.array(task['horizons_raw'])//task['history_spacing_raw']-1]
    gram=np.einsum('nti,ntj->nij',H,H)+1e-12*np.eye(5)[None];coef=np.linalg.solve(gram,np.einsum('nti,nt->ni',H,p['theta'][:,::-1])[...,None])[...,0];projection=np.linalg.solve(gram,np.einsum('nti,ntu->niu',H,F));residual=p['theta'][:,::-1]-np.einsum('nti,ni->nt',H,coef);force=F-np.einsum('nti,niu->ntu',H,projection);future_force=Ffuture-np.einsum('nti,niu->ntu',Y,projection)
    return dict(force=force,residual=residual,future_force=future_force,unforced=np.einsum('nti,ni->nt',Y,coef),free=free,channels=names,positions=list(positions))

def balanced_weights(scenario,drive):
    weights=np.zeros(len(scenario));ids=np.unique(scenario)
    for sid in ids:
        present=np.unique(drive[scenario==sid])
        for group in present:
            mask=(scenario==sid)&(drive==group);weights[mask]=len(scenario)/(len(ids)*len(present)*mask.sum())
    return weights

def fit_prior(prepared,p,drive,ridge):
    F=prepared['force'];r=prepared['residual'];weight=balanced_weights(p['scenario'],drive);den=len(F)*F.shape[1];scale=np.sqrt(np.einsum('ntu,ntu,n->u',F,F,weight)/den).clip(1e-6);Q=F/scale;gram=np.einsum('ntu,ntv,n->uv',Q,Q,weight)/den;rhs=np.einsum('ntu,nt,n->u',Q,r,weight)/den;gamma=np.linalg.solve(gram+ridge*np.eye(len(rhs)),rhs);bounds=np.array([[p[k].min(),p[k].max()] for k in prepared['positions']]);return dict(gamma=gamma,scale=scale,channels=np.array(prepared['channels']),positions=np.array(prepared['positions']),bounds=bounds)

def predict(prepared,model,strategy):
    if list(model['channels'])!=prepared['channels']:raise ValueError('Drive channels changed')
    if strategy=='fixed':gamma=np.broadcast_to(model['gamma'],(len(prepared['force']),len(model['gamma'])))
    elif strategy=='adapt':
        Q=prepared['force']/model['scale'];n,t,u=Q.shape;G=np.einsum('ntu,ntv->nuv',Q,Q)/t;rhs=np.einsum('ntu,nt->nu',Q,prepared['residual'])/t;gamma=np.linalg.solve(G+np.eye(u)[None],(rhs+model['gamma'])[...,None])[...,0]
    else:raise ValueError('Unknown coupling strategy')
    y=prepared['unforced']+np.einsum('ntu,nu->nt',prepared['future_force']/model['scale'],gamma)
    if not np.isfinite(y).all():raise ValueError('Nonfinite drive-conditioned forecast')
    return y,gamma

def io_features(p,representation):
    q=p['theta']-p['theta'][:,:1];w=p['omega'][:,None];X=np.column_stack([q,w,p['length'][:,0],q*w,q*w*w])
    if representation=='angle_lpv':return X
    if representation!='io_lpv':raise ValueError('Unknown representation')
    positions={k:p[k][:,::-1] for k in ['x','J'] if k in p};D,_,_=drive_features(positions,.05);D=D[:,::-1].copy()
    for j in range(len(positions)):D[:,:,j]-=D[:,:1,j]
    X=np.column_stack([X,p['length']-p['length'][:,:1],D.reshape(len(q),-1)])
    for key in ['xr','Jr']:
        if key in p:X=np.column_stack([X,p[key]-p[key][:,:1]])
    return X

def arx_base(p,free,form):
    if form=='direct':return np.repeat(p['theta'][:,:1],free.shape[1],axis=1)
    if form=='modal_residual':return free
    raise ValueError('Unknown form')

def arx_stats(p,target,free,drive,representation,form):
    X=io_features(p,representation);mean=X.mean(0);scale=X.std(0).clip(1e-8);X=np.column_stack([np.ones(len(X)),(X-mean)/scale]);y=target-arx_base(p,free,form);weight=balanced_weights(p['scenario'],drive);return dict(mean=mean,scale=scale,gram=(X.T*weight)@X/len(X),rhs=X.T@(weight[:,None]*y)/len(X),representation=representation,form=form)

def arx_fit(stats,alpha):
    P=np.eye(len(stats['rhs']))*alpha;P[0,0]=0
    return dict(mean=stats['mean'],scale=stats['scale'],coef=np.linalg.solve(stats['gram']+P,stats['rhs']),representation=stats['representation'],form=stats['form'],alpha=alpha)

def arx_predict(m,p,free):
    X=np.column_stack([np.ones(len(p['k'])),(io_features(p,m['representation'])-m['mean'])/m['scale']]);return arx_base(p,free,m['form'])+X@m['coef']

def save(path,m):np.savez_compressed(path,**m)
def load(path):
    with np.load(path,allow_pickle=False) as z:return {key:(str(z[key]) if key in ['representation','form'] else float(z[key]) if key=='alpha' else z[key]) for key in z.files}

def balanced_score(y,target,drive):
    return float(np.mean([np.abs(np.rad2deg(y[drive==group]-target[drive==group])).mean() for group in np.unique(drive)]))
