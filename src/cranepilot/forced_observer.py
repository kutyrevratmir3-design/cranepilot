"""Finite-history forced oscillator observer without a second x derivative.

Own discretization of a standard forced-oscillator transformation, not a
novelty claim. All arrays end at the forecast origin and are newest first.
"""
import numpy as np


def forced_observer(past_theta, past_x, omega, dt=.01, velocity=None):
    theta=np.asarray(past_theta,dtype=float)[:,::-1]
    x=np.asarray(past_x,dtype=float)[:,::-1]
    if theta.shape!=x.shape or theta.ndim!=2 or theta.shape[1]<3:
        raise ValueError('Matching two-dimensional history packets required')
    if dt<=0:raise ValueError('Positive sample interval required')
    omega=np.broadcast_to(np.asarray(omega,dtype=float),(len(theta),))
    if np.any(omega<=0):raise ValueError('Positive angular frequency required')
    x=x-x[:,:1]
    phase=omega*dt; c=np.cos(phase);s=np.sin(phase)
    omc=2*np.sin(phase/2)**2
    r=np.zeros(len(theta));v=np.zeros(len(theta));part=np.zeros_like(theta)
    for k in range(theta.shape[1]-1):
        slope=(x[:,k+1]-x[:,k])/dt
        r,v=(c*r+s*v+omc*x[:,k]+slope/omega*(phase-s),
             -s*r+c*v+s*x[:,k]+slope/omega*omc)
        part[:,k+1]=(omega**2/9.81)*(r-x[:,k+1])
    time=np.arange(theta.shape[1])*dt
    cosine=np.cos(omega[:,None]*time);sine=np.sin(omega[:,None]*time)
    basis=np.stack([cosine,sine],axis=-1)
    gram=np.einsum('nti,ntj->nij',basis,basis)
    rhs=np.einsum('nti,nt->ni',basis,theta-part)
    coef=np.linalg.solve(gram+1e-12*np.eye(2)[None],rhs[...,None])[...,0]
    fitted=np.einsum('nti,ni->nt',basis,coef)+part
    if velocity is None:
        # Regularize the endpoint derivative using a backward-only 0.2 s fit.
        count=min(21,theta.shape[1]);t=np.arange(count)*dt
        velocity=x[:,-count:]@((t-t.mean())/np.sum((t-t.mean())**2))
    q=fitted[:,-1]
    p=-coef[:,0]*sine[:,-1]+coef[:,1]*cosine[:,-1]
    p+=(omega**2/9.81)*(v-np.asarray(velocity)/omega)
    return q,p,(theta-fitted)[:,::-1].copy()


def analytic_rollout(q,p,omega,acceleration,dt=.05):
    q=np.asarray(q,dtype=float).copy();p=np.asarray(p,dtype=float).copy()
    phase=np.asarray(omega)*dt;c=np.cos(phase);s=np.sin(phase)
    omc=2*np.sin(phase/2)**2;out=[]
    for a in np.asarray(acceleration).T:
        force=-a/9.81
        q,p=c*q+s*p+omc*force,-s*q+c*p+s*force
        out.append(q.copy())
    return np.stack(out,axis=1)


def refine_windows(windows):
    """Derive observer features from causal context only; targets stay targets."""
    w=dict(windows)
    q,p,res=forced_observer(w['angular_context'][:,:41]*.1,
        w['actuator_context'][:,:41]*.1,w['omega'],velocity=w['initial_velocity'])
    w['observed_q']=w['initial_q'].copy()
    w['harmonic_p']=w['initial_p'].copy()
    w['initial_q']=q;w['initial_p']=p;w['forced_residual']=res
    context=w['angular_context'].copy();context[:,-2:]=np.column_stack([q,p])/.1
    w['angular_context']=context
    return w


def phase_kernel_features(raw,refined,acceleration):
    """Compact physical residual representation, same available information."""
    a=np.asarray(acceleration)
    indices=[1,3,6,9,14,19,29]
    response=analytic_rollout(np.zeros(len(a)),np.zeros(len(a)),raw['omega'],a)[:,indices]
    horizons=np.array([.1,.2,.35,.5,.75,1.,1.5])
    phase=raw['omega'][:,None]*horizons
    return np.column_stack([
        refined['initial_q'],refined['initial_p'],raw['initial_q'],raw['initial_p'],
        raw['omega'],raw['initial_velocity'],
        refined['forced_residual'][:,::3],raw['actuator_context'][:,:41:4],
        raw['actuator_context'][:,41::5],raw['future_u'],a[:,::3],
        response,np.sin(phase),np.cos(phase)])
