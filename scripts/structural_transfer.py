"""Causal control ablation and hidden structural plant mismatch; no training."""
import numpy as np
from scripts.adaptive_actuator import AdaptiveModel, prepare_history, solve

class TransferModel(AdaptiveModel):
    def update(self,position,commands,tick):
        # The frozen branch is bit-identical to the previous winner's update.
        result=super().update(position,commands,tick)
        error=np.zeros(len(position))
        c=self.config['identification']
        if self.spec.get('observer') and position.shape[1]>=c['minimum_samples']:
            from scripts.actuator_identification import fit_with_parameters
            _,_,error=fit_with_parameters(position,commands,self.tau,self.gain)
            if not self.spec.get('frozen_gate'):
                self.observed_valid=error<=c['maximum_fit_rmse_m']
                result['observed_valid']=self.observed_valid.copy()
        result['observer_fit_rmse']=error
        return result


def parameters(frame,calibration):
    p={key:frame[key].to_numpy(dtype=float) for key in ['tau','tau2','gain','deadzone','damping','delay_s','coulomb','switch_factor']}
    p['omega']=calibration.omega(frame.length.to_numpy())*frame.omega_ratio.to_numpy()
    p['forcing']=np.where(frame.force_law=='effective',p['omega']**2/9.81,1/frame.length.to_numpy())
    family=frame.family.to_numpy()
    p['delay_ticks']=np.where(np.isin(family,['delay','combined']),np.rint(p['delay_s']/.01),0).astype(int)
    p['two_lag']=np.isin(family,['two_lag','combined'])
    p['friction']=np.isin(family,['friction','combined'])
    p['gain_change']=np.isin(family,['gain_change','combined'])
    return p


def observable(state,p,cfg):
    s=state[:,:5].copy()
    friction=p['friction']*(p['coulomb']*np.tanh(state[:,1]/cfg['friction_velocity_scale_m_s'])+cfg['viscous_friction_per_s']*state[:,1])
    s[:,4]-=friction
    return s


def rhs(state,command,p,disturbance,time_s,cfg):
    x,v,theta,rate,a,z=state.T
    effective=np.sign(command)*np.maximum(np.abs(command)-p['deadzone'],0.)*p['gain']
    effective*=np.where(p['gain_change']&(time_s>=cfg['gain_switch_time_s']-1e-10),p['switch_factor'],1.)
    acceleration=observable(state,p,cfg)[:,4]
    da=np.where(p['two_lag'],(z-a)/p['tau2'],(effective-a)/p['tau'])
    dz=np.where(p['two_lag'],(effective-z)/p['tau'],0.)
    dr=-p['omega']**2*np.sin(theta)-p['forcing']*acceleration*np.cos(theta)-p['damping']*rate+disturbance
    return np.column_stack([v,acceleration,rate,dr,da,dz])


def plant_step(state,command,p,disturbance,substeps,time_s,cfg):
    # Input, gain-switch side and disturbance are constant over this raw interval.
    s=state.copy();h=.01/substeps
    for _ in range(substeps):
        k1=rhs(s,command,p,disturbance,time_s,cfg)
        k2=rhs(s+h*k1/2,command,p,disturbance,time_s,cfg)
        k3=rhs(s+h*k2/2,command,p,disturbance,time_s,cfg)
        k4=rhs(s+h*k3,command,p,disturbance,time_s,cfg)
        s+=h*(k1+2*k2+2*k3+k4)/6
    return s


def delayed_command(issued,index,delay_ticks):
    past=index-delay_ticks
    return np.where(past>=0,issued[np.arange(len(issued)),np.maximum(past,0)],0.)


def target_at(time_s,cfg):
    return cfg['target_x_m'] if time_s<cfg['reverse_time_s']-1e-10 else cfg['reverse_target_x_m']
