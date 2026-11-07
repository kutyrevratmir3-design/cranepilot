"""Fixed forced adaptive physics variants and three residual-transfer controls."""
import numpy as np
import torch
from cranepilot.rollout import tensors
from cranepilot.forced_observer import refine_windows
from scripts.combined_candidate import CombinedCandidate
from scripts.forced_adaptive_physics import forced_fit, damped_rollout, adaptive_forced_forecast
from scripts.run_broad_search import inputs, predict
from scripts.reliability_blend import blend


def compose(soft, nominal, adaptive, rms, tau=.5):
    residual = soft - nominal
    g = blend(soft, nominal, rms, tau)[1][:,None]
    return {'residual_transfer':adaptive + residual,
            'residual_gate':adaptive + g*residual,
            'soft_adaptive_gate':adaptive + g*(soft-adaptive)}


class ForcedAdaptiveCandidate(CombinedCandidate):
    def forecast(self, raw, prescribed_acceleration=None):
        if prescribed_acceleration is None:
            with torch.no_grad(): acceleration = self.actor(tensors(raw))[1].numpy()
        else:
            acceleration = np.asarray(prescribed_acceleration,dtype=float)
        if acceleration.shape != (len(raw['omega']),30) or not np.isfinite(acceleration).all():
            raise ValueError('Finite (N,30) acceleration schedule required')
        batch = inputs(raw)
        batch['acceleration'] = torch.as_tensor(acceleration,dtype=torch.float32)
        seeds = {name:np.stack([predict(model,batch) for model in models]) for name,models in self.models.items()}
        pred = {name:y.mean(0) for name,y in seeds.items()}
        theta = raw['angular_context'][:,:41]*.1
        x = raw['actuator_context'][:,:41]*.1
        q,v,res = forced_fit(theta,x,raw['omega'],0.,velocity=raw['initial_velocity'])
        nominal = damped_rollout(q,v,raw['omega'],0.,acceleration)
        pred['physical'] = nominal
        # Freeze old nominal observer residual, not the optimistically selected adaptive residual.
        rms = np.sqrt(np.mean(refine_windows(raw)['forced_residual']**2,axis=1))
        pred['combined'] = blend(pred['soft_symmetry'],nominal,rms,self.config['tau_bins'])[0]
        metadata = {}
        for name,choice in self.config['variants'].items():
            pred[name],metadata[name] = adaptive_forced_forecast(theta,x,raw['omega'],acceleration,
                velocity=raw['initial_velocity'],**choice)
        pred.update(compose(pred['soft_symmetry'],nominal,pred['adaptive'],rms,self.config['tau_bins']))
        if not all(np.isfinite(y).all() for y in pred.values()): raise ValueError('Nonfinite forecast')
        return pred,metadata,acceleration
