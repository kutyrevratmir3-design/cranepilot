"""Forecast from a history-only NPZ and retained local model weights.

Input: theta/x/reference arrays (N,41), newest first; length_m (N,);
future_reference (N,30), prescribed at 50 ms including the origin command.
Angles are radians, x/length are metres, reference retains recorded native scale.
No dataset file, measured future state or target array is required.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration,oscillator_forecast
from cranepilot.rollout import NeuralActuator,tensors,HORIZON_INDEX,HORIZONS_S
from cranepilot.forced_observer import refine_windows,analytic_rollout
from cranepilot.innovation import InnovationForecaster
from cranepilot.phase_normalized import PhaseNormalizedForecaster,GRUResidualForecaster,phase_context
from cranepilot.fast_inference import fast_prediction


def build_packet(theta,x,reference,length,future_reference,calibration):
    theta,x,reference,future_reference=[np.asarray(v,dtype=float) for v in (theta,x,reference,future_reference)]
    if theta.ndim!=2 or theta.shape[1]!=41 or x.shape!=theta.shape or reference.shape!=theta.shape:
        raise ValueError('Past inputs must share shape (N,41), newest first')
    if future_reference.shape!=(len(theta),30):raise ValueError('Future reference must have shape (N,30)')
    length=np.broadcast_to(np.asarray(length,dtype=float),(len(theta),))
    if not all(np.isfinite(v).all() for v in (theta,x,reference,length,future_reference)) or np.any(length<=0):
        raise ValueError('Finite inputs and positive cable length required')
    if not np.allclose(reference[:,0],future_reference[:,0],atol=1e-9,rtol=1e-9):
        raise ValueError('First prescribed command must match the origin command')
    omega=calibration.omega(length)
    harmonic,coef=oscillator_forecast(theta,omega,HORIZONS_S)
    t=-np.arange(21)*.01;derivative=(t-t.mean())/np.sum((t-t.mean())**2)
    return {'omega':omega,'initial_x':x[:,0],'initial_velocity':x[:,:21]@derivative,
        'initial_q':theta[:,0],'initial_p':coef[:,1],
        'actuator_context':np.column_stack([(x-x[:,:1])/.1,reference/2]),
        'angular_context':np.column_stack([theta/.1,(x-x[:,:1])/.1,reference/2,coef/.1]),
        'future_u':future_reference,'harmonic':harmonic}


def predict(run,variant,inputs,actor_path=Path('artifacts/runs/actuator_development/actuator.pt'),fast=False):
    p=json.loads((run/'provenance.json').read_text())
    if p['status']!='complete':raise ValueError('Only completed research runs may be loaded')
    if variant not in p['config']['variants']:raise ValueError('Variant absent from retained configuration')
    expected=next(h for path,h in p['guarded_hashes'].items() if path.replace('\\','/').endswith('/actuator.pt'))
    if sha256(actor_path)!=expected:raise ValueError('Actuator checkpoint changed')
    raw=build_packet(inputs['theta'],inputs['x'],inputs['reference'],inputs['length_m'],inputs['future_reference'],LengthCalibration(**p['calibration']))
    actor=NeuralActuator();actor.load_state_dict(torch.load(actor_path,weights_only=True));actor.eval()
    torch.set_num_threads(4)
    with torch.no_grad():acceleration=actor(tensors(raw))[1]
    if variant=='raw_driven_huber':w=raw
    else:w=refine_windows(raw)
    if variant in ['phase_passive','phase_driven']:
        w['phase_context']=phase_context(raw,w)
    if variant=='gru_reference':
        w['reference_context']=np.column_stack([raw['angular_context'],raw['omega'],raw['initial_x'],raw['initial_velocity']])
        w['physical_base']=analytic_rollout(w['initial_q'],w['initial_p'],w['omega'],acceleration.numpy())
    batch=tensors(w);ys=[]
    for seed in p['config']['seeds']:
        name=f'{variant}_seed{seed}.pt'
        if sha256(run/name)!=p['checkpoint_hashes'][name]:raise ValueError('Angular checkpoint changed')
        if variant=='gru_reference':model=GRUResidualForecaster()
        elif variant in ['phase_passive','phase_driven']:model=PhaseNormalizedForecaster(variant=='phase_driven')
        else:model=InnovationForecaster(driven=variant!='forced_observer_huber',measurement_anchor=variant=='measurement_driven')
        model.load_state_dict(torch.load(run/name,weights_only=True));model.eval()
        with torch.no_grad():
            if fast and isinstance(model,InnovationForecaster):y=fast_prediction(model,batch,acceleration)
            else:y,_=model(batch,acceleration)
        ys.append(y.numpy()[:,HORIZON_INDEX])
    return np.mean(ys,axis=0),np.std(ys,axis=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--variant',required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actuator',type=Path,default=Path('artifacts/runs/actuator_development/actuator.pt'))
    p.add_argument('--fast',action='store_true',help='Compile fixed angular flow once per forecast window')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a fresh output file')
    with np.load(a.input,allow_pickle=False) as data:mean,std=predict(a.run,a.variant,data,a.actuator,fast=a.fast)
    result=pd.DataFrame({f'angle_deg_{h:g}s':np.rad2deg(mean[:,j]) for j,h in enumerate(HORIZONS_S)})
    for j,h in enumerate(HORIZONS_S):result[f'seed_std_deg_{h:g}s']=np.rad2deg(std[:,j])
    a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
    print('Forecast saved; seed spread is descriptive, not a calibrated confidence interval.')


if __name__=='__main__':main()
