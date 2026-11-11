"""Warm single-packet latency: preprocessing, shared actor and predictor."""
from pathlib import Path
import json
import platform
import time
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import NeuralActuator,tensors,HORIZON_INDEX
from cranepilot.innovation import InnovationForecaster
from cranepilot.forced_observer import refine_windows,analytic_rollout
from predict_cranepilot import build_packet
from cranepilot.fast_inference import fast_prediction


def main():
    root=Path('artifacts/runs/forceaware_campaign_v2')
    provenance=json.loads((root/'provenance.json').read_text())
    if provenance['status']!='complete':raise ValueError('Completed model campaign required')
    fixture=np.load('.local/inference_smoke_input.npz')
    inputs={key:fixture[key][:1] for key in fixture.files}
    calibration=LengthCalibration(**provenance['calibration']);torch.set_num_threads(4)
    actor=NeuralActuator();actor.load_state_dict(torch.load('artifacts/runs/actuator_development/actuator.pt',weights_only=True));actor.eval()
    models=[]
    for seed in [0,1,2]:
        m=InnovationForecaster(True);m.load_state_dict(torch.load(root/f'forced_driven_huber_seed{seed}.pt',weights_only=True));m.eval();models.append(m)
    gp=joblib.load(root/'full_kernel.joblib')
    def prepare():
        raw=build_packet(inputs['theta'],inputs['x'],inputs['reference'],inputs['length_m'],inputs['future_reference'],calibration)
        position,acceleration=actor(tensors(raw));return raw,position,acceleration
    def neural():
        raw,_,acceleration=prepare();b=tensors(refine_windows(raw))
        return torch.stack([m(b,acceleration)[0][:,HORIZON_INDEX] for m in models]).mean(0)
    def kernel():
        raw,position,acceleration=prepare();acc=acceleration.numpy()
        features=np.column_stack([raw['angular_context'],raw['omega'],raw['initial_x'],raw['initial_velocity'],
            raw['future_u'],acc,position.numpy()-raw['initial_x'][:,None]])
        base=analytic_rollout(raw['initial_q'],raw['initial_p'],raw['omega'],acc)[:,HORIZON_INDEX]
        return gp['gp'].predict(gp['scaler'].transform(features))+base
    def compiled():
        raw,_,acceleration=prepare();b=tensors(refine_windows(raw))
        return torch.stack([fast_prediction(m,b,acceleration)[:,HORIZON_INDEX] for m in models]).mean(0)
    results={}
    with torch.no_grad(),threadpool_limits(limits=4):
        for name,function in [('three_seed_reference_flow',neural),('three_seed_compiled_flow',compiled),('full_gp_1024',kernel)]:
            for _ in range(5):function()
            times=[]
            for _ in range(30):
                start=time.perf_counter();function();times.append((time.perf_counter()-start)*1000)
            results[name]={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'repetitions':len(times)}
    report={'scope':'warm single validation packet, preprocessing plus actor plus predictor; excludes model loading, data acquisition and I/O; not a hard real-time guarantee',
            'platform':platform.platform(),'processor':platform.processor(),'torch_threads':4,'models':results}
    Path('results/latest/inference_latency.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
