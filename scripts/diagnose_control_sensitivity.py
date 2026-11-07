"""Post-hoc diagnostic, never used to select controller: compare command Jacobians.

All models see the SAME physical-baseline measured history and planned controls.
The independent plant's true state is used only in this explicitly oracle audit.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from scripts.angular_control import AngularModel,prepare_history,expand_plan,nominal_rollout,plant_step
from scripts.run_angular_control import parameters


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    run=Path('artifacts/runs/angular_control_20261107');out=Path('results/angular_control_20261107');dest=out/'sensitivity_diagnostic.json'
    if dest.exists():raise FileExistsError('Preserve diagnosis')
    prov=json.loads((run/'provenance.json').read_text());cfg=prov['config']
    if prov['status']!='complete':raise ValueError('Completed closed-loop experiment first')
    for name,digest in prov['result_hashes'].items():
        if sha256(run/name)!=digest:raise ValueError('Input changed')
    frame=pd.read_csv(run/'evaluate/cases.csv');n=len(frame);eps=.002
    with np.load(run/'evaluate/physical.npz') as z:r={k:z[k] for k in z.files}
    models={name:AngularModel(name) for name in cfg['methods']};params=parameters(frame,models['physical'].calibration)
    repeated={k:np.tile(v,13) for k,v in params.items()};rows=[]
    offsets=torch.cat([torch.zeros(1,6),eps*torch.eye(6),-eps*torch.eye(6)])
    for tick in [20,60,100]:
        k=40+tick*5;meas=r['raw_measurements'][:,k-np.arange(41)]
        batch,initial=prepare_history(meas[:,:,0].copy(),meas[:,:,1].copy(),frame.length.to_numpy(),models['physical'].calibration)
        levels=torch.as_tensor(r['plans'][:,tick,1::5]);committed=torch.as_tensor(r['commands'][:,tick])
        probes=(levels[None]+offsets[:,None]).reshape(-1,6);commands=expand_plan(probes,committed.repeat(13))
        _,_,a=nominal_rollout(commands,initial.repeat(13,1))
        local={key:value.repeat((13,)+(1,)*(value.ndim-1)) for key,value in batch.items()}
        state=np.tile(r['states'][:,tick],(13,1));true=[]
        for step in range(30):
            for sub in range(5):
                absolute=tick*.05+step*.05+sub*.01
                pulse=(absolute>=2.5-1e-10)and(absolute<2.8-1e-10)
                disturbance=np.tile(frame.pulse.to_numpy(),13)*cfg['angular_pulse_rad_s2']*pulse
                state=plant_step(state,commands[:,step].numpy(),repeated,disturbance,cfg['plant_substeps'])
            true.append(state[:,2].copy())
        truth=np.stack(true,1).reshape(13,n,30)
        true_jac=(truth[1:7]-truth[7:13]).transpose(1,2,0)/(2*eps)
        for name,model in models.items():
            prediction=model(local,a).numpy().reshape(13,n,30)
            jac=(prediction[1:7]-prediction[7:13]).transpose(1,2,0)/(2*eps)
            for j in range(n):
                u=true_jac[j].ravel();v=jac[j].ravel();norm=np.linalg.norm(u)
                rows.append(dict(case=j,context=int(frame.context.iloc[j]),tick=tick,method=name,
                    true_jacobian_norm=float(norm),predicted_jacobian_norm=float(np.linalg.norm(v)),
                    locally_unresponsive_plant=bool(norm<1e-6),absolute_jacobian_error=float(np.linalg.norm(v-u)),
                    relative_jacobian_error=float(np.linalg.norm(v-u)/norm) if norm>=1e-6 else np.nan,
                    jacobian_cosine=float(u@v/(norm*np.linalg.norm(v))) if norm>=1e-6 and np.linalg.norm(v)>=1e-6 else np.nan,
                    mean_counterfactual_forecast_error_deg=float(np.abs(np.rad2deg(prediction[0,j]-truth[0,j])).mean())))
    pd.DataFrame(rows).to_csv(out/'sensitivity_diagnostic.csv',index=False)
    result=dict(scope='Exploratory post-hoc oracle comparison, not used for fitting or selection',
        source_sha256=sha256(Path(__file__)),provenance_sha256=sha256(run/'provenance.json'),histories='physical controller only, all96cases, fixedticks20/60/100',
        true_state_used_only_for_oracle=True,future_actual_acceleration_not_given_to_predictors=True,
        finite_difference_epsilon=eps,infinitesimal_probes_not_box_clipped=True,comparisons=len(rows),
        locally_unresponsive_threshold=1e-6,relative_error_undefined_for_zero_plant_jacobian=True)
    dest.write_text(json.dumps(result,indent=2)+'\n');print(pd.DataFrame(rows).groupby('method').mean(numeric_only=True).to_string())

if __name__=='__main__':main()
