"""Reintegrate every closed-loop trajectory and replay every chosen forecast plan."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from scripts.adaptive_actuator import AdaptiveModel as AngularModel,prepare_history
from scripts.angular_control import plant_step,expand_plan
from scripts.run_adaptive_actuator import parameters,exogenous,observe,metric_rows
from scripts.verify_finetune_campaign import original_files


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,default=Path('artifacts/runs/adaptive_actuator_20261107'))
    ap.add_argument('--output',type=Path,default=Path('results/adaptive_actuator_20261107'));a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);dest=out/'verification.json'
    if dest.exists():raise FileExistsError('Preserve prior verification')
    p=json.loads((a.run/'provenance.json').read_text());cfg=p['config']
    if p['status']!='complete':raise ValueError('Completed experiment required')
    torch.set_num_threads(cfg['threads']);report=dict(passed=False,provenance_sha256=sha256(a.run/'provenance.json'),verifier_sha256=sha256(Path(__file__)),checks=[])
    try:
        for mapping,base in [('guarded_hashes',Path('.')),('result_hashes',a.run)]:
            for name,digest in p[mapping].items():
                if sha256(base/name)!=digest:raise ValueError(f'Changed {name}')
        report['original_files']=original_files()
        assert report['original_files']['passed']
        rawmetrics=[]
        for stage in ['development','evaluate']:
            folder=a.run/stage;frame=pd.read_csv(folder/'cases.csv');expected_noise,expected_disturbance=exogenous(frame,cfg)
            expected_rows=pd.read_csv(folder/'metrics.csv')
            for method in cfg['methods']:
                model=AngularModel(method);params=parameters(frame,model.calibration)
                with np.load(folder/f'{method}.npz') as f:r={k:f[k] for k in f.files}
                np.testing.assert_array_equal(r['noise'],expected_noise);np.testing.assert_array_equal(r['disturbance'],expected_disturbance)
                np.testing.assert_array_equal(r['raw_states'][:,40::5],r['states'])
                np.testing.assert_array_equal(r['raw_measurements'][:,40::5],r['measurements'])
                np.testing.assert_array_equal(r['commands'][:,1:],r['plans'][:,:-1,1]);np.testing.assert_array_equal(r['commands'][:,0],0)
                assert np.max(np.abs(r['plans']))<=cfg['solver']['command_bound']+1e-6
                assert np.all(r['cost']<=r['warm_cost']+1e-3)
                state=r['raw_states'][:,0].copy();fine=state.copy();maximum=0.;fine_error=0.;observation_error=0.
                for k in range(r['raw_states'].shape[1]-1):
                    command=np.zeros(len(frame)) if k<40 else r['commands'][:,(k-40)//5]
                    state=plant_step(state,command,params,r['disturbance'][:,k],cfg['plant_substeps'])
                    fine=plant_step(fine,command,params,r['disturbance'][:,k],cfg['plant_substeps']*2)
                    maximum=max(maximum,float(np.max(np.abs(state-r['raw_states'][:,k+1]))))
                    fine_error=max(fine_error,float(np.max(np.abs(fine-state))))
                    obs=observe(state,r['noise'][:,k+1],frame.noisy.to_numpy(),cfg)
                    observation_error=max(observation_error,float(np.max(np.abs(obs-r['raw_measurements'][:,k+1]))))
                assert maximum<1e-12 and observation_error<1e-12 and fine_error<1e-5
                pred_error=0.;cost_error=0.;estimate_error=0.
                for tick in range(r['commands'].shape[1]):
                    k=40+tick*5;past=k-np.arange(41)
                    measured=r['raw_measurements'][:,past]
                    batch,initial=prepare_history(measured[:,:,0].copy(),measured[:,:,1].copy(),frame.length.to_numpy(),model.calibration)
                    start=max(0,k-cfg['identification']['history_intervals']*5)
                    position=r['raw_measurements'][:,np.arange(start,k+1,5),1].copy()
                    command_history=np.concatenate([np.zeros((len(frame),8)),r['commands'][:,:tick]],1)[:,start//5:k//5].copy()
                    ident=model.update(position,command_history,tick)
                    for key,value in ident.items():np.testing.assert_array_equal(value,r['id_'+key][:,tick])
                    used=model.used_initial(initial)
                    estimate_error=max(estimate_error,float(np.max(np.abs(used.numpy()-r['initial_estimates'][:,tick]))))
                    warm=torch.zeros(len(frame),6) if tick==0 else torch.as_tensor(np.concatenate([r['plans'][:,tick-1,2:],r['plans'][:,tick-1,-1:]],1)[:,::5].copy())
                    committed=torch.as_tensor(r['commands'][:,tick].copy())
                    ctx=model.context(batch,initial,expand_plan(warm,committed))
                    assert float(ctx['offset'].abs().max())<=cfg['residual_bound_rad']+1e-8
                    with torch.no_grad():cost,pred=model.evaluate(ctx,torch.as_tensor(r['plans'][:,tick]),torch.full((len(frame),),cfg['target_x_m']))
                    pred_error=max(pred_error,float(np.max(np.abs(np.rad2deg(pred['theta'].numpy()-r['predicted_theta'][:,tick])))))
                    cost_error=max(cost_error,float(np.max(np.abs(cost.numpy()-r['cost'][:,tick]))))
                assert pred_error<5e-5 and cost_error<.001 and estimate_error==0,(pred_error,cost_error,estimate_error)
                recomputed=pd.DataFrame(metric_rows(frame,r,cfg,method));saved=expected_rows[expected_rows.method==method].reset_index(drop=True)
                numeric=list(saved.select_dtypes('number'));np.testing.assert_allclose(recomputed[numeric].astype(float),saved[numeric],rtol=1e-12,atol=1e-12,equal_nan=True)
                # Independent task metric computation, not the reporting helper.
                s=r['states'];grid=np.arange(s.shape[1])*.05
                np.testing.assert_allclose(np.trapezoid((s[:,:,0]-cfg['target_x_m'])**2,grid,axis=1),saved.tracking_ise_m2_s,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(np.trapezoid(np.rad2deg(s[:,:,2])**2,grid,axis=1),saved.sway_ise_deg2_s,rtol=1e-12,atol=1e-12)
                for j in range(len(frame)):
                    raw=r['raw_states'][j,40:]
                    rawmetrics.append(dict(stage=stage,method=method,case=j,peak_angle_100hz_deg=float(np.rad2deg(np.abs(raw[:,2])).max()),
                        rail_violation_100hz=bool(((raw[:,0]<0)|(raw[:,0]>.5)).any()),speed_violation_100hz=bool((np.abs(raw[:,1])>.5).any()),
                        angle_violation_100hz=bool((np.abs(np.rad2deg(raw[:,2]))>15).any())))
                check=dict(stage=stage,method=method,episodes=len(frame),decisions=len(frame)*r['commands'].shape[1],
                    plant_replay_max_state_difference=maximum,measurement_replay_max_difference=observation_error,
                    rk_halfstep_max_state_difference=fine_error,forecast_replay_max_deg=pred_error,cost_replay_max_difference=cost_error,
                    initial_estimates_exact=True,online_identification_states_exact=True,paired_noise_exact=True,paired_disturbances_exact=True)
                report['checks'].append(check);print('VERIFIED',json.dumps(check),flush=True)
        pd.DataFrame(rawmetrics).to_csv(out/'raw_sampling_metrics.csv',index=False)
        report.update(passed=True,episodes=sum(x['episodes'] for x in report['checks']),decisions=sum(x['decisions'] for x in report['checks']),
                      model_weights_trained=False,optimizer_reexecuted=False,plant_reintegrated=True,all_chosen_forecasts_replayed=True)
    except Exception as e:report['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
