"""Preregistered new closed-loop angular ablation; no neural training or controller selection."""
import argparse,itertools,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from cranepilot.benchmark import environment_record
from scripts.adaptive_actuator import AdaptiveModel as AngularModel,prepare_history,solve
from scripts.angular_control import plant_step

CONFIG=Path('configs/adaptive_actuator_20261107.json')
DESIGN=Path('docs/ADAPTIVE_ACTUATOR_DESIGN_20261107.md')


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def cases(cfg,stage):
    rows=[]
    if stage=='development':
        for L,angle in itertools.product(cfg['development_lengths_m'],[2.,6.]):
            rows.append(dict(length=L,omega_ratio=1.,tau=.1,gain=1.,damping=.04,deadzone=.01,
                             force_law='effective',angle_deg=angle,noisy=False,pulse=False))
    else:
        contexts=itertools.product(cfg['evaluation_lengths_m'],cfg['evaluation_omega_ratios'],cfg['evaluation_tau_s'],cfg['evaluation_gains'])
        for i,(L,r,tau,gain) in enumerate(contexts):
            for noisy,pulse in itertools.product([False,True],repeat=2):
                rows.append(dict(context=i,length=L,omega_ratio=r,tau=tau,gain=gain,
                    damping=[.02,.08][(i%2)^((i//2)%2)],deadzone=[0.,.02][((i//2)%2)^((i//4)%2)],
                    force_law=['effective','geometric'][(i%2)^((i//4)%2)],angle_deg=[3.,5.][(i%2)^((i//2)%2)^((i//4)%2)],noisy=noisy,pulse=pulse))
    return pd.DataFrame(rows)


def parameters(frame,calibration):
    p={key:frame[key].to_numpy(dtype=float) for key in ['tau','gain','deadzone','damping']}
    p['omega']=calibration.omega(frame.length.to_numpy())*frame.omega_ratio.to_numpy()
    p['forcing']=np.where(frame.force_law=='effective',p['omega']**2/9.81,1/frame.length.to_numpy())
    return p


def exogenous(frame,cfg):
    n=len(frame);steps=40+int(round(cfg['duration_s']/.01))+1
    # Generate both sensors for every case, then mask; indexed by physical time.
    rng=np.random.default_rng(cfg['noise_seed']);noise=rng.normal(size=(n,steps,2))
    noise[:,:,0]*=.25*cfg['angle_quantum_rad'];noise[:,:,1]*=.25*cfg['position_quantum_m']
    noise*=frame.noisy.to_numpy()[:,None,None]
    grid=np.arange(steps)*.01-.4
    interval=cfg['angular_pulse_interval_s'];pulse=(grid>=interval[0]-1e-10)&(grid<interval[1]-1e-10)
    disturbance=frame.pulse.to_numpy()[:,None]*pulse[None]*cfg['angular_pulse_rad_s2']
    return noise,disturbance


def observe(state,noise,noisy,cfg):
    measurement=state[:,[2,0]]+noise
    for j,quantum in enumerate([cfg['angle_quantum_rad'],cfg['position_quantum_m']]):
        measurement[:,j]=np.where(noisy,np.round(measurement[:,j]/quantum)*quantum,measurement[:,j])
    return measurement


def experiment(frame,cfg,model,folder):
    n=len(frame);p=parameters(frame,model.calibration);noise,disturbance=exogenous(frame,cfg)
    state=np.zeros((n,5));state[:,0]=cfg['initial_x_m'];state[:,2]=np.deg2rad(frame.angle_deg.to_numpy())
    history=[observe(state,noise[:,0],frame.noisy.to_numpy(),cfg)];raw_states=[state.copy()];raw_measurements=[history[0].copy()]
    for j in range(40):
        state=plant_step(state,np.zeros(n),p,np.zeros(n),cfg['plant_substeps'])
        history.append(observe(state,noise[:,j+1],frame.noisy.to_numpy(),cfg));raw_states.append(state.copy());raw_measurements.append(history[-1].copy())
    history=np.stack(history,1);initial_state=state.copy()
    committed=torch.zeros(n);warm=torch.zeros(n,6);target=torch.full((n,),cfg['target_x_m'])
    records=[state.copy()];measurements=[history[:,-1].copy()];commands=[];plans=[];predicted=[];estimated=[];timings=[];costs=[];warm_costs=[];failures=[];fallbacks=[];identification=[]
    steps=int(round(cfg['duration_s']/.05))
    for tick in range(steps):
        started=time.perf_counter()
        end=40+tick*5;start=max(0,end-cfg['identification']['history_intervals']*5)
        past_position=np.stack(raw_measurements[start:end+1:5],1)[:,:,1].copy()
        applied_past=np.column_stack([np.zeros((n,8))]+[c[:,None] for c in commands])[:,start//5:end//5].copy()
        ident=model.update(past_position,applied_past,tick);identification.append(ident)
        batch,initial=prepare_history(history[:,::-1,0].copy(),history[:,::-1,1].copy(),frame.length.to_numpy(),model.calibration)
        plan,warm,pred,diagnostics=solve(model,batch,initial,committed,warm,target,cfg['solver'])
        elapsed=time.perf_counter()-started
        applied=committed.numpy().copy();commands.append(applied);plans.append(plan.numpy());predicted.append(pred['theta'].numpy());estimated.append(model.used_initial(initial).numpy())
        costs.append(diagnostics['cost'].numpy());warm_costs.append(diagnostics['warm_cost'].numpy());failures.append(diagnostics['nonfinite'].numpy());fallbacks.append(diagnostics['fallback'].numpy());timings.append(elapsed)
        for sub in range(5):
            idx=40+tick*5+sub
            state=plant_step(state,applied,p,disturbance[:,idx],cfg['plant_substeps'])
            sample=observe(state,noise[:,idx+1],frame.noisy.to_numpy(),cfg)
            history=np.concatenate([history[:,1:],sample[:,None]],1);raw_states.append(state.copy());raw_measurements.append(sample.copy())
        if not np.isfinite(state).all():raise FloatingPointError('Nonfinite plant; retained failed provenance')
        records.append(state.copy());measurements.append(history[:,-1].copy());committed=plan[:,1].clone()
        if tick%20==0:print('TICK',model.name,tick,'batch_seconds',round(elapsed,3),flush=True)
    result=dict(raw_states=np.stack(raw_states,1),raw_measurements=np.stack(raw_measurements,1),states=np.stack(records,1),measurements=np.stack(measurements,1),commands=np.stack(commands,1),plans=np.stack(plans,1),
                predicted_theta=np.stack(predicted,1),initial_estimates=np.stack(estimated,1),cost=np.stack(costs,1),warm_cost=np.stack(warm_costs,1),
                failures=np.stack(failures,1),fallbacks=np.stack(fallbacks,1),batch_seconds=np.array(timings),**{'id_'+key:np.stack([r[key] for r in identification],axis=1) for key in identification[0]},initial_state=initial_state,noise=noise,disturbance=disturbance)
    assert np.max(np.abs(result['commands']))<=cfg['solver']['command_bound']+1e-6
    assert np.all(result['cost']<=result['warm_cost']+1e-3)
    np.testing.assert_array_equal(result['commands'][:,1:],result['plans'][:,:-1,1])
    np.savez_compressed(folder/f'{model.name}.npz',**result)
    return result


def metric_rows(frame,result,cfg,method):
    s=result['states'];dt=.05;grid=np.arange(s.shape[1])*dt;theta=np.rad2deg(s[:,:,2]);error=s[:,:,0]-cfg['target_x_m']
    rows=[]
    for i,case in frame.iterrows():
        settled=(np.abs(error[i])<.01)&(np.abs(s[i,:,1])<.02)&(np.abs(theta[i])<1.)
        stay=np.logical_and.accumulate(settled[::-1])[::-1];hits=np.flatnonzero(stay)
        stage=(error[i]/.05)**2+(s[i,:,2]/.035)**2+.02*(s[i,:,1]/.2)**2+.05*(s[i,:,4]/.5)**2
        total=float(np.trapezoid(stage,grid)+5*(error[i,-1]/.05)**2+5*(s[i,-1,2]/.035)**2+(s[i,-1,1]/.1)**2)
        rows.append(dict(case=int(i),method=method,**case.to_dict(),realized_cost=total,
            tracking_ise_m2_s=float(np.trapezoid(error[i]**2,grid)),sway_ise_deg2_s=float(np.trapezoid(theta[i]**2,grid)),
            tail_rms_deg=float(np.sqrt(np.mean(theta[i,grid>=cfg['duration_s']-2]**2))),peak_angle_deg=float(np.max(np.abs(theta[i]))),
            final_position_error_m=float(abs(error[i,-1])),effort_acceleration2=float(np.trapezoid(s[i,:,4]**2,grid)),
            command_variation=float(np.abs(np.diff(np.r_[0,result['commands'][i]])).sum()),
            settled=bool(hits.size),settling_time_s=float(grid[hits[0]]) if hits.size else None,
            rail_violation=bool(((s[i,:,0]<0)|(s[i,:,0]>.5)).any()),angle_violation=bool((abs(theta[i])>15).any()),speed_violation=bool((abs(s[i,:,1])>.5).any()),
            solver_nonfinite=int(result['failures'][i].sum()),fallbacks=int(result['fallbacks'][i].sum()),
            current_acceleration_estimate_rmse=float(np.sqrt(np.mean((result['initial_estimates'][i,:,2]-s[i,:-1,4])**2)))))
    for i,row in enumerate(rows):
        row['current_velocity_estimate_rmse']=float(np.sqrt(np.mean((result['initial_estimates'][i,:,1]-s[i,:-1,1])**2)))
        row['tail_tau_error_s']=float(np.mean(np.abs(result['id_tau'][i,-40:]-frame.tau.iloc[i])))
        row['tail_gain_error']=float(np.mean(np.abs(result['id_gain'][i,-40:]-frame.gain.iloc[i])))
        row['id_update_fraction']=float(result['id_accepted'][i].mean())
        row['observer_use_fraction']=float(result['id_observed_valid'][i].mean())
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['development','evaluate'],required=True)
    parser.add_argument('--output',type=Path,default=Path('artifacts/runs/adaptive_actuator_20261107'));a=parser.parse_args()
    cfg=json.loads(CONFIG.read_text());torch.set_num_threads(cfg['threads']);out=a.output
    locked=[CONFIG,DESIGN,Path('scripts/adaptive_actuator.py'),Path('scripts/actuator_identification.py'),Path('scripts/angular_control.py'),Path(__file__).relative_to(Path.cwd()),Path('tests/test_adaptive_actuator.py')]
    if a.stage=='development':
        for p in locked:
            if subprocess.check_output(['git','show',f'HEAD:{p}'])!=p.read_bytes():raise ValueError(f'Commit first: {p}')
        if out.exists():raise FileExistsError('Retain earlier attempts')
        out.mkdir(parents=True)
        paths=locked+list(Path(cfg['bundle']).rglob('*'))
        prov=dict(status='development_running',config=cfg,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            guarded_hashes={str(p):sha256(p) for p in paths if p.is_file()},environment=environment_record(),neural_training=False,online_actuator_identification=True,available_actuator_history_s=2.,angular_history_s=.4,
            scope=cfg['scope'],started_utc=pd.Timestamp.now(tz='UTC').isoformat())
        write(out/'provenance.json',prov)
    else:
        prov=json.loads((out/'provenance.json').read_text())
        if prov['status']!='development_complete':raise ValueError('Frozen development first')
        for name,digest in prov['guarded_hashes'].items():
            if sha256(Path(name))!=digest:raise ValueError(f'Changed source: {name}')
        for name,digest in prov['development_hashes'].items():
            if sha256(out/name)!=digest:raise ValueError('Development changed')
    folder=out/a.stage;folder.mkdir();frame=cases(cfg,a.stage);frame.to_csv(folder/'cases.csv',index=False)
    try:
        rows=[]
        for method in cfg['methods']:
            model=AngularModel(method);result=experiment(frame,cfg,model,folder);rows+=metric_rows(frame,result,cfg,method)
            print('METHOD_COMPLETE',a.stage,method,len(frame),flush=True)
        pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
        if a.stage=='development':
            prov.update(status='development_complete',selection_performed=False,parameters_retuned=False,
                development_hashes={str(p.relative_to(out)):sha256(p) for p in folder.rglob('*') if p.is_file()},locked_utc=pd.Timestamp.now(tz='UTC').isoformat())
        else:prov.update(status='complete',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),result_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
    except Exception as e:
        prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(out/'provenance.json',prov)
    print('COMPLETE',a.stage,flush=True)

if __name__=='__main__':main()
