"""No fitting: qualify frozen kernels on every past-motion regime of old records."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.benchmark import environment_record
from cranepilot.reliability_pilot import FrozenReliabilityPilot
from scripts.modal_history import load_allowed
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import write
from scripts.active_qualification import qualification_packets,frozen_predictions

CONFIG=Path('configs/active_qualification_20261107.json');RUN=Path('artifacts/runs/active_qualification_20261107')
LOCKED=[CONFIG,Path('docs/ACTIVE_QUALIFICATION_DESIGN_20261107.md'),Path('scripts/active_qualification.py'),Path('scripts/run_active_qualification.py'),Path('tests/test_active_qualification.py')]

def existing_records(task,fold):
    parts=[('train',load_allowed(task,fold['train'],'train')),('validation',load_allowed(task,fold['validation'],'validation')),('old_evaluation',load_confirmation(task,fold['confirmation']))];records={};roles={}
    for role,items in parts:
        for sid,seq in items.items():
            if sid in records:raise ValueError('Duplicate role')
            records[sid]=seq;roles[sid]=role
    return records,roles

def groups(labels,cfg):
    return dict(all=np.ones(len(labels['regime']),dtype=bool),**{name:labels['regime']==j for j,name in enumerate(cfg['regime_names'])},old_numerical_scope=labels['old_numerical_scope'])

def main():
    cfg=json.loads(CONFIG.read_text())
    for path in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit first {path}')
    if RUN.exists():raise FileExistsError('Retain all prior qualifications')
    pilot=FrozenReliabilityPilot(cfg['bundle']);task=pilot.cfg;assert task==json.loads(Path(cfg['task_source']).read_text());RUN.mkdir(parents=True);bundle=Path(cfg['bundle']);paths=LOCKED+[Path(cfg['data'])/'dataset.npz',Path(cfg['data'])/'manifest.json']+[f for f in bundle.rglob('*') if f.is_file()];sources=set(pilot.manifest['source_hashes'])|set(pilot.reliability_manifest['source_hashes']);paths += [Path(p) for p in sources]
    prov=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,task=task,guarded_hashes={str(f):sha256(f) for f in paths},environment=environment_record(),fitting_performed=False,selection_performed=False,api_gate_changed=False,all_records_previously_opened=True,independent_confirmation=False,final_goal_achieved=False);write(RUN/'provenance.json',prov);metrics=[];coverage=[];ids=[];packets_count=0
    try:
        for family,fold in task['families'].items():
            records,roles=existing_records(task,fold);ids += list(records)
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.cases[name]['calibration'])
                for sid,s in records.items():
                    p,target,labels=qualification_packets(s,axis,task,cfg,cal);y,diag=frozen_predictions(p,pilot,name);assert set(y)==set(cfg['methods']);packets_count+=len(p['k']);np.savez_compressed(RUN/f'{name}_s{sid}.npz',k=p['k'],**labels,**y,**{'diagnostic_'+k:v for k,v in diag.items()})
                    for group,mask in groups(labels,cfg).items():
                        count=int(mask.sum());coverage.append(dict(case=name,scenario=sid,role=roles[sid],group=group,windows=count,total_windows=len(mask),fraction=count/len(mask)))
                        for method,pred in y.items():
                            for j,h in enumerate(task['horizons_raw']):
                                e=np.rad2deg(pred[mask,j]-target[mask,j]);metrics.append(dict(case=name,scenario=sid,role=roles[sid],group=group,method=method,horizon_s=h*.01,windows=count,mae_deg=float(np.abs(e).mean()) if count else np.nan,rmse_deg=float(np.sqrt((e**2).mean())) if count else np.nan))
                    print('QUALIFIED',name,sid,roles[sid],len(p['k']),'regimes',np.bincount(labels['regime'],minlength=4).tolist(),flush=True)
        assert sorted(set(ids))==list(range(1,34));pd.DataFrame(metrics).to_csv(RUN/'metrics.csv',index=False);pd.DataFrame(coverage).to_csv(RUN/'coverage.csv',index=False);prov.update(status='complete',records_read=sorted(set(ids)),axis_records=53,packets=packets_count,result_hashes={str(f.relative_to(RUN)):sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name!='provenance.json'});print('COMPLETE',packets_count,'packets',flush=True)
    except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(RUN/'provenance.json',prov)
if __name__=='__main__':main()
