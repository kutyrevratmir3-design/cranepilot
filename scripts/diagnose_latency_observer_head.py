"""Post-selection VAL-only diagnosis; unavailable angle truth is used only for scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import load_allowed
from scripts.run_latency_observer import OUT,setup,case_data
from scripts import latency_observer as lo

def main():
    dest=OUT/'head_diagnostic.json'
    if dest.exists():raise FileExistsError('Retain previous diagnosis')
    cfg,qc,pilot,task=setup();rows=[];index=0
    for family,fold in task['families'].items():
        records=load_allowed(task,fold['validation'],'validation')
        for axis in fold['axes']:
            name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration'])
            for sid,age,noise,current,target,labels,raw,p,dated in case_data(records,axis,family,task,qc,cal,cfg,index,[a for a in cfg['train_ages_raw'] if a>0]):
                s=records[sid];truth=s.signals[axis][current['k'][:,None]-np.arange(age)]
                for method in cfg['methods']:
                    estimated=lo.complete_head(raw,age,axis,method,pilot,name)[axis][:,:age];err=np.rad2deg(estimated-truth);rows.append(dict(case=name,scenario=sid,age_ms=age*10,noise_quantum=noise,method=method,windows=len(err),current_angle_mae_deg=float(np.abs(err[:,0]).mean()),current_angle_rmse_deg=float(np.sqrt((err[:,0]**2).mean())),head_mae_deg=float(np.abs(err).mean())))
                print('HEAD_DIAGNOSIS',name,age,noise,flush=True)
            index+=1
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'head_diagnostic.csv',index=False);frame.groupby(['age_ms','noise_quantum','method'],as_index=False)[['current_angle_mae_deg','current_angle_rmse_deg','head_mae_deg']].mean().to_csv(OUT/'head_diagnostic_summary.csv',index=False);dest.write_text(json.dumps(dict(passed=True,source_sha256=sha256(Path(__file__)),data_sha256=sha256(OUT/'head_diagnostic.csv'),rows=len(rows),scope='Late post-selection diagnostic, original VAL only, all positive TRAIN/VALages/noises and adapters; original unreceived encoder angle used only as offline reconstruction target.',used_for_selection=False,independent_confirmation=False),indent=2)+'\n')
if __name__=='__main__':main()
