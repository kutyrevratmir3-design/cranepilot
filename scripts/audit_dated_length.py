"""Post-freeze coverage diagnostic; no selection or fitting."""
import json
import numpy as np
import pandas as pd
from cranepilot.physics import LengthCalibration
from scripts.run_dated_forecast import RUN,OUT,setup
from scripts.run_latency_observer import case_data
from scripts.modal_history import load_allowed
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import write

def main():
    if (OUT/'length_coverage.json').exists():raise FileExistsError('Retain prior diagnostic')
    cfg,qc,pilot,task=setup();rows=[];index=0
    for family,fold in task['families'].items():
        for axis in fold['axes']:
            name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration'])
            for role,records,ages in [('validation',load_allowed(task,fold['validation'],'validation'),cfg['train_ages_raw']),('old_evaluation',load_confirmation(task,fold['confirmation']),cfg['evaluation_ages_raw'])]:
                # Clean only: support/length are identical in both angle-noise conditions.
                clean=dict(cfg,noise_quantum_multipliers=[0])
                for sid,age,noise,current,target,labels,raw,p,dated in case_data(records,axis,family,task,qc,cal,clean,index,ages):
                    file=RUN/name/f'validation_s{sid}_a{age}_n0.npz' if role=='validation' else RUN/'retrospective'/f'{name}_s{sid}_a{age}_n0.npz'
                    with np.load(file) as z:delta=np.abs(z['direct_variable_length']-z['direct_free2p5']).max(1)
                    rows.append(dict(role=role,case=name,scenario=int(sid),age_ms=age*10,windows=len(target),past4s_length_changed=int((np.ptp(raw['L'],axis=1)>0).sum()),sensor_gap_length_changed=int((np.ptp(raw['L'][:,:age+1],axis=1)>0).sum()),different_forecasts=int((delta>1e-12).sum()),maximum_difference_deg=float(np.rad2deg(delta.max()))))
            index+=1
    pd.DataFrame(rows).to_csv(OUT/'length_coverage.csv',index=False);write(OUT/'length_coverage.json',dict(passed=True,posthoc_diagnostic=True,used_for_selection=False,validation_sensor_gap_changed=sum(r['sensor_gap_length_changed'] for r in rows if r['role']=='validation'),old_sensor_gap_changed=sum(r['sensor_gap_length_changed'] for r in rows if r['role']=='old_evaluation'),validation_forecasts_differ=sum(r['different_forecasts'] for r in rows if r['role']=='validation'),old_forecasts_differ=sum(r['different_forecasts'] for r in rows if r['role']=='old_evaluation'),repeated_windows_across_ages_are_not_independent=True))
if __name__=='__main__':main()
