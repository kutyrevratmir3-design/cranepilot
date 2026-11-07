"""Posthoc explanatory diagnostics; no predictor fitting or modification."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from scripts.external_crane_data import config,load_sequence
from scripts.external_baselines import FrozenAngular,geometric

OUT=Path('results/external_benchmark_20261107');RUN=Path('artifacts/runs/external_benchmark_20261107')

def main():
    cfg=config();torch.set_num_threads(cfg['threads']);dest=OUT/'posthoc_diagnostics.json'
    if dest.exists():raise FileExistsError('Retain earlier diagnostics')
    dt=[]
    for sid in cfg['train_ids']:
        s=load_sequence(cfg,sid,cfg['train_ids']);dt.append(float(np.diff(s['time']).mean()))
    selection=json.loads((RUN/'selection.json').read_text());omega=selection['oscillator_parameters'][0];ratio=float(np.mean(dt)/cfg['dt_s'])
    p=dict(theta=np.zeros((1,9)),x=np.zeros((1,9)),past_velocity=np.zeros((1,9)),past_acceleration=np.zeros((1,9)),future_velocity=np.zeros((1,30)),future_acceleration=np.zeros((1,30)))
    curves={'geometric':geometric(p,cfg)}
    for profile in ['retained','candidate']:curves['cranepilot_'+profile]=FrozenAngular(profile).forecast(p,cfg)
    rows=[]
    for method,y in curves.items():
        for j,value in enumerate(y[0]):rows.append(dict(method=method,nominal_horizon_s=(j+1)*.05,predicted_angle_deg=float(np.rad2deg(value))))
    pd.DataFrame(rows).to_csv(OUT/'posthoc_equilibrium.csv',index=False)
    info=dict(posthoc=True,no_controller_or_weight_changes=True,source_sha256=sha256(Path(__file__)),train_mean_camera_interval_s=float(np.mean(dt)),train_per_record_mean_camera_interval_s=dt,
        clock_ratio_actual_over_nominal=ratio,fitted_omega_per_nominal_second=omega,clock_rescaled_fitted_omega_rad_per_actual_second=omega/ratio,geometric_omega_rad_per_second=float(np.sqrt(9.81)),
        inference='Clock mismatch may explain part of fitted frequency shift. This is a diagnostic inference, not a new evaluated correction.',
        zero_packet_peak_deg={m:float(np.abs(np.rad2deg(y)).max()) for m,y in curves.items()},zero_packet_interpretation='Synthetic mathematical equilibrium input, not an observed test/hardware performance metric. A learned bias on this packet identifies no physical failure mechanism by itself.')
    dest.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))

if __name__=='__main__':main()
