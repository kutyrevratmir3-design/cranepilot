"""Isolated batch-one complete decision timing on fixed saved evaluation histories."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.scenario_control import ScenarioModel as AngularModel,prepare_history,solve
from cranepilot.data import sha256


def main():
    run=Path('artifacts/runs/scenario_control_20261107');out=Path('results/scenario_control_20261107')
    if (out/'runtime_batch1.csv').exists():raise FileExistsError('Preserve timings')
    p=json.loads((run/'provenance.json').read_text());cfg=p['config'];torch.set_num_threads(1)
    if p['status']!='complete':raise ValueError('Completed run')
    frame=pd.read_csv(run/'evaluate/cases.csv');rows=[]
    for method in cfg['methods']:
        model=AngularModel(method)
        with np.load(run/'evaluate'/f'{method}.npz') as z:r={key:z[key] for key in z.files}
        for case,tick in [(0,0),(31,40),(95,100)]:
            k=40+tick*5;measured=r['raw_measurements'][case:case+1,k-np.arange(41)].copy()
            warm=torch.zeros(1,6) if tick==0 else torch.as_tensor(np.concatenate([r['plans'][case:case+1,tick-1,2:],r['plans'][case:case+1,tick-1,-1:]],1)[:,::5].copy())
            committed=torch.as_tensor(r['commands'][case:case+1,tick].copy());target=torch.tensor([cfg['target_x_m']])
            for repeat in range(-1,5):
                started=time.perf_counter()
                b,i=prepare_history(measured[:,:,0].copy(),measured[:,:,1].copy(),frame.length.to_numpy()[case:case+1],model.calibration)
                plan,_,_,_=solve(model,b,i,committed,warm,target,cfg['solver'])
                elapsed=(time.perf_counter()-started)*1000
                difference=float(np.abs(plan.numpy()-r['plans'][case:case+1,tick]).max())
                if repeat>=0:rows.append(dict(method=method,case=case,tick=tick,repeat=repeat,elapsed_ms=elapsed,plan_max_difference=difference,over_50ms=elapsed>50))
    pd.DataFrame(rows).to_csv(out/'runtime_batch1.csv',index=False)
    (out/'runtime_batch1.json').write_text(json.dumps(dict(source_sha256=sha256(Path(__file__)),measurements=len(rows),includes='history, observer, all model seeds, FD, QP, nonlinear backtracking',excludes='model loading, I/O, scheduling',real_time_guarantee=False),indent=2)+'\n')
    print(pd.DataFrame(rows).groupby('method').elapsed_ms.agg(['median','max']).to_string())

if __name__=='__main__':main()
