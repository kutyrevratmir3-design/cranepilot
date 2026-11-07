"""Full warmed batch-one decision timing, including causal identifier updates.

Reconstruct identifier state from ONLY prior measured positions/applied commands,
then restore that state before every repetition. No fitted ground truth is read.
"""
import copy,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.structural_transfer import TransferModel as AdaptiveModel,prepare_history,solve,target_at
from cranepilot.data import sha256


def window(r,case,tick,cfg):
    end=40+tick*5;start=max(0,end-cfg['identification']['history_intervals']*5)
    pos=r['raw_measurements'][case:case+1,np.arange(start,end+1,5),1].copy()
    c=np.concatenate([np.zeros((1,8)),r['commands'][case:case+1,:tick]],1)[:,start//5:end//5].copy()
    return pos,c


def main():
    run=Path('artifacts/runs/structural_transfer_20261107');out=Path('results/structural_transfer_20261107')
    if (out/'runtime_batch1.csv').exists():raise FileExistsError('Preserve timings')
    p=json.loads((run/'provenance.json').read_text());cfg=p['config'];torch.set_num_threads(1)
    if p['status']!='complete':raise ValueError('Completed run')
    frame=pd.read_csv(run/'evaluate/cases.csv');rows=[]
    for method in cfg['methods']:
        with np.load(run/'evaluate'/f'{method}.npz') as z:r={key:z[key] for key in z.files}
        for case,tick in [(0,0),(31,40),(95,100)]:
            model=AdaptiveModel(method,cfg)
            for j in range(tick):model.update(*window(r,case,j,cfg),j)
            saved={key:copy.deepcopy(getattr(model,key)) for key in ['tau','gain','ever_valid','observed','observed_valid']}
            k=40+tick*5;measured=r['raw_measurements'][case:case+1,k-np.arange(41)].copy()
            warm=torch.zeros(1,6) if tick==0 else torch.as_tensor(np.concatenate([r['plans'][case:case+1,tick-1,2:],r['plans'][case:case+1,tick-1,-1:]],1)[:,::5].copy())
            committed=torch.as_tensor(r['commands'][case:case+1,tick].copy());target=torch.tensor([target_at(tick*.05,cfg)])
            for repeat in range(-1,5):
                for key,value in saved.items():setattr(model,key,copy.deepcopy(value))
                started=time.perf_counter();model.update(*window(r,case,tick,cfg),tick)
                b,i=prepare_history(measured[:,:,0].copy(),measured[:,:,1].copy(),frame.length.to_numpy()[case:case+1].copy(),model.calibration)
                plan,_,_,_=solve(model,b,i,committed,warm,target,cfg['solver'])
                elapsed=(time.perf_counter()-started)*1000;difference=float(np.abs(plan.numpy()-r['plans'][case:case+1,tick]).max())
                if repeat>=0:rows.append(dict(method=method,case=case,tick=tick,repeat=repeat,elapsed_ms=elapsed,plan_max_difference=difference,over_50ms=elapsed>50))
    pd.DataFrame(rows).to_csv(out/'runtime_batch1.csv',index=False)
    (out/'runtime_batch1.json').write_text(json.dumps(dict(source_sha256=sha256(Path(__file__)),measurements=len(rows),includes='past window assembly, identifier update, observer and full solver',excludes='model loading, I/O, scheduling, reconstruction of pre-tick estimator state',same_pre_tick_state_restored_for_every_repeat=True,real_time_guarantee=False),indent=2)+'\n')
    print(pd.DataFrame(rows).groupby('method').elapsed_ms.agg(['median','max']).to_string())

if __name__=='__main__':main()
