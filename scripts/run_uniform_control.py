"""Matched training-protocol control: unchanged PEPM architecture, new sampler/loss."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import torch
from cranepilot.data import load_dataset,sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import (command_windows,stack_windows,tensors,NeuralActuator,
    PassiveForecaster,train_module,HORIZON_INDEX,HORIZONS_S)
from cranepilot.metrics import regression_metrics
from cranepilot.benchmark import environment_record


def main():
    run=Path('artifacts/runs/uniform_passive_control')
    if run.exists():raise FileExistsError('Retain previous evidence')
    run.mkdir(parents=True)
    cfg=json.loads(Path('configs/innovation_development.json').read_text())
    scenarios=load_dataset(Path('data/processed'))
    cal=LengthCalibration.fit(scenarios,cfg['train'],'theta')
    actor_path=Path('artifacts/runs/actuator_development/actuator.pt')
    actor=NeuralActuator();actor.load_state_dict(torch.load(actor_path,weights_only=True));actor.eval()
    torch.set_num_threads(4)
    provenance={'status':'running','config':cfg,'architecture':'unchanged PassiveForecaster(memory_dim=4)',
        'environment':environment_record(),'script_sha256':sha256(Path(__file__)),
        'calibration':asdict(cal),'actuator_sha256':sha256(actor_path)}
    def save(): (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    save()
    def window(sid,stride):return command_windows(scenarios[sid],cal,np.arange(40,len(scenarios[sid].time)-150,stride))
    def prepare(parts):
        b=tensors(stack_windows(parts))
        with torch.no_grad():b['acceleration']=actor(b)[1]
        return b
    tr=prepare([window(s,cfg['train_stride']) for s in cfg['train']])
    va=prepare([window(s,cfg['evaluation_stride']) for s in cfg['validation']])
    models=[];logs=[]
    for seed in cfg['seeds']:
        torch.manual_seed(seed);model=PassiveForecaster(memory_dim=4)
        def loss(module,b):
            y,d=module(b,b['acceleration']);error=(y-b['target_q'])/.1
            if not module.training:return error[:,HORIZON_INDEX].abs().mean()
            return error.square().mean()+.01*d['initial_memory'].square().mean()
        print(f'CONTROL seed={seed}',flush=True)
        log=train_module(model,tr,va,loss,cfg['training'],seed)
        logs.extend({'seed':seed,**r} for r in log);models.append(model)
        torch.save(model.state_dict(),run/f'control_seed{seed}.pt')
        pd.DataFrame(logs).to_csv(run/'training.csv',index=False)
    provenance['status']='validation_complete';save()
    metrics=[]
    for group,sids in [('validation',cfg['validation']),('evaluation',cfg['evaluation'])]:
        for sid in sids:
            w=window(sid,cfg['evaluation_stride']);b=prepare([w])
            with torch.no_grad():y=np.mean([m(b,b['acceleration'])[0].numpy()[:,HORIZON_INDEX] for m in models],axis=0)
            for j,h in enumerate(HORIZONS_S):
                for regime,mask in [('all',np.ones(len(y),bool)),('active_at_origin',abs(w['initial_velocity'])>.005),('quiet_at_origin',abs(w['initial_velocity'])<=.005)]:
                    if mask.any():metrics.append({'group':group,'scenario':sid,'model':'uniform_control','horizon_s':h,'regime':regime,**regression_metrics(w['target_q'][mask,HORIZON_INDEX[j]],y[mask,j])})
            np.savez_compressed(run/f'predictions_s{sid}.npz',k=w['k'],target=w['target_q'][:,HORIZON_INDEX],uniform_control=y)
    pd.DataFrame(metrics).to_csv(run/'evaluation_metrics.csv',index=False)
    provenance['status']='complete';save()
    print(pd.DataFrame(metrics).query("regime == 'all'").groupby(['group','horizon_s']).mae_deg.mean().unstack().round(5).to_string())


if __name__=='__main__':main()
