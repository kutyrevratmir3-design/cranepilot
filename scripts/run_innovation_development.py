"""Validation-first development; evaluation requires an already completed fit."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import numpy as np
import pandas as pd
import torch
from cranepilot.data import load_dataset, sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import (command_windows, stack_windows, tensors,
    NeuralActuator, train_module, HORIZON_INDEX, HORIZONS_S)
from cranepilot.innovation import InnovationForecaster
from cranepilot.benchmark import environment_record
from cranepilot.metrics import regression_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('artifacts/runs/innovation_development'))
    p.add_argument('--evaluate',action='store_true')
    args=p.parse_args()
    cfg_path=Path('configs/innovation_development.json')
    cfg=json.loads(cfg_path.read_text())
    run=args.output
    if not args.evaluate:
        if run.exists(): raise FileExistsError('Retain earlier runs; use a fresh output')
        run.mkdir(parents=True)
    provenance_path=run/'provenance.json'
    if args.evaluate:
        retained=json.loads(provenance_path.read_text())
        if retained['status']!='validation_complete': raise ValueError('Completed validation required')
        if retained['config_sha256']!=sha256(cfg_path): raise ValueError('Configuration changed')
        if (run/'evaluation_metrics.csv').exists(): raise FileExistsError('Evaluation already retained')
        for path,expected in retained['code_sha256'].items():
            if sha256(Path(path))!=expected: raise ValueError('Source changed after validation')
    scenarios=load_dataset(Path('data/processed'))
    calibration=LengthCalibration.fit(scenarios,cfg['train'],'theta')
    actor_path=Path('artifacts/runs/actuator_development/actuator.pt')
    actor=NeuralActuator();actor.load_state_dict(torch.load(actor_path,weights_only=True));actor.eval()
    torch.set_num_threads(4)
    def windows(sid,stride):
        return command_windows(scenarios[sid],calibration,np.arange(40,len(scenarios[sid].time)-150,stride))
    def prepare(parts):
        b=tensors(stack_windows(parts))
        with torch.no_grad(): b['acceleration']=actor(b)[1]
        return b
    models={}
    if not args.evaluate:
        provenance={'status':'running','config':cfg,'config_sha256':sha256(cfg_path),
            'environment':environment_record(),'calibration':asdict(calibration),
            'actuator_sha256':sha256(actor_path),
            'manifest_sha256':sha256(Path('data/processed/manifest.json')),
            'code_sha256':{str(x):sha256(x) for x in
                [Path(__file__).relative_to(Path.cwd()),Path('src/cranepilot/innovation.py'),Path('src/cranepilot/rollout.py'),Path('src/cranepilot/passive_memory.py')]}}
        provenance_path.write_text(json.dumps(provenance,indent=2))
        tr=prepare([windows(s,cfg['train_stride']) for s in cfg['train']])
        va=prepare([windows(s,cfg['evaluation_stride']) for s in cfg['validation']])
        logs=[]; rows=[]
        for variant in cfg['variants']:
            for seed in cfg['seeds']:
                torch.manual_seed(seed);model=InnovationForecaster(driven=variant=='driven')
                def loss(module,b):
                    y,d=module(b,b['acceleration'])
                    error=(y-b['target_q'])/.1
                    if not module.training:
                        return error[:,HORIZON_INDEX].abs().mean()
                    return error.square().mean()+.01*d['initial_memory'].square().mean()+.01*d['innovation_force'].square().mean()
                print(f'FIT {variant} seed={seed}',flush=True)
                log=train_module(model,tr,va,loss,cfg['training'],seed)
                logs.extend({'variant':variant,'seed':seed,**r} for r in log)
                torch.save(model.state_dict(),run/f'{variant}_seed{seed}.pt')
                with torch.no_grad(): pred,d=model(va,va['acceleration'])
                rows.append({'variant':variant,'seed':seed,'validation_mae_deg':float(torch.rad2deg(pred[:,HORIZON_INDEX]-va['target_q'][:,HORIZON_INDEX]).abs().mean()),
                    'parameters':sum(x.numel() for x in model.parameters()),
                    'frequency_ratio_min':float(d['frequency_ratio'].min()),'frequency_ratio_max':float(d['frequency_ratio'].max())})
                pd.DataFrame(logs).to_csv(run/'training.csv',index=False)
                pd.DataFrame(rows).to_csv(run/'validation_seeds.csv',index=False)
        provenance['status']='validation_complete'
        provenance_path.write_text(json.dumps(provenance,indent=2))
        print(pd.DataFrame(rows).to_string(index=False),flush=True)
        return
    if retained['actuator_sha256']!=sha256(actor_path):raise ValueError('Actuator changed')
    for variant in cfg['variants']:
        for seed in cfg['seeds']:
            model=InnovationForecaster(driven=variant=='driven')
            model.load_state_dict(torch.load(run/f'{variant}_seed{seed}.pt',weights_only=True));model.eval()
            models[variant,seed]=model
    rows=[]; diagnostics=[]
    for group,sids in [('validation',cfg['validation']),('evaluation',cfg['evaluation'])]:
        for sid in sids:
            w=windows(sid,cfg['evaluation_stride']); b=prepare([w]); preds={}
            with torch.no_grad():
                for variant in cfg['variants']:
                    ys=[]
                    for seed in cfg['seeds']:
                        y,d=models[variant,seed](b,b['acceleration']);ys.append(y.numpy()[:,HORIZON_INDEX])
                        diagnostics.append({'scenario':sid,'variant':variant,'seed':seed,
                            'balance_max':float(d['balance'].abs().max()),
                            'innovation_rms':float(d['innovation_force'].square().mean().sqrt())})
                    preds[variant]=np.mean(ys,axis=0)
            for variant,y in preds.items():
                for j,h in enumerate(HORIZONS_S):
                    for regime,mask in [('all',np.ones(len(y),bool)),('active_at_origin',abs(w['initial_velocity'])>.005),('quiet_at_origin',abs(w['initial_velocity'])<=.005)]:
                        if mask.any():rows.append({'group':group,'scenario':sid,'model':variant,'horizon_s':h,'regime':regime,**regression_metrics(w['target_q'][mask,HORIZON_INDEX[j]],y[mask,j])})
            np.savez_compressed(run/f'predictions_s{sid}.npz',k=w['k'],target=w['target_q'][:,HORIZON_INDEX],**preds)
    pd.DataFrame(rows).to_csv(run/'evaluation_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(run/'diagnostics.csv',index=False)
    retained['status']='complete';provenance_path.write_text(json.dumps(retained,indent=2))
    print(pd.DataFrame(rows).query("regime == 'all'").groupby(['group','model','horizon_s']).mae_deg.mean().unstack().round(5).to_string())


if __name__=='__main__': main()
