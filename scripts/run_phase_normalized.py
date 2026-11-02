"""Phase-similarity angular model, separate validation and guarded evaluation."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from cranepilot.data import load_dataset,sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows,stack_windows,tensors,NeuralActuator,train_module,HORIZON_INDEX,HORIZONS_S
from cranepilot.forced_observer import refine_windows,analytic_rollout
from cranepilot.phase_normalized import PhaseNormalizedForecaster,GRUResidualForecaster,phase_context
from cranepilot.innovation import InnovationForecaster
from cranepilot.benchmark import environment_record
from cranepilot.metrics import regression_metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('artifacts/runs/phase_normalized_campaign'))
    parser.add_argument('--config',type=Path,default=Path('configs/phase_normalized_campaign.json'))
    args=parser.parse_args();run=args.output
    cfg_path=args.config;cfg=json.loads(cfg_path.read_text())
    if cfg['phase_points']!=25:raise ValueError('Model requires exactly 25 phase points')
    actor_path=Path('artifacts/runs/actuator_development/actuator.pt')
    guarded=[cfg_path,actor_path,Path('data/processed/dataset.npz'),Path('data/processed/manifest.json'),
             Path(__file__).relative_to(Path.cwd()),*sorted(Path('src/cranepilot').glob('*.py'))]
    hashes={str(p):sha256(p) for p in guarded}
    if not args.evaluate:
        if run.exists():raise FileExistsError('Output exists; preserve earlier evidence')
        run.mkdir(parents=True);provenance={'status':'running','config':cfg,'guarded_hashes':hashes,
            'environment':environment_record(),'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    else:
        provenance=json.loads((run/'provenance.json').read_text())
        if provenance['status']!='validation_complete' or provenance['guarded_hashes']!=hashes:
            raise ValueError('Completed unchanged validation required')
        for p,h in provenance['checkpoint_hashes'].items():
            if sha256(run/p)!=h:raise ValueError('Checkpoint changed')
    def save(): (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if not args.evaluate:save()
    scenarios=load_dataset(Path('data/processed'))
    cal=LengthCalibration(**provenance['calibration']) if args.evaluate else LengthCalibration.fit(scenarios,cfg['train'],'theta')
    if not args.evaluate:provenance['calibration']=asdict(cal);save()
    actor=NeuralActuator();actor.load_state_dict(torch.load(actor_path,weights_only=True));actor.eval()
    torch.set_num_threads(4)
    def make_model(choice):
        if choice=='anchor':return InnovationForecaster(driven=True,measurement_anchor=True)
        return GRUResidualForecaster() if choice=='gru' else PhaseNormalizedForecaster(choice)
    def prepare(sids,stride):
        raw=stack_windows([command_windows(scenarios[s],cal,np.arange(40,len(scenarios[s].time)-150,stride)) for s in sids])
        refined=refine_windows(raw)
        refined['phase_context']=phase_context(raw,refined,span=cfg['phase_span_rad'],points=cfg['phase_points'])
        refined['reference_context']=np.column_stack([raw['angular_context'],raw['omega'],raw['initial_x'],raw['initial_velocity']])
        batch=tensors(refined)
        with torch.no_grad():batch['acceleration']=actor(tensors(raw))[1]
        batch['physical_base']=torch.as_tensor(analytic_rollout(refined['initial_q'],refined['initial_p'],raw['omega'],batch['acceleration'].numpy()),dtype=torch.float32)
        return raw,batch
    models={}
    if not args.evaluate:
        _,tr=prepare(cfg['train'],cfg['train_stride']);_,va=prepare(cfg['validation'],cfg['evaluation_stride'])
        logs=[];scores=[]
        for name,driven in cfg['variants'].items():
            for seed in cfg['seeds']:
                torch.manual_seed(seed);model=make_model(driven)
                def loss(module,b):
                    y,d=module(b,b['acceleration']);error=(y-b['target_q'])/.1
                    if not module.training:return error[:,HORIZON_INDEX].abs().mean()
                    fit=F.smooth_l1_loss(error,torch.zeros_like(error),beta=cfg['smooth_l1_beta'])
                    return fit+.01*d['initial_memory'].square().mean()+.01*d['innovation_force'].square().mean() if d else fit
                print(f'FIT {name} seed={seed}',flush=True)
                log=train_module(model,tr,va,loss,cfg['training'],seed)
                logs.extend({'model':name,'seed':seed,**r} for r in log)
                torch.save(model.state_dict(),run/f'{name}_seed{seed}.pt')
                with torch.no_grad():y,_=model(va,va['acceleration'])
                scores.append({'model':name,'seed':seed,'validation_mae_deg':float(torch.rad2deg(y[:,HORIZON_INDEX]-va['target_q'][:,HORIZON_INDEX]).abs().mean()),'parameters':sum(p.numel() for p in model.parameters())})
                pd.DataFrame(logs).to_csv(run/'training.csv',index=False);pd.DataFrame(scores).to_csv(run/'validation_seeds.csv',index=False)
        provenance['checkpoint_hashes']={p.name:sha256(p) for p in run.glob('*.pt')}
        provenance['status']='validation_complete';save();print(pd.DataFrame(scores).to_string(index=False));return
    for name,driven in cfg['variants'].items():
        for seed in cfg['seeds']:
            model=make_model(driven);model.load_state_dict(torch.load(run/f'{name}_seed{seed}.pt',weights_only=True));model.eval();models[name,seed]=model
    metrics=[];diagnostics=[]
    for group,sids in [('validation',cfg['validation']),('evaluation',cfg['evaluation']),('load_stress',cfg['load_stress'])]:
        for sid in sids:
            raw,b=prepare([sid],cfg['evaluation_stride']);predictions={}
            with torch.no_grad():
                for name in cfg['variants']:
                    ys=[]
                    for seed in cfg['seeds']:
                        y,d=models[name,seed](b,b['acceleration']);ys.append(y.numpy()[:,HORIZON_INDEX])
                        diagnostics.append({'scenario':sid,'model':name,'seed':seed,'balance_max':float(d['balance'].abs().max()) if d else None})
                    predictions[name]=np.mean(ys,axis=0)
            for name,y in predictions.items():
                for j,h in enumerate(HORIZONS_S):
                    for regime,mask in [('all',np.ones(len(y),bool)),('active_at_origin',abs(raw['initial_velocity'])>.005),('quiet_at_origin',abs(raw['initial_velocity'])<=.005)]:
                        if mask.any():metrics.append({'group':group,'scenario':sid,'model':name,'horizon_s':h,'regime':regime,**regression_metrics(raw['target_q'][mask,HORIZON_INDEX[j]],y[mask,j])})
            np.savez_compressed(run/f'predictions_s{sid}.npz',k=raw['k'],target=raw['target_q'][:,HORIZON_INDEX],**predictions)
            pd.DataFrame(metrics).to_csv(run/'metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(run/'diagnostics.csv',index=False)
    provenance['status']='complete';provenance['finished_utc']=pd.Timestamp.now(tz='UTC').isoformat();save()
    print(pd.DataFrame(metrics).query("regime == 'all'").groupby(['group','model','horizon_s']).mae_deg.mean().unstack().round(5).to_string())


if __name__=='__main__':main()
