"""Frozen causal observer/robust-loss campaign with stronger kernel references."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern,WhiteKernel
from threadpoolctl import threadpool_limits
from cranepilot.data import load_dataset,sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import (command_windows,stack_windows,tensors,NeuralActuator,
    train_module,HORIZON_INDEX,HORIZONS_S)
from cranepilot.innovation import InnovationForecaster
from cranepilot.forced_observer import refine_windows,analytic_rollout,phase_kernel_features
from cranepilot.benchmark import environment_record
from cranepilot.metrics import regression_metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('artifacts/runs/forceaware_campaign'))
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args();run=args.output
    cfg_path=Path('configs/forceaware_campaign.json');cfg=json.loads(cfg_path.read_text())
    actor_path=Path('artifacts/runs/actuator_development/actuator.pt')
    guarded=[cfg_path,actor_path,Path('data/processed/dataset.npz'),Path('data/processed/manifest.json'),
             Path(__file__).relative_to(Path.cwd()),*sorted(Path('src/cranepilot').glob('*.py'))]
    hashes={str(p):sha256(p) for p in guarded}
    if not args.evaluate:
        if run.exists():raise FileExistsError('Choose fresh output; preserve retained runs')
        run.mkdir(parents=True)
        provenance={'status':'running','config':cfg,'guarded_hashes':hashes,
                    'environment':environment_record(),'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    else:
        provenance=json.loads((run/'provenance.json').read_text())
        if provenance['status']!='validation_complete':raise ValueError('Validation must finish first')
        if provenance['guarded_hashes']!=hashes:raise ValueError('Code, data, configuration or actor changed')
        for p,h in provenance['checkpoint_hashes'].items():
            if sha256(run/p)!=h:raise ValueError('Trained checkpoint changed')
    def save(): (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if not args.evaluate:save()
    scenarios=load_dataset(Path('data/processed'))
    if args.evaluate:
        calibration=LengthCalibration(**provenance['calibration'])
    else:
        calibration=LengthCalibration.fit(scenarios,cfg['train'],'theta')
        provenance['calibration']=asdict(calibration);save()
    actor=NeuralActuator();actor.load_state_dict(torch.load(actor_path,weights_only=True));actor.eval()
    torch.set_num_threads(4)
    def prepare(sids,stride):
        raw=stack_windows([command_windows(scenarios[s],calibration,
            np.arange(40,len(scenarios[s].time)-150,stride)) for s in sids])
        refined=refine_windows(raw)
        with torch.no_grad():positions,acceleration=actor(tensors(raw))
        acc=acceleration.numpy()
        base=analytic_rollout(refined['initial_q'],refined['initial_p'],raw['omega'],acc)[:,HORIZON_INDEX]
        rawbase=analytic_rollout(raw['initial_q'],raw['initial_p'],raw['omega'],acc)[:,HORIZON_INDEX]
        full=np.column_stack([raw['angular_context'],raw['omega'],raw['initial_x'],raw['initial_velocity'],
            raw['future_u'],acc,positions.numpy()-raw['initial_x'][:,None]])
        compact=phase_kernel_features(raw,refined,acc)
        batches={}
        for key,w in [('raw',raw),('forced',refined)]:
            batches[key]=tensors(w);batches[key]['acceleration']=acceleration
        return raw,base,rawbase,{'full_kernel':full,'phase_kernel':compact},batches
    models={};kernels={}
    if not args.evaluate:
        tr=prepare(cfg['train'],cfg['train_stride']);va=prepare(cfg['validation'],cfg['evaluation_stride'])
        logs=[];scores=[]
        for name,choice in cfg['variants'].items():
            key='forced' if choice['forced_observer'] else 'raw'
            for seed in cfg['seeds']:
                torch.manual_seed(seed);model=InnovationForecaster(driven=choice['driven'])
                def loss(module,b):
                    y,d=module(b,b['acceleration']);error=(y-b['target_q'])/.1
                    if not module.training:return error[:,HORIZON_INDEX].abs().mean()
                    fit=error.square().mean() if choice['loss']=='mse' else F.smooth_l1_loss(error,torch.zeros_like(error),beta=cfg['smooth_l1_beta'])
                    return fit+.01*d['initial_memory'].square().mean()+.01*d['innovation_force'].square().mean()
                print(f'FIT {name} seed={seed}',flush=True)
                log=train_module(model,tr[4][key],va[4][key],loss,cfg['training'],seed)
                logs.extend({'model':name,'seed':seed,**r} for r in log)
                torch.save(model.state_dict(),run/f'{name}_seed{seed}.pt')
                with torch.no_grad():y,_=model(va[4][key],va[4][key]['acceleration'])
                score=float(torch.rad2deg(y[:,HORIZON_INDEX]-va[4][key]['target_q'][:,HORIZON_INDEX]).abs().mean())
                scores.append({'model':name,'seed':seed,'validation_mae_deg':score,'parameters':sum(p.numel() for p in model.parameters())})
                pd.DataFrame(logs).to_csv(run/'training.csv',index=False)
                pd.DataFrame(scores).to_csv(run/'validation_seeds.csv',index=False)
        target=tr[0]['target_q'][:,HORIZON_INDEX];vtarget=va[0]['target_q'][:,HORIZON_INDEX]
        kernel_scores=[]
        with threadpool_limits(limits=4):
            for name,xt in tr[3].items():
                xv=va[3][name];scaler=StandardScaler().fit(xt)
                base_index=2 if name=='full_kernel' else 1
                residual=target-tr[base_index]
                selected=np.linspace(0,len(xt)-1,min(len(xt),cfg['kernel']['max_points'])).astype(int)
                best=None
                for family in cfg['kernel']['families']:
                    for length in cfg['kernel']['length_scales']:
                        kernel=RBF(length) if family=='rbf' else Matern(length,nu=1.5)
                        gp=GaussianProcessRegressor(kernel=kernel+WhiteKernel(cfg['kernel']['noise']),optimizer=None,normalize_y=True,alpha=1e-6)
                        gp.fit(scaler.transform(xt[selected]),residual[selected])
                        prediction=gp.predict(scaler.transform(xv))+va[base_index]
                        score=float(np.mean(abs(np.rad2deg(prediction-vtarget))))
                        kernel_scores.append({'model':name,'family':family,'length':length,'validation_mae_deg':score,'training_points':len(selected)})
                        print('KERNEL',kernel_scores[-1],flush=True)
                        if best is None or score<best[0]:best=(score,gp,family,length)
                joblib.dump({'gp':best[1],'scaler':scaler,'base_index':base_index},run/f'{name}.joblib')
        pd.DataFrame(kernel_scores).to_csv(run/'kernel_validation.csv',index=False)
        provenance['checkpoint_hashes']={p.name:sha256(p) for p in run.iterdir() if p.suffix in ['.pt','.joblib']}
        provenance['status']='validation_complete';save()
        print(pd.DataFrame(scores).to_string(index=False),flush=True)
        return
    for name,choice in cfg['variants'].items():
        for seed in cfg['seeds']:
            model=InnovationForecaster(driven=choice['driven'])
            model.load_state_dict(torch.load(run/f'{name}_seed{seed}.pt',weights_only=True));model.eval()
            models[name,seed]=model
    kernels={name:joblib.load(run/f'{name}.joblib') for name in ['full_kernel','phase_kernel']}
    metrics=[];diagnostics=[];timing=[]
    for group,sids in [('validation',cfg['validation']),('evaluation',cfg['evaluation']),('load_stress',cfg['load_stress'])]:
        for sid in sids:
            pack=prepare([sid],cfg['evaluation_stride']);raw,base,rawbase,features,batches=pack
            predictions={'forced_physics':base,'action_physics':rawbase}
            with torch.no_grad():
                for name,choice in cfg['variants'].items():
                    batch=batches['forced' if choice['forced_observer'] else 'raw'];ys=[]
                    for seed in cfg['seeds']:
                        start=time.perf_counter();y,d=models[name,seed](batch,batch['acceleration'])
                        timing.append({'scenario':sid,'model':name,'seed':seed,'seconds':time.perf_counter()-start,'n':len(y)})
                        ys.append(y.numpy()[:,HORIZON_INDEX])
                        diagnostics.append({'scenario':sid,'model':name,'seed':seed,'balance_max':float(d['balance'].abs().max()),
                            'frequency_min':float(d['frequency_ratio'].min()),'frequency_max':float(d['frequency_ratio'].max())})
                    predictions[name]=np.mean(ys,axis=0)
            with threadpool_limits(limits=4):
                for name,kernel in kernels.items():
                    predictions[name]=kernel['gp'].predict(kernel['scaler'].transform(features[name]))+pack[kernel['base_index']]
            for name,y in predictions.items():
                for j,h in enumerate(HORIZONS_S):
                    for regime,mask in [('all',np.ones(len(y),bool)),('active_at_origin',abs(raw['initial_velocity'])>.005),('quiet_at_origin',abs(raw['initial_velocity'])<=.005)]:
                        if mask.any():metrics.append({'group':group,'scenario':sid,'model':name,'horizon_s':h,'regime':regime,
                            **regression_metrics(raw['target_q'][mask,HORIZON_INDEX[j]],y[mask,j])})
            np.savez_compressed(run/f'predictions_s{sid}.npz',k=raw['k'],target=raw['target_q'][:,HORIZON_INDEX],**predictions)
            pd.DataFrame(metrics).to_csv(run/'metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(run/'diagnostics.csv',index=False)
    pd.DataFrame(timing).to_csv(run/'timing.csv',index=False)
    provenance['status']='complete';provenance['finished_utc']=pd.Timestamp.now(tz='UTC').isoformat();save()
    print(pd.DataFrame(metrics).query("regime == 'all'").groupby(['group','model','horizon_s']).mae_deg.mean().unstack().round(5).to_string())


if __name__=='__main__':main()
