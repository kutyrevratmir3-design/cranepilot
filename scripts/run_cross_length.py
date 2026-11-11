"""Additional length-range tests, with per-fold actors and calibration."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
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
from cranepilot.rollout import command_windows,stack_windows,tensors,NeuralActuator,train_module,HORIZON_INDEX,HORIZONS_S
from cranepilot.forced_observer import refine_windows,analytic_rollout
from cranepilot.innovation import InnovationForecaster
from cranepilot.phase_normalized import GRUResidualForecaster
from cranepilot.benchmark import environment_record
from cranepilot.metrics import regression_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true')
    p.add_argument('--output',type=Path,default=Path('artifacts/runs/cross_length_campaign'))
    args=p.parse_args();run=args.output;cfg_path=Path('configs/cross_length_campaign.json');cfg=json.loads(cfg_path.read_text())
    guarded=[cfg_path,Path('data/processed/dataset.npz'),Path('data/processed/manifest.json'),
             Path(__file__).relative_to(Path.cwd()),*sorted(Path('src/cranepilot').glob('*.py'))]
    hashes={str(x):sha256(x) for x in guarded}
    if not args.evaluate:
        if run.exists():raise FileExistsError('Preserve previous runs')
        run.mkdir(parents=True);provenance={'status':'running','config':cfg,'guarded_hashes':hashes,
            'environment':environment_record(),'folds':{},'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    else:
        provenance=json.loads((run/'provenance.json').read_text())
        if provenance['status']!='validation_complete' or provenance['guarded_hashes']!=hashes:
            raise ValueError('Unchanged completed validation required')
        for name,h in provenance['checkpoint_hashes'].items():
            if sha256(run/name)!=h:raise ValueError('Checkpoint changed')
    def save():(run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if not args.evaluate:save()
    scenarios=load_dataset(Path('data/processed'));torch.set_num_threads(4)
    def model_for(name):return InnovationForecaster(True) if name=='forced_driven_huber' else GRUResidualForecaster()
    scores=[];logs=[];kernel_scores=[];metrics=[]
    for fold,split in cfg['folds'].items():
        directory=run/fold
        if not args.evaluate:directory.mkdir()
        cal=LengthCalibration(**provenance['folds'][fold]['calibration']) if args.evaluate else LengthCalibration.fit(scenarios,split['train'],'theta')
        def raw_windows(sids,stride):
            return stack_windows([command_windows(scenarios[s],cal,np.arange(40,len(scenarios[s].time)-150,stride)) for s in sids])
        torch.manual_seed(0);actor=NeuralActuator()
        if not args.evaluate:
            tr=tensors(raw_windows(split['train'],cfg['train_stride']));va=tensors(raw_windows(split['validation'],cfg['evaluation_stride']))
            def actor_loss(m,b):
                x,a=m(b);return ((x-b['target_x'])/.05).square().mean()+1e-5*a.square().mean()
            print('ACTOR',fold,flush=True)
            actor_log=train_module(actor,tr,va,actor_loss,cfg['actuator_training'],seed=0)
            pd.DataFrame(actor_log).to_csv(directory/'actor_training.csv',index=False);torch.save(actor.state_dict(),directory/'actor.pt')
            provenance['folds'][fold]={'calibration':asdict(cal),'train_lengths_m':[float(scenarios[s].signals['L'][0]) for s in split['train']]};save()
        else:actor.load_state_dict(torch.load(directory/'actor.pt',weights_only=True));actor.eval()
        def prepare(sids,stride):
            raw=raw_windows(sids,stride);w=refine_windows(raw)
            with torch.no_grad():position,acceleration=actor(tensors(raw))
            base=analytic_rollout(w['initial_q'],w['initial_p'],w['omega'],acceleration.numpy())
            w['reference_context']=np.column_stack([raw['angular_context'],raw['omega'],raw['initial_x'],raw['initial_velocity']])
            w['physical_base']=base;b=tensors(w);b['acceleration']=acceleration
            features=np.column_stack([raw['angular_context'],raw['omega'],raw['initial_x'],raw['initial_velocity'],raw['future_u'],acceleration.numpy(),position.numpy()-raw['initial_x'][:,None]])
            return raw,b,base[:,HORIZON_INDEX],features
        if not args.evaluate:
            tr=prepare(split['train'],cfg['train_stride']);va=prepare(split['validation'],cfg['evaluation_stride'])
            for name in cfg['models']:
                for seed in cfg['seeds']:
                    torch.manual_seed(seed);model=model_for(name)
                    def loss(module,b):
                        y,d=module(b,b['acceleration']);error=(y-b['target_q'])/.1
                        if not module.training:return error[:,HORIZON_INDEX].abs().mean()
                        fit=F.smooth_l1_loss(error,torch.zeros_like(error),beta=cfg['smooth_l1_beta'])
                        return fit+.01*d['initial_memory'].square().mean()+.01*d['innovation_force'].square().mean() if d else fit
                    print('FIT',fold,name,seed,flush=True)
                    log=train_module(model,tr[1],va[1],loss,cfg['training'],seed)
                    logs.extend({'fold':fold,'model':name,'seed':seed,**r} for r in log)
                    torch.save(model.state_dict(),directory/f'{name}_seed{seed}.pt')
                    with torch.no_grad():y,_=model(va[1],va[1]['acceleration'])
                    scores.append({'fold':fold,'model':name,'seed':seed,'validation_mae_deg':float(torch.rad2deg(y[:,HORIZON_INDEX]-va[1]['target_q'][:,HORIZON_INDEX]).abs().mean())})
                    pd.DataFrame(logs).to_csv(run/'training.csv',index=False);pd.DataFrame(scores).to_csv(run/'validation_seeds.csv',index=False)
            scaler=StandardScaler().fit(tr[3]);idx=np.linspace(0,len(tr[3])-1,min(len(tr[3]),cfg['kernel']['max_points'])).astype(int)
            # Same forced physical base as neural models; full available raw feature packet.
            residual=tr[0]['target_q'][:,HORIZON_INDEX]-tr[2];best=None
            with threadpool_limits(limits=4):
                for family in cfg['kernel']['families']:
                    for length in cfg['kernel']['length_scales']:
                        kernel=RBF(length) if family=='rbf' else Matern(length,nu=1.5)
                        gp=GaussianProcessRegressor(kernel=kernel+WhiteKernel(cfg['kernel']['noise']),optimizer=None,normalize_y=True,alpha=1e-6)
                        gp.fit(scaler.transform(tr[3][idx]),residual[idx]);pred=gp.predict(scaler.transform(va[3]))+va[2]
                        score=float(np.mean(abs(np.rad2deg(pred-va[0]['target_q'][:,HORIZON_INDEX]))))
                        kernel_scores.append({'fold':fold,'family':family,'length':length,'validation_mae_deg':score})
                        if best is None or score<best[0]:best=(score,gp)
                joblib.dump({'gp':best[1],'scaler':scaler},directory/'gp.joblib')
            pd.DataFrame(kernel_scores).to_csv(run/'kernel_validation.csv',index=False)
            continue
        models={}
        for name in cfg['models']:
            for seed in cfg['seeds']:
                m=model_for(name);m.load_state_dict(torch.load(directory/f'{name}_seed{seed}.pt',weights_only=True));m.eval();models[name,seed]=m
        kernel=joblib.load(directory/'gp.joblib')
        for group,sids in [('validation',split['validation']),('evaluation',split['evaluation'])]:
            for sid in sids:
                raw,b,base,features=prepare([sid],cfg['evaluation_stride'])
                pred={'forced_physics':base}
                with torch.no_grad():
                    for name in cfg['models']:
                        pred[name]=np.mean([models[name,seed](b,b['acceleration'])[0].numpy()[:,HORIZON_INDEX] for seed in cfg['seeds']],axis=0)
                with threadpool_limits(limits=4):pred['gp_reference']=kernel['gp'].predict(kernel['scaler'].transform(features))+base
                length=float(scenarios[sid].signals['L'][0]);low=min(provenance['folds'][fold]['train_lengths_m']);high=max(provenance['folds'][fold]['train_lengths_m'])
                transfer='short_extrapolation' if length<low else ('long_extrapolation' if length>high else 'interpolation')
                for name,y in pred.items():
                    for j,h in enumerate(HORIZONS_S):
                        for regime,mask in [('all',np.ones(len(y),bool)),('active_at_origin',abs(raw['initial_velocity'])>.005),('quiet_at_origin',abs(raw['initial_velocity'])<=.005)]:
                            if mask.any():metrics.append({'fold':fold,'group':group,'scenario':sid,'model':name,'horizon_s':h,'regime':regime,'transfer':transfer,'length_m':length,**regression_metrics(raw['target_q'][mask,HORIZON_INDEX[j]],y[mask,j])})
                np.savez_compressed(directory/f'predictions_s{sid}.npz',k=raw['k'],target=raw['target_q'][:,HORIZON_INDEX],**pred)
                pd.DataFrame(metrics).to_csv(run/'metrics.csv',index=False)
    if not args.evaluate:
        provenance['checkpoint_hashes']={str(p.relative_to(run)):sha256(p) for p in run.rglob('*') if p.suffix in ['.pt','.joblib']}
        provenance['status']='validation_complete';save();print(pd.DataFrame(scores).to_string(index=False));return
    provenance['status']='complete';provenance['finished_utc']=pd.Timestamp.now(tz='UTC').isoformat();save()
    print(pd.DataFrame(metrics).query("group == 'evaluation' and regime == 'all'").groupby(['fold','model','horizon_s']).mae_deg.mean().unstack().round(5).to_string())


if __name__=='__main__':main()
