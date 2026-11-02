"""GP reference with a physical zero-residual prior outside training support."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern,WhiteKernel
from threadpoolctl import threadpool_limits
from cranepilot.data import load_dataset,sha256
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows,stack_windows,tensors,NeuralActuator,HORIZON_INDEX,HORIZONS_S
from cranepilot.forced_observer import refine_windows,analytic_rollout
from cranepilot.metrics import regression_metrics
from cranepilot.benchmark import environment_record


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true')
    p.add_argument('--output',type=Path,default=Path('artifacts/runs/zero_prior_gp_campaign'))
    args=p.parse_args();run=args.output;cfg_path=Path('configs/zero_prior_gp_campaign.json');cfg=json.loads(cfg_path.read_text())
    paths=[cfg_path,Path('data/processed/dataset.npz'),Path('data/processed/manifest.json'),Path(__file__).relative_to(Path.cwd()),
        *[Path(s['actor']) for s in cfg['folds'].values()],*sorted(Path('src/cranepilot').glob('*.py'))]
    hashes={str(x):sha256(x) for x in paths}
    if not args.evaluate:
        if run.exists():raise FileExistsError('Preserve earlier runs')
        run.mkdir(parents=True);provenance={'status':'running','config':cfg,'guarded_hashes':hashes,'environment':environment_record(),'calibrations':{}}
    else:
        provenance=json.loads((run/'provenance.json').read_text())
        if provenance['status']!='validation_complete' or provenance['guarded_hashes']!=hashes:raise ValueError('Unchanged completed validation required')
        for name,h in provenance['checkpoint_hashes'].items():
            if sha256(run/name)!=h:raise ValueError('Kernel changed')
    def save():(run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if not args.evaluate:save()
    scenarios=load_dataset(Path('data/processed'));torch.set_num_threads(4);scores=[];metrics=[]
    with threadpool_limits(limits=4):
        for fold,split in cfg['folds'].items():
            cal=LengthCalibration(**provenance['calibrations'][fold]) if args.evaluate else LengthCalibration.fit(scenarios,split['train'],'theta')
            if not args.evaluate:provenance['calibrations'][fold]=asdict(cal);save()
            actor=NeuralActuator();actor.load_state_dict(torch.load(split['actor'],weights_only=True));actor.eval()
            def prepare(sids,stride):
                raw=stack_windows([command_windows(scenarios[s],cal,np.arange(40,len(scenarios[s].time)-150,stride)) for s in sids])
                with torch.no_grad():x,a=actor(tensors(raw))
                w=refine_windows(raw) if split['base']=='forced' else raw
                base=analytic_rollout(w['initial_q'],w['initial_p'],w['omega'],a.numpy())[:,HORIZON_INDEX]
                features=np.column_stack([raw['angular_context'],raw['omega'],raw['initial_x'],raw['initial_velocity'],raw['future_u'],a.numpy(),x.numpy()-raw['initial_x'][:,None]])
                return raw,features,base
            if not args.evaluate:
                tr=prepare(split['train'],cfg['train_stride']);va=prepare(split['validation'],cfg['evaluation_stride'])
                scaler=StandardScaler().fit(tr[1]);residual=tr[0]['target_q'][:,HORIZON_INDEX]-tr[2]
                scale=np.maximum(residual.std(axis=0),1e-3)
                selected=np.linspace(0,len(tr[1])-1,min(len(tr[1]),cfg['kernel']['max_points'])).astype(int);best=None
                for family in cfg['kernel']['families']:
                    for length in cfg['kernel']['length_scales']:
                        kernel=RBF(length) if family=='rbf' else Matern(length,nu=1.5)
                        gp=GaussianProcessRegressor(kernel=kernel+WhiteKernel(cfg['kernel']['noise']),optimizer=None,normalize_y=False,alpha=1e-6)
                        gp.fit(scaler.transform(tr[1][selected]),residual[selected]/scale)
                        pred=gp.predict(scaler.transform(va[1]))*scale+va[2]
                        score=float(np.mean(abs(np.rad2deg(pred-va[0]['target_q'][:,HORIZON_INDEX]))))
                        scores.append({'fold':fold,'family':family,'length':length,'validation_mae_deg':score});print(scores[-1],flush=True)
                        if best is None or score<best[0]:best=(score,gp)
                joblib.dump({'gp':best[1],'scaler':scaler,'scale':scale},run/f'{fold}.joblib')
                pd.DataFrame(scores).to_csv(run/'kernel_validation.csv',index=False);continue
            gp=joblib.load(run/f'{fold}.joblib')
            for group,sids in [('validation',split['validation']),('evaluation',split['evaluation']),('load_stress',split.get('load_stress',[]))]:
                for sid in sids:
                    raw,x,base=prepare([sid],cfg['evaluation_stride']);y=gp['gp'].predict(gp['scaler'].transform(x))*gp['scale']+base
                    lengths=[float(scenarios[s].signals['L'][0]) for s in split['train']];length=float(scenarios[sid].signals['L'][0])
                    transfer='short_extrapolation' if length<min(lengths) else ('long_extrapolation' if length>max(lengths) else 'interpolation')
                    for j,h in enumerate(HORIZONS_S):
                        for regime,mask in [('all',np.ones(len(y),bool)),('active_at_origin',abs(raw['initial_velocity'])>.005),('quiet_at_origin',abs(raw['initial_velocity'])<=.005)]:
                            if mask.any():metrics.append({'fold':fold,'group':group,'scenario':sid,'model':'zero_prior_gp','horizon_s':h,'regime':regime,'transfer':transfer,**regression_metrics(raw['target_q'][mask,HORIZON_INDEX[j]],y[mask,j])})
                    np.savez_compressed(run/f'predictions_{fold}_s{sid}.npz',k=raw['k'],target=raw['target_q'][:,HORIZON_INDEX],zero_prior_gp=y)
    if not args.evaluate:
        provenance['checkpoint_hashes']={p.name:sha256(p) for p in run.glob('*.joblib')};provenance['status']='validation_complete';save();return
    pd.DataFrame(metrics).to_csv(run/'metrics.csv',index=False);provenance['status']='complete';save()
    print(pd.DataFrame(metrics).query("group == 'evaluation' and regime == 'all'").groupby(['fold','horizon_s']).mae_deg.mean().unstack().round(5).to_string())


if __name__=='__main__':main()
