"""Same-information residual references for the retained action-conditioned pilot."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from sklearn.preprocessing import StandardScaler

from cranepilot.data import load_dataset,sha256
from cranepilot.physics import LengthCalibration,phase_features
from cranepilot.benchmark import environment_record
from cranepilot.metrics import regression_metrics
from cranepilot.models import NeuralForecaster,NeuralConfig
from cranepilot.rollout import (command_windows,stack_windows,tensors,NeuralActuator,
    PassiveMemoryCell,N_STEPS,STEP,HORIZON_INDEX,HORIZONS_S)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('artifacts/runs/rollout_references'))
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists')
    args.output.mkdir(parents=True)
    config=json.loads(Path('configs/passive_rollout_pilot.json').read_text())
    settings={'gp_length_scales':[1.5,3.,6.,12.],'gp_max_points':512,
              'gp_noise_level':.03,'mlp_seeds':[0,1,2],
              'mlp_hidden':32,'mlp_epochs':50,'mlp_patience':9,
              'selection':'mean validation MAE across seven horizons; no evaluation tuning',
              'scope':'direct multi-horizon residual references; not recursive control models'}
    provenance={'status':'running','settings':settings,'config':config,'environment':environment_record(),
                'actuator_sha256':sha256(Path('artifacts/runs/actuator_development/actuator.pt'))}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    scenarios=load_dataset(Path('data/processed'))
    calibration=LengthCalibration.fit(scenarios,config['train'],'theta')
    actuator=NeuralActuator()
    actuator.load_state_dict(torch.load('artifacts/runs/actuator_development/actuator.pt',weights_only=True))
    actuator.eval();torch.set_num_threads(4)
    def features(w):
        b=tensors(w)
        with torch.no_grad():
            positions,acceleration=actuator(b)
            state=torch.stack([b['initial_q'],b['initial_p']],-1)
            cell=PassiveMemoryCell(memory_dim=0)
            co=(torch.zeros(len(state)),torch.zeros(len(state),0),torch.zeros(len(state),0))
            predictions=[]
            for i in range(N_STEPS):
                state,_=cell.step(state,b['omega'],acceleration[:,i],STEP,co)
                predictions.append(state[:,0])
            baseline=torch.stack(predictions,1).numpy()[:,HORIZON_INDEX]
        # Features retain the complete past angle packet and complete prescribed
        # input sequence, plus the deterministic actuator response shared by PEPM.
        x=np.column_stack([w['angular_context'],w['omega'],w['initial_x'],w['initial_velocity'],
                           w['future_u'],acceleration.numpy(),positions.numpy()-w['initial_x'][:,None]])
        return x,baseline,w['target_q'][:,HORIZON_INDEX]
    grouped={}
    for group in ('train','validation','test'):
        parts=[]
        for sid in config[group]:
            end=len(scenarios[sid].time)-150
            origins=np.r_[np.arange(40,2000,10),np.arange(2000,end,50)] if group=='train' else np.arange(40,end,10)
            parts.append(command_windows(scenarios[sid],calibration,origins))
        grouped[group]=(parts,features(stack_windows(parts)))
    _,(xt,bt,yt)=grouped['train'];_,(xv,bv,yv)=grouped['validation']
    scaler=StandardScaler().fit(xt)
    selected=np.linspace(0,len(xt)-1,min(len(xt),settings['gp_max_points'])).astype(int)
    candidates=[];best=None
    for length in settings['gp_length_scales']:
        gp=GaussianProcessRegressor(kernel=RBF(length)+WhiteKernel(.03),optimizer=None,
                                     normalize_y=True,alpha=1e-6)
        gp.fit(scaler.transform(xt[selected]),(yt-bt)[selected])
        error=float(np.mean(abs(np.rad2deg(gp.predict(scaler.transform(xv))+bv-yv))))
        candidates.append({'length_scale':length,'validation_mae_deg':error})
        if best is None or error<best[0]:
            best=(error,length,gp)
    (args.output/'gp_validation.json').write_text(json.dumps(candidates,indent=2))
    print('Frozen GP selection:',best[:2],flush=True)
    networks=[];training=[]
    for seed in settings['mlp_seeds']:
        model=NeuralForecaster(NeuralConfig(hidden=32,epochs=50,patience=9),seed)
        model.fit(xt,yt-bt,xv,yv-bv)
        model.save(args.output/f'residual_mlp_seed{seed}.pt')
        networks.append(model)
        training.extend({'seed':seed,**r} for r in model.log)
    pd.DataFrame(training).to_csv(args.output/'training.csv',index=False)
    metrics=[]
    for group in ('validation','test'):
        for part in grouped[group][0]:
            x,base,target=features(part)
            pred={'gp_action_residual':base+best[2].predict(scaler.transform(x)),
                  'mlp_action_residual':base+np.mean([m.predict(x) for m in networks],axis=0)}
            for name,prediction in pred.items():
                for j,horizon in enumerate(HORIZONS_S):
                    for regime,mask in [('all',np.ones(len(target),dtype=bool)),
                                        ('active_at_origin',abs(part['initial_velocity'])>.005),
                                        ('quiet_at_origin',abs(part['initial_velocity'])<=.005)]:
                        if mask.any():
                            metrics.append({'group':group,'scenario':int(part['scenario'][0]),'model':name,
                                            'horizon_s':horizon,'regime':regime,
                                            **regression_metrics(target[mask,j],prediction[mask,j])})
            np.savez_compressed(args.output/f'predictions_s{int(part["scenario"][0])}.npz',target=target,**pred)
    pd.DataFrame(metrics).to_csv(args.output/'metrics.csv',index=False)
    provenance.update(status='complete',gp_selected_length=best[1])
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(pd.DataFrame(metrics).query("regime == 'all'").groupby(['group','model','horizon_s']).mae_deg.mean().unstack().round(4).to_string())


if __name__=='__main__':
    main()
