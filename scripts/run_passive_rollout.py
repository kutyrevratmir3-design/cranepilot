"""Fixed development campaign; log actuator reliability before neural sway fitting."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from cranepilot.data import load_dataset,sha256
from cranepilot.physics import LengthCalibration
from cranepilot.benchmark import environment_record
from cranepilot.metrics import regression_metrics
from cranepilot.rollout import (command_windows,stack_windows,tensors,NeuralActuator,
    PassiveForecaster,train_module,N_STEPS,RAW_STEP,STEP,HORIZONS_S,HORIZON_INDEX)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/passive_rollout_pilot.json'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/runs/passive_rollout'))
    parser.add_argument('--actuator-only',action='store_true')
    parser.add_argument('--actuator-checkpoint',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new run directory; never overwrite retained evidence')
    args.output.mkdir(parents=True)
    config=json.loads(args.config.read_text())
    scenarios=load_dataset(Path('data/processed'))
    calibration=LengthCalibration.fit(scenarios,config['train'],'theta')
    provenance={'status':'running','config':config,'config_sha256':sha256(args.config),
                'environment':environment_record(),'data_manifest_sha256':sha256(Path('data/processed/manifest.json')),
                'actuator_only':args.actuator_only,'started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    train_parts=[]; validation_parts=[]
    for group,output in [('train',train_parts),('validation',validation_parts)]:
        for sid in config[group]:
            end=len(scenarios[sid].time)-N_STEPS*RAW_STEP
            if group=='train':
                origins=np.r_[np.arange(40,2000,10),np.arange(2000,end,50)]
            else:
                origins=np.arange(40,end,10)
            output.append(command_windows(scenarios[sid],calibration,origins))
    tr=tensors(stack_windows(train_parts)); va=tensors(stack_windows(validation_parts))
    torch.manual_seed(config['actuator_seed'])
    actuator=NeuralActuator()
    def actuator_loss(model,batch):
        position,acceleration=model(batch)
        return ((position-batch['target_x'])/.05).square().mean()+1e-5*acceleration.square().mean()
    print('Training the native-reference actuator model',flush=True)
    if args.actuator_checkpoint:
        actuator.load_state_dict(torch.load(args.actuator_checkpoint,weights_only=True))
        actuator.eval()
        provenance['retained_actuator_sha256']=sha256(args.actuator_checkpoint)
        (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    else:
        log=train_module(actuator,tr,va,actuator_loss,config['actuator_training'],seed=config['actuator_seed'])
        pd.DataFrame(log).to_csv(args.output/'actuator_training.csv',index=False)
    torch.save(actuator.state_dict(),args.output/'actuator.pt')
    rows=[]
    # Development actuator scoring is kept separate from angular model selection.
    for sid in config['validation']:
        w=command_windows(scenarios[sid],calibration,np.arange(40,len(scenarios[sid].time)-150,10))
        batch=tensors(w)
        with torch.no_grad():
            position,_=actuator(batch)
        const=w['initial_x'][:,None]+w['initial_velocity'][:,None]*np.arange(1,31)[None,:]*STEP
        persistence=np.broadcast_to(w['initial_x'][:,None],const.shape)
        for name,prediction in [('neural_actuator',position.numpy()),('constant_velocity',const),('persistence',persistence)]:
            for j,horizon in enumerate(HORIZONS_S):
                column=HORIZON_INDEX[j]
                error=prediction[:,column]-w['target_x'][:,column]
                rows.append({'scenario':sid,'model':name,'horizon_s':horizon,
                             'mae_m':float(np.mean(abs(error))),'rmse_m':float(np.sqrt(np.mean(error**2)))})
    pd.DataFrame(rows).to_csv(args.output/'actuator_validation.csv',index=False)
    print(pd.DataFrame(rows).pivot(index='model',columns='horizon_s',values='mae_m').round(5).to_string(),flush=True)
    if args.actuator_only:
        provenance['status']='complete_actuator_development'
        (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
        return
    with torch.no_grad():
        for batch in (tr,va):
            _,acceleration=actuator(batch)
            batch['predicted_acceleration']=acceleration.detach()
    training=[]; models={}
    for memory in config['angular_memory_dimensions']:
        for seed in config['seeds']:
            torch.manual_seed(seed)
            model=PassiveForecaster(memory_dim=memory)
            def angular_loss(module,batch):
                prediction,diagnostics=module(batch,batch['predicted_acceleration'])
                error=(prediction-batch['target_q'])/.1
                memory_penalty=diagnostics['initial_memory'].square().sum(-1).mean()
                return error.square().mean()+.01*memory_penalty
            print(f'Training passive memory={memory} seed={seed}',flush=True)
            log=train_module(model,tr,va,angular_loss,config['angular_training'],seed)
            training.extend({'memory':memory,'seed':seed,**row} for row in log)
            torch.save(model.state_dict(),args.output/f'passive_m{memory}_seed{seed}.pt')
            models[(memory,seed)]=model
            pd.DataFrame(training).to_csv(args.output/'angular_training.csv',index=False)
    metrics=[]; timings=[]
    for group in ('validation','test'):
        for sid in config[group]:
            w=command_windows(scenarios[sid],calibration,np.arange(40,len(scenarios[sid].time)-150,10))
            batch=tensors(w)
            with torch.no_grad():
                x,acceleration=actuator(batch)
                # Identical action-conditioned inputs and analytic initial state.
                zero=PassiveForecaster(memory_dim=0).cell
                coefficients=(torch.zeros(len(x)),torch.zeros(len(x),0),torch.zeros(len(x),0))
                state=torch.stack([batch['initial_q'],batch['initial_p']],-1)
                output=[]
                for k in range(N_STEPS):
                    state,_=zero.step(state,batch['omega'],acceleration[:,k],STEP,coefficients)
                    output.append(state[:,0])
                predictions={'calibrated_harmonic':w['harmonic'],
                             'action_physics':torch.stack(output,1).numpy()[:,HORIZON_INDEX]}
                balance={}
                for memory in config['angular_memory_dimensions']:
                    ensemble=[]
                    for seed in config['seeds']:
                        started=time.perf_counter()
                        predicted,diagnostics=models[(memory,seed)](batch,acceleration)
                        elapsed=time.perf_counter()-started
                        timings.append({'group':group,'scenario':sid,'memory':memory,'seed':seed,
                                        'total_s':elapsed,'n_origins':len(x),
                                        'parameters':sum(p.numel() for p in models[(memory,seed)].parameters()),
                                        'maximum_balance_residual':float(diagnostics['balance'].abs().max())})
                        ensemble.append(predicted.numpy())
                    predictions[f'passive_m{memory}']=np.mean(ensemble,axis=0)[:,HORIZON_INDEX]
            for name,prediction in predictions.items():
                for j,horizon in enumerate(HORIZONS_S):
                    target=w['target_q'][:,HORIZON_INDEX[j]]
                    for regime,mask in [('all',np.ones(len(target),dtype=bool)),
                                        ('active_at_origin',np.abs(w['initial_velocity'])>.005),
                                        ('quiet_at_origin',np.abs(w['initial_velocity'])<=.005)]:
                        if mask.any():
                            metrics.append({'group':group,'scenario':sid,'model':name,'horizon_s':horizon,
                                            'regime':regime,**regression_metrics(target[mask],prediction[mask,j])})
            np.savez_compressed(args.output/f'predictions_s{sid}.npz',k=w['k'],target=w['target_q'][:,HORIZON_INDEX],
                                predicted_x=x.numpy(),**predictions)
            pd.DataFrame(metrics).to_csv(args.output/'metrics.csv',index=False)
            pd.DataFrame(timings).to_csv(args.output/'timing.csv',index=False)
    provenance['status']='complete'
    provenance['finished_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(pd.DataFrame(metrics).query("regime == 'all'").groupby(['group','model','horizon_s']).mae_deg.mean().unstack().round(4).to_string())


if __name__=='__main__':
    main()
