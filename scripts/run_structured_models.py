"""New structured residual and stable-latent models; locked TRAIN then evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from cranepilot.benchmark import environment_record
from cranepilot.broad_strategies import make_model
from cranepilot.data import load_dataset,sha256
from cranepilot.rollout import command_windows,stack_windows,HORIZON_INDEX,HORIZONS_S
from scripts.structured_candidate import StructuredCandidate, make_structured_model, fit_structured, CONTROLS
from scripts.run_global_research import verify,write
from scripts.run_broad_search import inputs,predict,scores
from scripts.predict_cranepilot import build_packet
from scripts.forced_adaptive_simulator import simulate_forced


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['train','evaluate'],required=True)
    parser.add_argument('--output',type=Path,default=Path('artifacts/runs/structured_models_20261107'))
    args=parser.parse_args();out=args.output
    cfgpath=Path('configs/structured_models_20261107.json');cfg=json.loads(cfgpath.read_text())
    parent=Path(cfg['parent_run']);previous=json.loads((parent/'provenance.json').read_text())
    if previous['status']!='complete':raise ValueError('Completed parent required')
    verify(previous['guarded_hashes']);verify(previous['checkpoint_hashes'],parent);verify(previous['result_hashes'],parent)
    locked=[cfgpath,Path('docs/STRUCTURED_MODELS_DESIGN_20261107.md'),Path(__file__).relative_to(Path.cwd()),
            Path('scripts/structured_candidate.py'),Path('scripts/residual_adapter.py'),Path('scripts/stable_latent.py'),
            Path('tests/test_residual_adapter.py'),Path('tests/test_stable_latent.py')]
    if args.stage=='train':
        for path in locked:
            if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit first: {path}')
        if out.exists():raise FileExistsError('Retain earlier attempts')
        out.mkdir(parents=True)
        hashes=dict(previous['guarded_hashes']);hashes.update({str(p):sha256(p) for p in [*locked,parent/'provenance.json',parent/'selection.json']})
        prov=dict(status='training',config=cfg,guarded_hashes=hashes,environment=environment_record(),
                  started_utc=pd.Timestamp.now(tz='UTC').isoformat(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  real_records_retrospective=True,actor_training=False,neural_finetuning=False,new_architecture_training=True,closed_loop_control_evaluated=False)
        write(out/'provenance.json',prov)
    else:
        prov=json.loads((out/'provenance.json').read_text())
        if prov['status']!='selection_complete':raise ValueError('Frozen selection required')
        verify(prov['guarded_hashes']);verify(prov['checkpoint_hashes'],out)
        verify({'selection.json':prov['selection_sha256']},out)
    try:run(args.stage,out,cfg,prov)
    except Exception as exc:
        prov.update(status='failed',failure=f'{type(exc).__name__}: {exc}');write(out/'provenance.json',prov);raise


def run(stage,out,cfg,prov):
    torch.set_num_threads(4)
    data=load_dataset(Path('data/processed'))
    old=json.loads((Path(cfg['source_run'])/'provenance.json').read_text())
    frozen=json.loads((out/'selection.json').read_text()) if stage=='evaluate' else None
    selection={};metrics=[];risks=[];checks=[];timings=[];logs=[];fits=[];simmetrics=[]

    def record(fold,group,sid,extra,raw,predictions):
        active=np.abs(raw['initial_velocity'])>=cfg['active_threshold']
        for name,y in predictions.items():
            error=np.rad2deg(y-raw['target_q']);absolute=np.abs(error)
            increment=np.abs(np.diff(error,axis=1)/(raw['omega'][:,None]*.05))
            for regime,mask in [('all',np.ones(len(y),bool)),('active',active),('quiet',~active)]:
                if not mask.any():continue
                for j,h in enumerate(HORIZONS_S):
                    metrics.append(dict(fold=fold,group=group,scenario=sid,extrapolation=extra,model=name,regime=regime,
                                        horizon_s=h,mae_deg=float(absolute[mask,HORIZON_INDEX[j]].mean())))
                risks.append(dict(fold=fold,group=group,scenario=sid,extrapolation=extra,model=name,regime=regime,
                                  mean_packet_peak_deg=float(absolute[mask].max(1).mean()),
                                  packet_mae_p95_deg=float(np.quantile(absolute[mask].mean(1),.95)),
                                  tail_mae_deg=float(absolute[mask,-10:].mean()),
                                  normalized_increment_mae_deg=float(increment[mask].mean())))

    for fold in cfg['folds']:
        split=old['config']['folds'][fold];directory=out/fold
        parent=StructuredCandidate(cfg,fold);cal=parent.calibration
        lengths=[float(data[s].signals['L'][0]) for s in split['train']]
        def packet(sids,stride):
            return stack_windows([command_windows(data[s],cal,np.arange(40,len(data[s].time)-150,stride)) for s in sids])
        if stage=='train':
            directory.mkdir();raw_train=packet(split['train'],cfg['train_stride']);train=inputs(raw_train,parent.actor)
            raw_val=packet(split['validation'],cfg['evaluation_stride']);validation=inputs(raw_val,parent.actor)
            target=torch.as_tensor(raw_train['target_q'],dtype=torch.float32)
            active=torch.as_tensor(np.abs(raw_train['initial_velocity'])>=cfg['active_threshold'])
            scenario=torch.as_tensor(raw_train['scenario'])
            valtarget=raw_val['target_q'];valactive=np.abs(raw_val['initial_velocity'])>=cfg['active_threshold']
            for name,choice in cfg['variants'].items():
                for seed in cfg['seeds']:
                    torch.manual_seed(seed);model=make_structured_model(cfg,fold,seed,choice)
                    start=time.perf_counter()
                    state,history,meta=fit_structured(model,train,target,active,scenario,validation,valtarget,valactive,
                                                   cfg['training'],choice,seed)
                    torch.save(state,directory/f'{name}_seed{seed}.pt')
                    model.load_state_dict(state);expected=predict(model,validation)
                    reloaded=make_structured_model(cfg,fold,seed,choice);reloaded.load_state_dict(torch.load(directory/f'{name}_seed{seed}.pt',weights_only=True))
                    delta=float(np.abs(np.rad2deg(expected-predict(reloaded,validation))).max())
                    if delta>cfg['replay_tolerance_deg']:raise AssertionError(('checkpoint_reload',delta))
                    checks.append(dict(kind='checkpoint_reload',fold=fold,model=name,seed=seed,max_deg=delta))
                    fits.append(dict(fold=fold,model=name,elapsed_seconds=time.perf_counter()-start,**meta))
                    logs.extend(dict(fold=fold,model=name,seed=seed,**row) for row in history)
                    write(out/'fits.json',fits);pd.DataFrame(logs).to_csv(out/'training.csv',index=False)
                    print('FIT',fold,name,seed,json.dumps(meta),flush=True)
        predictor=StructuredCandidate(cfg,fold,directory)
        groups=[('validation',split['validation'])] if stage=='train' else [('evaluation',split['evaluation']),('load_stress',split['load_stress'])]
        values=[]
        for group,sids in groups:
            for sid in sids:
                raw=packet([sid],cfg['evaluation_stride']);k=raw['k'];s=data[sid]
                start=time.perf_counter();pred,seeds,acc=predictor.forecast(raw)
                timings.append(dict(fold=fold,scenario=sid,packets=len(k),all_methods_seconds=time.perf_counter()-start))
                if set(pred)!=set(CONTROLS+list(cfg['variants'])):raise AssertionError('Fixed method set')
                with np.load(Path(cfg['parent_run'])/fold/f'predictions_s{sid}.npz') as retained:
                    np.testing.assert_array_equal(k,retained['k']);np.testing.assert_array_equal(raw['target_q'],retained['target'])
                    for name in CONTROLS:
                        prior_name=predictor.parent_selected if name=='previous_local' else name
                        delta=float(np.abs(np.rad2deg(pred[name]-retained[prior_name])).max())
                        if delta>cfg['replay_tolerance_deg']:raise AssertionError(('parent_replay',delta))
                        checks.append(dict(kind='parent_replay',fold=fold,scenario=sid,model=name,max_deg=delta))
                if stage=='train':
                    a=np.abs(raw['initial_velocity'])>=cfg['active_threshold']
                    values.append({name:scores(y,raw['target_q'],a) for name,y in pred.items()})
                else:
                    pred['selected']=pred[frozen['folds'][fold]['model']];pred['common']=pred[frozen['common']['model']]
                extra=float(s.signals['L'][0])<min(lengths) or float(s.signals['L'][0])>max(lengths)
                record(fold,group,sid,extra,raw,pred)
                np.savez_compressed(directory/f'predictions_s{sid}.npz',k=k,target=raw['target_q'],acceleration=acc,
                                    **pred,**{n+'__seeds':y for n,y in seeds.items()})
                ids=np.array([0,len(k)//2,len(k)-1]);small={key:v[ids].copy() for key,v in raw.items()}
                clean,_,_=predictor.forecast(small);small['target_q'][:]=999;small['target_x'][:]=-999
                poisoned,_,_=predictor.forecast(small)
                for name in clean:np.testing.assert_array_equal(clean[name],poisoned[name])
                past=k[ids,None]-np.arange(41)
                assembled=build_packet(s.signals['theta'][past],s.signals['x'][past],s.signals['xr'][past],s.signals['L'][k[ids]],
                                       s.signals['xr'][k[ids,None]+np.arange(30)*5],cal)
                independent,_,_=predictor.forecast(assembled)
                delta=max(float(np.abs(np.rad2deg(independent[n]-clean[n])).max()) for n in clean)
                if delta>cfg['replay_tolerance_deg']:raise AssertionError(('history_only',delta))
                checks.append(dict(kind='history_only',fold=fold,scenario=sid,max_deg=delta))
                print('REAL',stage,fold,sid,flush=True)
        if stage=='train':
            validation={n:{reg:float(np.mean([v[n][reg] for v in values])) for reg in ['all','active','quiet']} for n in values[0]}
            eligible=[n for n in validation if all(validation[n][r]<=validation[cfg['selection']['reference']][r]*cfg['selection'][r+'_max_ratio']+1e-12 for r in ['all','active','quiet'])]
            chosen=min(eligible,key=lambda n:validation[n]['all']);selection[fold]=dict(model=chosen,eligible=eligible,validation=validation)
            print('SELECT',fold,chosen,validation[chosen],flush=True);continue
        meta,pkt,acc,target=simulate_forced(cfg['simulation'],cal)
        _,_,fineacc,finetarget=simulate_forced(cfg['simulation'],cal,dt=.0005)
        np.testing.assert_array_equal(acc,fineacc);delta=float(np.abs(np.rad2deg(target-finetarget)).max())
        if delta>cfg['replay_tolerance_deg']:raise AssertionError(('integrator',delta))
        checks.append(dict(kind='simulation_convergence',fold=fold,max_deg=delta))
        raw=build_packet(pkt['theta'],pkt['x'],pkt['reference'],pkt['length_m'],pkt['future_reference'],cal)
        pred,seeds,_=predictor.forecast(raw,prescribed_acceleration=acc)
        pred['selected']=pred[frozen['folds'][fold]['model']];pred['common']=pred[frozen['common']['model']]
        a=np.flatnonzero(meta.future_acceleration_gain.to_numpy()==1);b=np.flatnonzero(meta.future_acceleration_gain.to_numpy()==1.15)
        for value in pkt.values():np.testing.assert_array_equal(value[a],value[b])
        delta=max(float(np.abs(np.rad2deg(y[a]-y[b])).max()) for y in pred.values())
        if delta>cfg['replay_tolerance_deg']:raise AssertionError(('future_gain',delta))
        checks.append(dict(kind='future_gain_invariance',fold=fold,max_deg=delta))
        for name,y in pred.items():
            error=np.abs(np.rad2deg(y[:,HORIZON_INDEX]-target[:,HORIZON_INDEX]))
            for j,h in enumerate(HORIZONS_S):
                frame=meta.copy();frame['fold']=fold;frame['model']=name;frame['horizon_s']=h;frame['absolute_error_deg']=error[:,j];simmetrics.append(frame)
        meta.to_csv(directory/'simulation_cases.csv',index=False)
        np.savez_compressed(directory/'simulation_predictions.npz',target=target,acceleration=acc,**pkt,**pred,**{n+'__seeds':y for n,y in seeds.items()})
        print('SIMULATION',fold,len(meta),flush=True)
    pd.DataFrame(metrics).to_csv(out/f'{stage}_metrics.csv',index=False);pd.DataFrame(risks).to_csv(out/f'{stage}_risk_metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(out/f'{stage}_timings.csv',index=False)
    write(out/f'{stage}_verification.json',dict(passed=True,checks=checks,tolerance_deg=cfg['replay_tolerance_deg']))
    if stage=='train':
        common=[n for n in selection[cfg['folds'][0]]['eligible'] if all(n in selection[f]['eligible'] for f in cfg['folds'])]
        chosen=min(common,key=lambda n:np.mean([selection[f]['validation'][n]['all']/selection[f]['validation'][cfg['selection']['reference']]['all'] for f in cfg['folds']]))
        write(out/'selection.json',dict(folds=selection,common=dict(model=chosen,eligible=common),uses_evaluation_this_stage=False,uses_simulation=False,real_records_reused=True))
        prov.update(status='selection_complete',selection_sha256=sha256(out/'selection.json'),selection_locked_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                    checkpoint_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*.pt')},completed_fits=len(fits))
    else:
        pd.concat(simmetrics).to_csv(out/'simulation_metrics.csv',index=False)
        prov.update(status='complete',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),result_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
    write(out/'provenance.json',prov);print('COMPLETE',stage,flush=True)


if __name__=='__main__':main()
