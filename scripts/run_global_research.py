"""Preregister, fit global models on TRAIN, lock validation selection, then evaluate."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from cranepilot.data import load_dataset, sha256
from cranepilot.benchmark import environment_record
from cranepilot.rollout import command_windows, stack_windows, HORIZON_INDEX, HORIZONS_S
from scripts.predict_cranepilot import build_packet
from scripts.combined_candidate import CombinedCandidate
from scripts.global_research_candidate import GlobalResearchCandidate, control_forecasts, CONTROLS
from scripts.global_delay_lift import fit_path, save_model
from scripts.global_weak_physics import fit_global
from scripts.forced_adaptive_simulator import simulate_forced
from scripts.run_broad_search import scores


def write(path, obj): path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def verify(mapping, base=Path('.')):
    for name, digest in mapping.items():
        if sha256(base/name.replace('\\', '/')) != digest: raise ValueError(f'Changed evidence: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/runs/global_research_20261107'))
    args = parser.parse_args(); out = args.output
    cfgpath = Path('configs/global_research_20261107.json'); cfg = json.loads(cfgpath.read_text())
    parent = Path(cfg['parent_run']); previous = json.loads((parent/'provenance.json').read_text())
    if previous['status'] != 'complete': raise ValueError('Completed parent required')
    verify(previous['guarded_hashes']); verify(previous['result_hashes'], parent)
    locked = [cfgpath, Path('docs/GLOBAL_RESEARCH_DESIGN_20261107.md'), Path('scripts/run_global_research.py'),
              *Path('scripts').glob('global_*.py'), *Path('tests').glob('test_global_*.py')]
    if args.stage == 'train':
        for path in locked:
            if subprocess.check_output(['git', 'show', f'HEAD:{path}']) != path.read_bytes():
                raise ValueError(f'Commit experimental sources first: {path}')
        if out.exists(): raise FileExistsError('Preserve previous attempts')
        out.mkdir(parents=True)
        hashes = dict(previous['guarded_hashes'])
        hashes.update({str(p): sha256(p) for p in [*locked, parent/'provenance.json']})
        prov = {'status': 'training', 'config': cfg, 'guarded_hashes': hashes, 'environment': environment_record(),
                'started_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'git_head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'historically_reused_real_records': True, 'neural_training': False, 'global_linear_model_fitting': True}
        write(out/'provenance.json', prov)
    else:
        prov = json.loads((out/'provenance.json').read_text())
        if prov['status'] != 'selection_complete': raise ValueError('Frozen selection required')
        verify(prov['guarded_hashes']); verify(prov['checkpoint_hashes'], out)
        if sha256(out/'selection.json') != prov['selection_sha256']: raise ValueError('Selection changed')
    try: run(args.stage, out, cfg, prov)
    except Exception as exc:
        prov.update(status='failed', failure=f'{type(exc).__name__}: {exc}'); write(out/'provenance.json', prov); raise


def run(stage, out, cfg, prov):
    torch.set_num_threads(4)
    data = load_dataset(Path('data/processed'))
    old = json.loads((Path(cfg['source_run'])/'provenance.json').read_text())
    selection = {}; metrics = []; checks = []; fitting = []; simulation = []; timings = []
    frozen = json.loads((out/'selection.json').read_text()) if stage == 'evaluate' else None

    def record(fold, group, sid, extra, raw, predictions):
        active = np.abs(raw['initial_velocity']) >= cfg['active_threshold']
        for name, y in predictions.items():
            error = np.abs(np.rad2deg(y[:, HORIZON_INDEX]-raw['target_q'][:, HORIZON_INDEX]))
            for regime, mask in [('all',np.ones(len(y),bool)),('active',active),('quiet',~active)]:
                if not mask.any(): continue
                for j,h in enumerate(HORIZONS_S):
                    metrics.append(dict(fold=fold, group=group, scenario=sid, extrapolation=extra,
                                        model=name, regime=regime, horizon_s=h, mae_deg=float(error[mask,j].mean())))

    for fold in cfg['folds']:
        split = old['config']['folds'][fold]; directory = out/fold
        parent = CombinedCandidate(cfg, fold); cal = parent.calibration
        lengths = [float(data[s].signals['L'][0]) for s in split['train']]
        if stage == 'train':
            directory.mkdir(); manifest = {}
            train = stack_windows([command_windows(data[s],cal,np.arange(40,len(data[s].time)-150,cfg['train_stride'])) for s in split['train']])
            controls, acc = control_forecasts(parent, train)
            target = train['target_q']-controls['physical']
            # Equal total weight per complete training record, even if lengths differ.
            weights = np.array([1/np.count_nonzero(train['scenario']==sid) for sid in train['scenario']])
            for index,ridge in enumerate(cfg['weak_ridges']):
                started = time.perf_counter(); model = fit_global([data[s] for s in split['train']], cal, ridge=ridge)
                name = f'weak_r{index}'; file = f'{name}.json'; write(directory/file, model)
                manifest[name] = {'family':'weak','file':file,'ridge':ridge}
                fitting.append(dict(fold=fold, model=name, seconds=time.perf_counter()-started, train_scenarios=split['train']))
            for basis in cfg['bases']:
                for mode in cfg['input_modes']:
                    started = time.perf_counter()
                    models = fit_path(train,acc,target,alphas=cfg['alphas'],basis=basis,input_mode=mode,weights=weights)
                    elapsed = time.perf_counter()-started
                    for index,alpha in enumerate(cfg['alphas']):
                        model = models[alpha]; name = f'{basis}_{mode}_r{index}'; file = f'{name}.npz'
                        save_model(model,directory/file)
                        manifest[name] = {'family':'delay','basis':basis,'input_mode':mode,'alpha':alpha,'file':file}
                        fitting.append(dict(fold=fold,model=name,path_fit_seconds=elapsed,train_windows=len(target),train_scenarios=split['train']))
                    print('FIT',fold,basis,mode,round(elapsed,2),'seconds',flush=True)
            write(directory/'models.json',manifest)
        predictor = GlobalResearchCandidate(cfg,fold,directory)
        groups = [('validation',split['validation'])] if stage=='train' else [('evaluation',split['evaluation']),('load_stress',split['load_stress'])]
        valparts = []
        for group,sids in groups:
            for sid in sids:
                s = data[sid]; k = np.arange(40,len(s.time)-150,cfg['evaluation_stride']); raw = command_windows(s,cal,k)
                start = time.perf_counter(); pred,acc = predictor.forecast(raw)
                timings.append(dict(fold=fold,scenario=sid,packets=len(k),all_models_seconds=time.perf_counter()-start))
                if set(pred) != set(CONTROLS+list(predictor.manifest)): raise AssertionError('Fixed model set')
                with np.load(Path(cfg['parent_run'])/fold/f'predictions_s{sid}.npz') as retained:
                    np.testing.assert_array_equal(k,retained['k']); np.testing.assert_array_equal(raw['target_q'],retained['target'])
                    for name in CONTROLS:
                        delta = float(np.abs(np.rad2deg(pred[name]-retained[name])).max())
                        if delta > 5e-5: raise AssertionError(('parent_replay',fold,sid,name,delta))
                        checks.append(dict(kind='parent_replay',fold=fold,scenario=sid,model=name,max_deg=delta))
                if stage=='train':
                    active = np.abs(raw['initial_velocity']) >= cfg['active_threshold']
                    valparts.append({n:scores(y,raw['target_q'],active) for n,y in pred.items()})
                else:
                    pred['selected'] = pred[frozen['folds'][fold]['model']]; pred['common'] = pred[frozen['common']['model']]
                extra = float(s.signals['L'][0]) < min(lengths) or float(s.signals['L'][0]) > max(lengths)
                record(fold,group,sid,extra,raw,pred)
                np.savez_compressed(directory/f'predictions_s{sid}.npz',k=k,target=raw['target_q'],acceleration=acc,**pred)
                # Every real record: independent history-only API, poison targets, reload saved fits.
                ix = np.array([0,len(k)//2,len(k)-1]); small = {key:v[ix].copy() for key,v in raw.items()}
                clean,_ = predictor.forecast(small); small['target_q'][:]=999; small['target_x'][:]=-999
                poisoned,_ = predictor.forecast(small)
                for name in clean: np.testing.assert_array_equal(clean[name],poisoned[name])
                past = k[ix,None]-np.arange(41)
                packet = build_packet(s.signals['theta'][past],s.signals['x'][past],s.signals['xr'][past],s.signals['L'][k[ix]],s.signals['xr'][k[ix,None]+np.arange(30)*5],cal)
                independent,_ = predictor.forecast(packet)
                delta = max(float(np.abs(np.rad2deg(independent[n]-clean[n])).max()) for n in clean)
                if delta > 5e-5: raise AssertionError(('history_only',delta))
                checks.append(dict(kind='history_only_target_invariance',fold=fold,scenario=sid,max_deg=delta))
                print('REAL',stage,fold,sid,flush=True)
        if stage=='train':
            val = {n:{reg:float(np.mean([v[n][reg] for v in valparts])) for reg in ['all','active','quiet']} for n in valparts[0]}
            eligible = [n for n in val if all(val[n][r] <= val['incumbent'][r]*cfg['selection'][r+'_max_ratio']+1e-12 for r in ['all','active','quiet'])]
            chosen = min(eligible,key=lambda n:val[n]['all']); selection[fold] = dict(model=chosen,eligible=eligible,validation=val)
            print('SELECT',fold,chosen,val[chosen],flush=True)
            continue
        meta,packet,acc,target = simulate_forced(cfg['simulation'],cal)
        _,_,accfine,targetfine = simulate_forced(cfg['simulation'],cal,dt=.0005)
        np.testing.assert_array_equal(acc,accfine)
        delta = float(np.abs(np.rad2deg(target-targetfine)).max())
        if delta > 5e-5: raise AssertionError(('integrator',delta))
        checks.append(dict(kind='simulation_convergence',fold=fold,max_deg=delta))
        raw = build_packet(packet['theta'],packet['x'],packet['reference'],packet['length_m'],packet['future_reference'],cal)
        pred,_ = predictor.forecast(raw,prescribed_acceleration=acc)
        # Commands are placeholders in this angular diagnostic: command-lift results
        # have no interpretable actuator contract and are deliberately not scored.
        excluded = [n for n,spec in predictor.manifest.items() if spec.get('input_mode')=='command']
        pred = {n:y for n,y in pred.items() if n not in excluded}
        one = np.flatnonzero(meta.future_acceleration_gain.to_numpy()==1.)
        other = np.flatnonzero(meta.future_acceleration_gain.to_numpy()==1.15)
        for value in packet.values(): np.testing.assert_array_equal(value[one],value[other])
        delta = max(float(np.abs(np.rad2deg(y[one]-y[other])).max()) for y in pred.values())
        if delta > 5e-5: raise AssertionError(('future_gain_invariance',delta))
        checks.append(dict(kind='future_gain_invariance',fold=fold,max_deg=delta))
        for name,y in pred.items():
            error = np.abs(np.rad2deg(y[:,HORIZON_INDEX]-target[:,HORIZON_INDEX]))
            for j,h in enumerate(HORIZONS_S):
                frame = meta.copy(); frame['fold']=fold; frame['model']=name; frame['horizon_s']=h; frame['absolute_error_deg']=error[:,j]; simulation.append(frame)
        meta.to_csv(directory/'simulation_cases.csv',index=False)
        np.savez_compressed(directory/'simulation_predictions.npz',target=target,acceleration=acc,**packet,**pred)
        write(directory/'simulation_scope.json',{'excluded_command_models':excluded,'reason':'No actuator/reference generator; zero references are placeholders','angular_methods':list(pred)})
        print('SIMULATION',fold,len(meta),flush=True)
    pd.DataFrame(metrics).to_csv(out/f'{stage}_metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(out/f'{stage}_timings.csv',index=False)
    write(out/f'{stage}_verification.json',dict(passed=True,checks=checks,tolerance_deg=5e-5))
    if stage=='train':
        common = [n for n in selection[cfg['folds'][0]]['eligible'] if all(n in selection[f]['eligible'] for f in cfg['folds'])]
        chosen = min(common,key=lambda n:np.mean([selection[f]['validation'][n]['all']/selection[f]['validation']['incumbent']['all'] for f in cfg['folds']]))
        write(out/'selection.json',dict(folds=selection,common=dict(model=chosen,eligible=common),uses_evaluation_this_stage=False,uses_simulation=False,history_samples=41,real_records_reused=True))
        write(out/'fitting.json',fitting)
        checkpoints = [p for folder in cfg['folds'] for p in (out/folder).glob('*') if p.suffix in ['.npz','.json'] and not p.name.startswith('predictions')]
        prov.update(status='selection_complete',selection_sha256=sha256(out/'selection.json'),checkpoint_hashes={str(p.relative_to(out)):sha256(p) for p in checkpoints},selection_locked_utc=pd.Timestamp.now(tz='UTC').isoformat())
    else:
        pd.concat(simulation).to_csv(out/'simulation_metrics.csv',index=False)
        prov.update(status='complete',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),result_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
    write(out/'provenance.json',prov); print('COMPLETE',stage,flush=True)


if __name__=='__main__': main()
