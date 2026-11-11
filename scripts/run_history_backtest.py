"""Select causal within-history backtest strategies, then evaluate every frozen method."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from cranepilot.data import load_dataset,sha256
from cranepilot.benchmark import environment_record
from cranepilot.rollout import command_windows,HORIZON_INDEX,HORIZONS_S
from scripts.predict_cranepilot import build_packet
from scripts.history_backtest_candidate import HistoryBacktestCandidate
from scripts.forced_adaptive_simulator import simulate_forced
from scripts.run_broad_search import scores

CONTROLS = ['incumbent', 'soft_symmetry', 'combined', 'physical', 'adaptive', 'adaptive_fine']
METHODS = CONTROLS + ['bt21', 'bt31', 'bt_multi', 'bt_guarded',
                      'history_residual_transfer', 'history_residual_gate', 'history_soft_gate']


def check_predictions(pred, count):
    if set(pred) != set(METHODS):
        raise ValueError(f'Expected the fixed13 methods, got {sorted(pred)}')
    for name, values in pred.items():
        if values.shape != (count, 30) or not np.isfinite(values).all():
            raise ValueError(f'Invalid forecast {name}')


def write(p,obj): p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def verify(mapping,base=Path('.')):
    for name,digest in mapping.items():
        if sha256(base/name.replace('\\','/'))!=digest: raise ValueError(f'Changed dependency: {name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['select','evaluate'],required=True)
    parser.add_argument('--output',type=Path,default=Path('artifacts/runs/history_backtest_20261107'))
    args=parser.parse_args();out=args.output
    cfgpath=Path('configs/history_backtest_20261107.json');cfg=json.loads(cfgpath.read_text())
    parent=Path(cfg['parent_run']); prior=json.loads((parent/'provenance.json').read_text())
    if prior['status']!='complete': raise ValueError('Completed parent required')
    verify(prior['guarded_hashes']);verify(prior['result_hashes'],parent)
    locked=[cfgpath,Path('docs/HISTORY_BACKTEST_DESIGN_20261107.md'),Path('scripts/run_history_backtest.py'),
            *Path('scripts').glob('history_backtest_*.py'),*Path('tests').glob('test_history_backtest*.py')]
    if args.stage=='select':
        for p in locked:
            if subprocess.check_output(['git','show',f'HEAD:{p.as_posix()}'])!=p.read_bytes(): raise ValueError(f'Commit first: {p}')
        if out.exists(): raise FileExistsError('Preserve prior attempt')
        out.mkdir(parents=True)
        hashes=dict(prior['guarded_hashes']);hashes.update({str(p):sha256(p) for p in [*locked,parent/'provenance.json']})
        prov={'status':'selecting','config':cfg,'guarded_hashes':hashes,'environment':environment_record(),
              'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),'neural_training':False,'real_records_retrospective':True,
              'history_budget_samples':41,'within_history_heldout_theta':True,
              'history_backtest_conditions_on_observed_trolley_positions':True}
        write(out/'provenance.json',prov)
    else:
        prov=json.loads((out/'provenance.json').read_text())
        if prov['status']!='selection_complete': raise ValueError('Frozen selection required')
        verify(prov['guarded_hashes'])
        if sha256(out/'selection.json')!=prov['selection_sha256']: raise ValueError('Selection changed')
    try: run(args.stage,out,cfg,prov)
    except Exception as e:
        prov.update(status='failed',failure=f'{type(e).__name__}: {e}');write(out/'provenance.json',prov);raise


def run(stage,out,cfg,prov):
    torch.set_num_threads(4)
    old=json.loads((Path(cfg['source_run'])/'provenance.json').read_text());data=load_dataset(Path('data/processed'))
    selection={};metrics=[];checks=[];diagnostics=[];simmetrics=[];timings=[]
    frozen=json.loads((out/'selection.json').read_text()) if stage=='evaluate' else None
    for fold in cfg['folds']:
        predictor=HistoryBacktestCandidate(cfg,fold);cal=predictor.calibration;split=old['config']['folds'][fold]
        directory=out/fold
        if stage=='select': directory.mkdir()
        lengths=[float(data[s].signals['L'][0]) for s in split['train']]
        groups=[('validation',split['validation'])] if stage=='select' else [('evaluation',split['evaluation']),('load_stress',split['load_stress'])]
        for group,sids in groups:
            for sid in sids:
                s=data[sid];k=np.arange(40,len(s.time)-150,cfg['evaluation_stride']);raw=command_windows(s,cal,k)
                start=time.perf_counter();pred,meta,acc=predictor.forecast(raw);elapsed=time.perf_counter()-start
                check_predictions(pred,len(k))
                timings.append({'fold':fold,'scenario':sid,'packets':len(k),'all_variants_seconds':elapsed})
                with np.load(Path(cfg['parent_run'])/fold/f'predictions_s{sid}.npz') as retained:
                    np.testing.assert_array_equal(k,retained['k'])
                    np.testing.assert_array_equal(raw['target_q'],retained['target'])
                    for name in CONTROLS:
                        delta=float(np.abs(np.rad2deg(pred[name]-retained[name])).max())
                        if delta>5e-5: raise AssertionError(('parent_replay',fold,sid,name,delta))
                        checks.append({'kind':'parent_replay','fold':fold,'sid':sid,'model':name,'max_deg':delta})
                active=np.abs(raw['initial_velocity'])>=cfg['active_threshold']
                if stage=='select':
                    val={name:scores(y,raw['target_q'],active) for name,y in pred.items()}
                    pool=[name for name in pred if all(val[name][reg]<=val['incumbent'][reg]*cfg['selection'][reg+'_max_ratio']+1e-12 for reg in ['all','active','quiet'])]
                    chosen=min(pool,key=lambda name:val[name]['all'])
                    selection[fold]={'model':chosen,'eligible':pool,'validation':val}
                    # Same raw packet before/after target poisoning, and independent packet assembly.
                    ix=np.array([0,len(k)//2,len(k)-1]);small={key:v[ix].copy() for key,v in raw.items()}
                    clean,_,_=predictor.forecast(small);small['target_q'][:]=999;small['target_x'][:]=-999
                    changed,_,_=predictor.forecast(small)
                    for name in clean: np.testing.assert_array_equal(clean[name],changed[name])
                    past=k[ix,None]-np.arange(41)
                    rebuilt=build_packet(s.signals['theta'][past],s.signals['x'][past],s.signals['xr'][past],
                        s.signals['L'][k[ix]],s.signals['xr'][k[ix,None]+np.arange(30)*5],cal)
                    independent,_,_=predictor.forecast(rebuilt)
                    delta=max(float(np.abs(np.rad2deg(independent[n]-clean[n])).max()) for n in clean)
                    if delta>5e-5: raise AssertionError(('history_only',delta))
                    checks.append({'kind':'history_only_target_invariance','fold':fold,'max_deg':delta})
                    print('SELECT',fold,chosen,val[chosen],flush=True)
                else:
                    pred['selected']=pred[frozen['folds'][fold]['model']]
                    pred['common']=pred[frozen['common']['model']]
                extra=float(s.signals['L'][0])<min(lengths) or float(s.signals['L'][0])>max(lengths)
                for name,y in pred.items():
                    error=np.abs(np.rad2deg(y[:,HORIZON_INDEX]-raw['target_q'][:,HORIZON_INDEX]))
                    for regime,mask in [('all',np.ones(len(k),bool)),('active',active),('quiet',~active)]:
                        if not mask.any():continue
                        for j,h in enumerate(HORIZONS_S): metrics.append({'fold':fold,'group':group,'scenario':sid,'extrapolation':extra,
                            'model':name,'regime':regime,'horizon_s':h,'mae_deg':float(error[mask,j].mean())})
                for name,m in meta.items():
                    frame=pd.DataFrame(m);frame['fold']=fold;frame['model']=name;frame['group']=group;frame['scenario']=sid;frame['k']=k
                    diagnostics.append(frame)
                np.savez_compressed(directory/f'predictions_s{sid}.npz',k=k,target=raw['target_q'],acceleration=acc,**pred)
                print('REAL',stage,fold,sid,round(elapsed,2),'seconds',flush=True)
        if stage=='select':continue
        meta,packet,acc,target=simulate_forced(cfg['simulation'],cal)
        _,_,accfine,targetfine=simulate_forced(cfg['simulation'],cal,dt=.0005)
        np.testing.assert_array_equal(acc,accfine)
        delta=float(np.abs(np.rad2deg(target-targetfine)).max())
        if delta>5e-5:raise AssertionError(('integrator',delta))
        checks.append({'kind':'integrator','fold':fold,'max_deg':delta})
        raw=build_packet(packet['theta'],packet['x'],packet['reference'],packet['length_m'],packet['future_reference'],cal)
        pred,estimates,_=predictor.forecast(raw,prescribed_acceleration=acc)
        check_predictions(pred,len(meta))
        if len(meta)!=1152:raise AssertionError('Expected1152 synthetic observation cases per fold')
        pred['selected']=pred[frozen['folds'][fold]['model']];pred['common']=pred[frozen['common']['model']]
        gain1=np.flatnonzero(meta.future_acceleration_gain.to_numpy()==1.);gain2=np.flatnonzero(meta.future_acceleration_gain.to_numpy()==1.15)
        for v in packet.values():np.testing.assert_array_equal(v[gain1],v[gain2])
        np.testing.assert_array_equal(acc[gain1],acc[gain2])
        delta=max(float(np.abs(np.rad2deg(y[gain1]-y[gain2])).max()) for y in pred.values())
        if delta>5e-5:raise AssertionError(('future_gain_leak',delta))
        checks.append({'kind':'future_gain_invariance','fold':fold,'max_deg':delta})
        for name,y in pred.items():
            error=np.abs(np.rad2deg(y[:,HORIZON_INDEX]-target[:,HORIZON_INDEX]))
            for j,h in enumerate(HORIZONS_S):
                frame=meta.copy();frame['fold']=fold;frame['model']=name;frame['horizon_s']=h;frame['absolute_error_deg']=error[:,j];simmetrics.append(frame)
        for name,values in estimates.items():
            frame=pd.DataFrame(values);frame['case']=np.arange(len(meta));frame['fold']=fold;frame['model']=name
            frame.to_csv(directory/f'simulation_estimates_{name}.csv',index=False)
        meta.to_csv(directory/'simulation_cases.csv',index=False)
        np.savez_compressed(directory/'simulation_predictions.npz',target=target,acceleration=acc,**packet,**pred)
        print('SIMULATION',fold,len(meta),flush=True)
    pd.DataFrame(metrics).to_csv(out/f'{stage}_metrics.csv',index=False)
    pd.concat(diagnostics).to_csv(out/f'{stage}_parameter_estimates.csv',index=False)
    pd.DataFrame(timings).to_csv(out/f'{stage}_timings.csv',index=False)
    write(out/f'{stage}_verification.json',{'passed':True,'checks':checks,'tolerance_deg':5e-5})
    if stage=='select':
        common=[name for name in selection[cfg['folds'][0]]['eligible'] if all(name in selection[f]['eligible'] for f in cfg['folds'])]
        chosen=min(common,key=lambda name:np.mean([selection[f]['validation'][name]['all']/selection[f]['validation']['incumbent']['all'] for f in cfg['folds']]))
        write(out/'selection.json',{'folds':selection,'common':{'model':chosen,'eligible':common},'uses_new_simulation':False,'uses_real_evaluation_this_stage':False,'real_history_reused':True,'history_budget_samples':41,
            'within_history_backtest_uses_only_current_packet':True})
        prov.update(status='selection_complete',selection_sha256=sha256(out/'selection.json'),selection_locked_utc=pd.Timestamp.now(tz='UTC').isoformat())
    else:
        pd.concat(simmetrics).to_csv(out/'simulation_metrics.csv',index=False)
        prov.update(status='complete',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),result_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
    write(out/'provenance.json',prov)
    print('COMPLETE',stage,flush=True)


if __name__=='__main__':main()
