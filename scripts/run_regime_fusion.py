"""Freeze validation routing, then replay opened records and locked synthetic stresses."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256,QUANTIZATION_RAD
from cranepilot.benchmark import environment_record
from cranepilot.physics import LengthCalibration
from cranepilot.reliability_pilot import FrozenReliabilityPilot
from scripts import drive_conditioned as dc
from scripts.regime_fusion import context,expert_inputs,fuse
from scripts.modal_history import load_allowed
from scripts.active_qualification import qualification_packets
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_drive_conditioned import metric_rows,score_groups
from scripts.run_external_benchmark import write,verify_hashes

CONFIG=Path('configs/regime_fusion_20261107.json');RUN=Path('artifacts/runs/regime_fusion_20261107');OUT=Path('results/regime_fusion_20261107')
LOCKED=[CONFIG,Path('docs/REGIME_FUSION_DESIGN_20261107.md'),Path('scripts/regime_fusion.py'),Path('scripts/run_regime_fusion.py'),Path('src/cranepilot/regime_pilot.py'),Path('tests/test_regime_fusion.py')]

def setup():
    cfg=json.loads(CONFIG.read_text());qc=json.loads(Path(cfg['qualification']).read_text());parent=Path(cfg['parent']);dev=json.loads((parent/'development.json').read_text());assert dev['status']=='complete';verify_hashes(dev['guarded_hashes']);verify_hashes(dev['result_hashes'],parent);selection=json.loads((parent/'selection.json').read_text());assert selection['selected']=='blend_linear_adapt';pilot=FrozenReliabilityPilot(cfg['bundle']);models={name:dict(reference=c['reference'],long=dc.load(parent/name/c['long_model']),short=dc.load(parent/name/'short_arx.npz'),prior=dc.load(parent/name/'prior.npz')) for name,c in selection['cases'].items()};return cfg,qc,pilot,pilot.cfg,models,selection

def packet(s,axis,task,qc,cal):
    p,target,labels=qualification_packets(s,axis,task,qc,cal);raw=p['k'][:,None]-np.arange(401);history={key:s.signals[key][raw] for key in ['x','J'] if key in s.signals};ratio=context(history,task['past_position_range']);np.testing.assert_array_equal(ratio>1,labels['drive_active']);return p,target,labels,ratio

def perturb(p,condition,seed,sid,case_index):
    out={k:v.copy() for k,v in p.items()};rng=np.random.default_rng(np.random.SeedSequence([seed,sid,case_index,['angle_jitter_half_quantum','angle_jitter_one_quantum','angle_lag_50ms'].index(condition)]))
    if condition.startswith('angle_jitter_'):
        scale=QUANTIZATION_RAD*(.5 if condition=='angle_jitter_half_quantum' else 1);out['theta']+=rng.uniform(-scale,scale,(len(out['theta']),401))[:,::5]
    elif condition=='angle_lag_50ms':out['theta']=np.concatenate([out['theta'][:,1:],out['theta'][:,-1:]],axis=1)
    else:raise ValueError('Unknown synthetic stress')
    return out

def forecasts(e,cfg,selected=None):
    preds={};weights={}
    for key in cfg['variants']:preds[key],weights['weight_'+key]=fuse(e,key)
    preds.update(frozen_arx=e['forecasts'][:,:,0],active_arx=e['forecasts'][:,:,1],free_modal=e['forecasts'][:,:,2],forced_modal=e['forecasts'][:,:,3])
    if selected is not None:preds['selected']=preds[selected].copy()
    return preds,weights

def develop():
    cfg,qc,pilot,task,models,parent_selection=setup()
    for f in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{f}'])!=f.read_bytes():raise ValueError(f'Commit before selection {f}')
    if RUN.exists():raise FileExistsError('Retain prior development')
    RUN.mkdir(parents=True);sources=set(pilot.manifest['source_hashes'])|set(pilot.reliability_manifest['source_hashes']);paths=LOCKED+[Path(cfg['qualification']),Path('scripts/drive_conditioned.py'),Path('scripts/active_qualification.py'),Path('scripts/run_drive_conditioned.py')]+[Path(x) for x in sources]+[f for root in [Path(cfg['parent']),Path(cfg['bundle'])] for f in root.rglob('*') if f.is_file()];p=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,qualification=qc,task=task,guarded_hashes={str(f):sha256(f) for f in paths},environment=environment_record(),new_model_fitting=False,old_evaluation_read_in_development=False,independent_confirmation=False,final_goal_achieved=False);write(RUN/'development.json',p);rows=[];scores={};feasible={key:True for key in cfg['variants']}
    try:
        for family,fold in task['families'].items():
            records=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.cases[name]['calibration']);scores[name]={}
                for sid,s in records.items():
                    part,target,labels,ratio=packet(s,axis,task,qc,cal);e=expert_inputs(part,ratio,pilot,name,models[name]);pred,weights=forecasts(e,cfg);quiet=ratio<=1
                    # Parent parity before any routing interpretation.
                    with np.load(Path(cfg['parent'])/name/'validation.npz') as z:
                        np.testing.assert_array_equal(z['k'],part['k']);np.testing.assert_allclose(e['legacy'],z['old_blend'],rtol=0,atol=1e-12);np.testing.assert_allclose(e['active'],z['blend_linear_adapt'],rtol=0,atol=1e-12)
                    for key in cfg['variants']:feasible[key] &= np.array_equal(pred[key][quiet],e['legacy'][quiet])
                    scores[name]={key:score_groups(y,target,labels) for key,y in pred.items()};np.savez_compressed(RUN/f'{name}_s{sid}.npz',k=part['k'],**labels,**pred,**weights,**{'evidence_'+k:v for k,v in e.items()});rows+=metric_rows(pred,target,labels,name,sid,'validation',task,qc);print('VALIDATION',name,sid,{k:v['balanced'] for k,v in scores[name].items()},flush=True)
        means={key:float(np.mean([c[key]['balanced'] for c in scores.values()])) for key in next(iter(scores.values()))};selected=min([key for key in cfg['variants'] if feasible[key]],key=lambda key:means[key]);wins=sum(c[selected]['balanced']<c['legacy']['balanced'] for c in scores.values());gate=cfg['development_gate'];passed=means[selected]<=gate['maximum_legacy_ratio']*means['legacy'] and means[selected]<=gate['maximum_active_arx_ratio']*means['active_arx'] and wins>=gate['minimum_legacy_case_wins'];selection=dict(selected=selected,case_scores=scores,validation_means=means,quiet_feasible={k:bool(v) for k,v in feasible.items()},validation_legacy_change_percent=100*(means[selected]/means['legacy']-1),legacy_case_wins=int(wins),development_gate_passed=bool(passed),parent_cases=parent_selection['cases'],independent_confirmation=False,final_goal_achieved=False);write(RUN/'selection.json',selection);pd.DataFrame(rows).to_csv(RUN/'validation_metrics.csv',index=False);p.update(status='complete',selection_sha256=sha256(RUN/'selection.json'),result_hashes={str(f.relative_to(RUN)):sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name!='development.json'});print('SELECTED',selected,'MEANS',means,'FEASIBLE',feasible,'GATE',passed,flush=True)
    except Exception as error:p.update(status='failed',failure=f'{type(error).__name__}: {error}');raise
    finally:write(RUN/'development.json',p)

def evaluate():
    cfg,qc,pilot,task,models,_=setup();p=json.loads((RUN/'development.json').read_text());assert p['status']=='complete';verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN)
    for rel in ['development.json']+list(p['result_hashes']):
        f=OUT/'frozen_development'/rel
        if f.read_bytes()!=(RUN/rel).read_bytes() or subprocess.check_output(['git','show',f'HEAD:{f}'])!=f.read_bytes():raise ValueError('Freeze committed selection first')
    selection=json.loads((RUN/'selection.json').read_text());out=RUN/'retrospective'
    if out.exists():raise FileExistsError('Retain previous replay')
    out.mkdir();q=dict(status='running',selection_sha256=sha256(RUN/'selection.json'),development_sha256=sha256(RUN/'development.json'),conditions=['clean']+cfg['stress_conditions'],independent_confirmation=False,final_goal_achieved=False);write(out/'provenance.json',q);rows=[];case_index=0
    try:
        for family,fold in task['families'].items():
            records=load_confirmation(task,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.cases[name]['calibration'])
                for sid,s in records.items():
                    part,target,labels,ratio=packet(s,axis,task,qc,cal)
                    for condition in q['conditions']:
                        supplied=part if condition=='clean' else perturb(part,condition,cfg['stress_seed'],sid,case_index);e=expert_inputs(supplied,ratio,pilot,name,models[name]);pred,weights=forecasts(e,cfg,selection['selected']);np.savez_compressed(out/f'{condition}_{name}_s{sid}.npz',k=part['k'],**labels,**pred,**weights,**{'evidence_'+k:v for k,v in e.items()});current=metric_rows(pred,target,labels,name,sid,'old_evaluation',task,qc)
                        for row in current:row['condition']=condition
                        rows+=current;print('REPLAY',condition,name,sid,flush=True)
                case_index+=1
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);q.update(status='complete',result_hashes={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='provenance.json'})
    except Exception as error:q.update(status='failed',failure=f'{type(error).__name__}: {error}');raise
    finally:write(out/'provenance.json',q)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');evaluate() if parser.parse_args().evaluate else develop()
