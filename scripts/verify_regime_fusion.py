"""Raw-to-forecast replay, quiet constraint, VAL selection, portable API and stress QA."""
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from cranepilot.data import sha256,QUANTIZATION_RAD
from cranepilot.physics import LengthCalibration
from cranepilot.regime_pilot import FrozenRegimePilot
from scripts.regime_fusion import expert_inputs,fuse
from scripts.run_regime_fusion import RUN,OUT,setup,packet,perturb,forecasts
from scripts.run_drive_conditioned import score_groups,metric_rows
from scripts.modal_history import load_allowed
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

def recompute_selection(cfg,scores,feasible):
    means={key:float(np.mean([c[key]['balanced'] for c in scores.values()])) for key in next(iter(scores.values()))};selected=min([key for key in cfg['variants'] if feasible[key]],key=lambda key:means[key]);wins=sum(c[selected]['balanced']<c['legacy']['balanced'] for c in scores.values());g=cfg['development_gate'];passed=means[selected]<=g['maximum_legacy_ratio']*means['legacy'] and means[selected]<=g['maximum_active_arx_ratio']*means['active_arx'] and wins>=g['minimum_legacy_case_wins'];return means,selected,wins,bool(passed)

def api_evidence(result):
    return dict(forecasts=result['expert_forecasts_rad'],errors=result['prefix_errors_rad'],old_prior=result['old_prior'],new_prior=result['active_prior'],legacy=result['legacy_forecast_rad'],active=result['active_forecast_rad'],context=result['context_ratio'])

def raw_history(s,k,axis,family):
    coordinates=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if x=='x' else 'Jr' for x in coordinates];raw=k[:,None]-np.arange(401);return {key:s.signals[key][raw].copy() for key in ['L',axis]+coordinates+refs}

def stress_raw(history,axis,condition,cfg,sid,case_index):
    raw={k:v.copy() for k,v in history.items()}
    if condition=='clean':return raw
    if condition=='angle_lag_50ms':raw[axis]=np.concatenate([raw[axis][:,5:],np.repeat(raw[axis][:,-1:],5,axis=1)],axis=1)
    else:
        rng=np.random.default_rng(np.random.SeedSequence([cfg['stress_seed'],sid,case_index,cfg['stress_conditions'].index(condition)]));scale=QUANTIZATION_RAD*(.5 if condition=='angle_jitter_half_quantum' else 1);raw[axis]+=rng.uniform(-scale,scale,raw[axis].shape)
    return raw

def compare_metrics(rows,actual):
    expected=pd.DataFrame(rows);keys=['case','scenario','role','group','method','horizon_s']+(['condition'] if 'condition' in expected else []);pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),actual.sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)

def verify(development_only):
    dest=OUT/('development_verification.json' if development_only else 'verification.json')
    if dest.exists():raise FileExistsError('Retain all verification attempts')
    report=dict(passed=False,source_sha256=sha256(Path(__file__)),checks=[],independent_confirmation=False,final_goal_achieved=False)
    try:
        cfg,qc,pilot,task,models,_=setup();p=json.loads((RUN/'development.json').read_text());selection=json.loads((RUN/'selection.json').read_text());assert p['status']=='complete' and not p['new_model_fitting'] and not p['old_evaluation_read_in_development'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);rows=[];scores={};feasible={key:True for key in cfg['variants']};case_index=0
        if not development_only:
            q=json.loads((RUN/'retrospective/provenance.json').read_text());assert q['status']=='complete' and q['selection_sha256']==sha256(RUN/'selection.json');verify_hashes(q['result_hashes'],RUN/'retrospective');bundle=Path('artifacts/regime_bundle_20261107').resolve();original_load=np.load
            def restricted_load(path,*args,**kwargs):
                if not Path(path).resolve().is_relative_to(bundle):raise AssertionError('Portable inference attempted dataset/ancestral access')
                return original_load(path,*args,**kwargs)
            with patch('numpy.load',restricted_load):api=FrozenRegimePilot(bundle)
            report['original_files']=original_files();assert report['original_files']['passed']
        for family,fold in task['families'].items():
            validation=load_allowed(task,fold['validation'],'validation');roles=[('validation',validation)]
            if not development_only:roles.append(('old_evaluation',load_confirmation(task,fold['confirmation'])))
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.cases[name]['calibration']);scores[name]={}
                for role,records in roles:
                    for sid,s in records.items():
                        part,target,labels,ratio=packet(s,axis,task,qc,cal);conditions=['clean'] if role=='validation' else ['clean']+cfg['stress_conditions']
                        for condition in conditions:
                            file=RUN/f'{name}_s{sid}.npz' if role=='validation' else RUN/'retrospective'/f'{condition}_{name}_s{sid}.npz';supplied=part if condition=='clean' else perturb(part,condition,cfg['stress_seed'],sid,case_index)
                            if development_only:e=expert_inputs(supplied,ratio,pilot,name,models[name])
                            else:
                                raw=stress_raw(raw_history(s,part['k'],axis,family),axis,condition,cfg,sid,case_index);np.testing.assert_array_equal(raw[axis][:,::5],supplied['theta'])
                                with patch('numpy.load',restricted_load):result=api.forecast(raw,family=family,axis=axis)
                                e=api_evidence(result);np.testing.assert_array_equal(e['context'],ratio)
                            pred,weights=forecasts(e,cfg,None if role=='validation' else selection['selected']);maximum=0.
                            with np.load(file) as z:
                                np.testing.assert_array_equal(z['k'],part['k'])
                                for key,value in labels.items():np.testing.assert_array_equal(z[key],value)
                                for key,value in dict(**pred,**weights,**{'evidence_'+k:v for k,v in e.items()}).items():
                                    np.testing.assert_allclose(value,z[key],rtol=0,atol=1e-12);maximum=max(maximum,float(np.abs(value-z[key]).max()))
                            if not development_only:np.testing.assert_allclose(result['forecast_rad'],pred[selection['selected']] if role!='validation' else pred[selection['selected']],rtol=0,atol=1e-12)
                            current=metric_rows(pred,target,labels,name,sid,role,task,qc)
                            if not development_only:
                                for row in current:row['condition']=condition
                            rows+=current
                            if role=='validation':
                                scores[name]={key:score_groups(value,target,labels) for key,value in pred.items()}
                                for key in cfg['variants']:feasible[key] &= np.array_equal(pred[key][ratio<=1],e['legacy'][ratio<=1])
                            if not development_only and selection['quiet_feasible'][selection['selected']]:np.testing.assert_array_equal(result['forecast_rad'][ratio<=1],e['legacy'][ratio<=1])
                            report['checks'].append(dict(case=name,scenario=int(sid),role=role,condition=condition,windows=len(part['k']),methods=len(pred),quiet_windows=int((ratio<=1).sum()),maximum_replay_error_rad=maximum));print('VERIFIED',role,condition,name,sid,maximum,flush=True)
                case_index+=1
        means,chosen,wins,gate=recompute_selection(cfg,scores,feasible);assert chosen==selection['selected'] and wins==selection['legacy_case_wins'] and gate==selection['development_gate_passed'];assert {key:bool(value) for key,value in feasible.items()}==selection['quiet_feasible']
        for key,value in means.items():np.testing.assert_allclose(value,selection['validation_means'][key],rtol=1e-12,atol=1e-12)
        actual=pd.read_csv(RUN/'validation_metrics.csv')
        if not development_only:actual['condition']='clean';actual=pd.concat([actual,pd.read_csv(RUN/'retrospective/metrics.csv')],ignore_index=True)
        compare_metrics(rows,actual);report.update(passed=True,development_sha256=sha256(RUN/'development.json'),selected=chosen,development_gate_reproduced=gate,quiet_constraint_reproduced=True,forecast_comparisons=sum(x['windows']*x['methods'] for x in report['checks']),all_metrics_recomputed=True)
        if not development_only:
            probes=[];timing=[]
            for name,case in selection['parent_cases'].items():
                family=case['family'];axis=case['axis'];coords=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if x=='x' else 'Jr' for x in coords]
                for length in [.1,.4,.8]:
                    raw={key:np.zeros((1,401)) for key in [axis,'L']+coords+refs};raw['L'][:]=length
                    with patch('numpy.load',restricted_load):result=api.forecast(raw,family=family,axis=axis)
                    np.testing.assert_array_equal(result['forecast_rad'],result['legacy_forecast_rad']);probes.append(dict(case=name,length_m=length,maximum_bias_deg=float(np.abs(np.rad2deg(result['forecast_rad'])).max())))
                samples=[]
                for _ in range(5):
                    start=time.perf_counter();api.forecast(raw,family=family,axis=axis);samples.append((time.perf_counter()-start)*1000)
                timing.append(dict(case=name,packet_count=1,repeats=5,median_ms=float(np.median(samples)),minimum_ms=float(min(samples))))
            report.update(portable_api_requires_no_raw_or_parent_runs=True,synthetic_zero_probes=probes,maximum_zero_bias_deg=max(x['maximum_bias_deg'] for x in probes),single_packet_timing=timing,bundle_manifest_sha256=sha256(bundle/'manifest.json'),retrospective_sha256=sha256(RUN/'retrospective/provenance.json'))
    except Exception as error:report.update(passed=False,failure=f'{type(error).__name__}: {error}');raise
    finally:dest.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--development-only',action='store_true');verify(parser.parse_args().development_only)
