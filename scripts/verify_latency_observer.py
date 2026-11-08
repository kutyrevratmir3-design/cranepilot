"""Rebuild selected TRAIN fits, all VAL scores, and portable known-age replay."""
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.latency_pilot import FrozenLatencyPilot
from scripts import latency_observer as lo
from scripts import drive_conditioned as dc
from scripts.modal_history import load_allowed,concatenate
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_latency_observer import RUN,OUT,setup,case_data,add_rows,balanced
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

def same(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    return float(np.max(np.abs(a-b))) if np.asarray(a).size and np.asarray(a).dtype.kind!='b' else 0.

def compare_rows(rows,actual):
    keys=['case','scenario','role','group','method','horizon_s','age_ms','noise_quantum'];pd.testing.assert_frame_equal(pd.DataFrame(rows).sort_values(keys).reset_index(drop=True),actual.sort_values(keys).reset_index(drop=True),check_exact=False,check_dtype=False,rtol=1e-12,atol=1e-12)

def model_names(cfg):return [f'arx_{representation}_{form}_{alpha:g}.npz' for representation in cfg['representations'] for form in cfg['forms'] for alpha in cfg['ridge_grid']]

def development():
    dest=OUT/'development_verification.json'
    if dest.exists():raise FileExistsError('Retain prior verification')
    r=dict(passed=False,source_sha256=sha256(Path(__file__)),checks=[],independent_confirmation=False,final_goal_achieved=False)
    try:
        cfg,qc,pilot,task=setup();p=json.loads((RUN/'development.json').read_text());selection=json.loads((RUN/'selection.json').read_text());assert p['status']=='complete' and not p['old_evaluation_read_during_development'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);index=0;rows=[];scores={};positive={};refit=[];grid=pd.read_csv(RUN/'arx_grid.csv')
        for family,fold in task['families'].items():
            train=load_allowed(task,fold['train'],'train');val=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';case=selection['cases'][name];base=RUN/name;cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);model=dc.load(base/case['augmented_file']);parts=[];ys=[];bs=[];drive=[]
                for sid,age,noise,current,target,labels,raw,packet,dated in case_data(train,axis,family,task,qc,cal,cfg,index,cfg['train_ages_raw']):parts.append(packet);ys.append(target);bs.append(dated);drive.append(labels['drive_active'])
                tr=concatenate(parts);yt=np.concatenate(ys);bt=np.concatenate(bs);weights=dc.balanced_weights(tr['scenario'],np.concatenate(drive));X=lo.arx_features(tr,model['representation']);base_y=np.repeat(tr['theta'][:,:1],7,axis=1) if model['form']=='direct' else bt;rebuild=dc.arx_fit(lo.fit_stats(X,yt,base_y,weights,model['representation'],model['form']),model['alpha'])
                for key in ['mean','scale','coef']:np.testing.assert_allclose(rebuild[key],model[key],rtol=1e-12,atol=1e-12)
                refit.append(dict(case=name,file=case['augmented_file'],train_windows=len(yt)));del tr,yt,bt,parts,ys,bs,drive,X
                models={filename:dc.load(base/filename) for filename in model_names(cfg)};reference_scores={filename:[] for filename in models};c_scores={method:[] for method in cfg['methods']};c_positive={method:[] for method in cfg['methods']}
                for sid,age,noise,current,target,labels,raw,packet,dated in case_data(val,axis,family,task,qc,cal,cfg,index,cfg['train_ages_raw']):
                    cache={};pred={};maximum=0.
                    with np.load(base/f'validation_s{sid}_a{age}_n{noise:g}.npz') as z:
                        np.testing.assert_array_equal(z['k'],current['k'])
                        for key,value in labels.items():np.testing.assert_array_equal(value,z[key])
                        for method in cfg['methods']:
                            if age==0 and cache:cache[method]=next(iter(cache.values()))
                            else:cache[method]=lo.predict(raw,age,axis,method,pilot,name)
                            y,d=cache[method];pred[method]=y;maximum=max(maximum,same(y,z[method]),same(d['weights'],z['weight_'+method]));score=balanced(y,target,labels['drive_active']);c_scores[method].append(score)
                            if age>0:c_positive[method].append(score)
                            if age>0 and method!='unaware':np.testing.assert_array_equal(d['valid_heads'],[1,1,1,1,1,0])
                            for expert in ['frozen_arx','active_arx']:
                                key=method+'__'+expert;maximum=max(maximum,same(d[expert],z['ref_'+key]));reference_scores.setdefault(key,[]).append(balanced(d[expert],target,labels['drive_active']))
                        if age==0:
                            actual=pilot.forecast(raw,family=family,axis=axis)['forecast_rad']
                            for y in pred.values():np.testing.assert_array_equal(y,actual)
                    for filename,m in models.items():reference_scores[filename].append(balanced(lo.arx_prediction(m,packet,dated),target,labels['drive_active']))
                    choice=case['reference'];ref=lo.arx_prediction(models[choice['file']],packet,dated) if choice['type']=='augmented' else cache[choice['method']][1][choice['expert']];aug=lo.arx_prediction(models[case['augmented_file']],packet,dated)
                    with np.load(base/f'reference_validation_s{sid}_a{age}_n{noise:g}.npz') as z:
                        for key,value in dict(matched_arx=ref,augmented_arx=aug,dated_modal=dated).items():maximum=max(maximum,same(value,z[key]))
                    pred.update(matched_arx=ref,augmented_arx=aug,dated_modal=dated);add_rows(rows,pred,target,labels,name,sid,'validation',task,qc,age,noise);r['checks'].append(dict(case=name,scenario=int(sid),age_raw=age,noise=noise,windows=len(target),methods=len(pred),maximum_error_rad=maximum));print('VERIFIED_VAL',name,age,noise,maximum,flush=True)
                means_ref={key:float(np.mean(value)) for key,value in reference_scores.items()};best_aug=min(models,key=lambda key:means_ref[key]);best=min(means_ref,key=means_ref.get);assert best_aug==case['augmented_file']
                if choice['type']=='augmented':assert best==choice['file']
                else:assert best==choice['method']+'__'+choice['expert']
                for key,value in means_ref.items():np.testing.assert_allclose(value,case['validation_reference_scores'][key],rtol=1e-12,atol=1e-12)
                for row in grid[grid['case']==name].itertuples():np.testing.assert_allclose(row.validation_mae_deg,means_ref[row.file],rtol=1e-12,atol=1e-12)
                scores[name]={key:float(np.mean(value)) for key,value in c_scores.items()};scores[name].update(matched_arx=means_ref[best],augmented_arx=means_ref[best_aug]);positive[name]={key:float(np.mean(value)) for key,value in c_positive.items()};index+=1
        means={key:float(np.mean([c[key] for c in scores.values()])) for key in next(iter(scores.values()))};pos={key:float(np.mean([c[key] for c in positive.values()])) for key in cfg['methods']};chosen=min(cfg['methods'],key=lambda key:means[key]);wins=sum(c[chosen]<c['matched_arx'] for c in scores.values());g=cfg['development_gate'];passed=pos[chosen]<=g['maximum_unaware_positive_age_ratio']*pos['unaware'] and pos[chosen]<=g['maximum_hold_positive_age_ratio']*pos['hold'] and means[chosen]<=g['maximum_matched_arx_ratio']*means['matched_arx'] and wins>=g['minimum_arx_case_wins'];assert chosen==selection['selected'] and bool(passed)==selection['development_gate_passed'] and wins==selection['arx_case_wins'];assert len(grid)==selection['arx_fits']==120
        for key,value in means.items():np.testing.assert_allclose(value,selection['validation_means'][key],rtol=1e-12,atol=1e-12)
        compare_rows(rows,pd.read_csv(RUN/'validation_metrics.csv'));r.update(passed=True,selected=chosen,gate_reproduced=bool(passed),selected_arx_train_refits=refit,all_120_arx_validation_scores_recomputed=True,age_zero_identity_verified=True,missing_head_truth_excluded=True,forecast_comparisons=sum(x['windows']*x['methods'] for x in r['checks']),development_sha256=sha256(RUN/'development.json'))
    except Exception as error:r['failure']=f'{type(error).__name__}: {error}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

def full():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain prior verification')
    r=dict(passed=False,source_sha256=sha256(Path(__file__)),checks=[],independent_confirmation=False,final_goal_achieved=False)
    try:
        cfg,qc,pilot,task=setup();p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());s=json.loads((RUN/'selection.json').read_text());dev=json.loads((OUT/'development_verification.json').read_text());assert p['status']==q['status']=='complete' and dev['passed'] and dev['development_sha256']==sha256(RUN/'development.json') and q['selection_sha256']==sha256(RUN/'selection.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');r['original_files']=original_files();assert r['original_files']['passed'];root=Path('artifacts/latency_bundle_20261108').resolve();np_load=np.load
        def restricted(path,*args,**kwargs):
            if not Path(path).resolve().is_relative_to(root):raise AssertionError('Portable inference accessed dataset or ancestral runs')
            return np_load(path,*args,**kwargs)
        with patch('numpy.load',restricted):api=FrozenLatencyPilot(root)
        rows=[];index=0
        for family,fold in task['families'].items():
            records=load_confirmation(task,fold['confirmation']);validation=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';case=s['cases'][name];cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);aug=dc.load(RUN/name/case['augmented_file'])
                for role,source,ages in [('validation',validation,cfg['train_ages_raw']),('old_evaluation',records,cfg['evaluation_ages_raw'])]:
                    for sid,age,noise,current,target,labels,raw,packet,dated in case_data(source,axis,family,task,qc,cal,cfg,index,ages):
                        with patch('numpy.load',restricted):result=api.forecast(raw,family=family,axis=axis,angle_age_samples=age,include_reference=True)
                        expected_valid=np.ones(6,dtype=bool) if age==0 or s['selected']=='unaware' else np.array([1,1,1,1,1,0],dtype=bool);np.testing.assert_array_equal(result['backtest_valid'],expected_valid);maximum=0.
                        if role=='validation':
                            with np.load(RUN/name/f'validation_s{sid}_a{age}_n{noise:g}.npz') as z:maximum=max(maximum,same(result['forecast_rad'],z[s['selected']]))
                            with np.load(RUN/name/f'reference_validation_s{sid}_a{age}_n{noise:g}.npz') as z:maximum=max(maximum,same(result['matched_arx_rad'],z['matched_arx']))
                            methods=2
                        else:
                            cache={s['selected']:(result['forecast_rad'],None)}
                            for method in ['hold','unaware']:
                                if method not in cache:cache[method]=lo.predict(raw,age,axis,method,pilot,name)
                            pred=dict(selected=result['forecast_rad'],timestamp_hold=cache['hold'][0],unaware=cache['unaware'][0],matched_arx=result['matched_arx_rad'],augmented_arx=lo.arx_prediction(aug,packet,dated),dated_modal=dated);methods=len(pred)
                            with np.load(RUN/'retrospective'/f'{name}_s{sid}_a{age}_n{noise:g}.npz') as z:
                                np.testing.assert_array_equal(z['k'],current['k'])
                                for key,value in labels.items():np.testing.assert_array_equal(value,z[key])
                                for key,value in pred.items():maximum=max(maximum,same(value,z[key]))
                                same(result['weights'],z['diagnostic_weights']);same(result['head_estimate_rad'],z['diagnostic_head']);np.testing.assert_array_equal(result['backtest_valid'],z['diagnostic_valid_heads'])
                            add_rows(rows,pred,target,labels,name,sid,role,task,qc,age,noise)
                        if age==0:np.testing.assert_array_equal(result['forecast_rad'],pilot.forecast(raw,family=family,axis=axis)['forecast_rad'])
                        r['checks'].append(dict(case=name,scenario=int(sid),role=role,age_raw=age,noise=noise,windows=len(target),methods=methods,maximum_error_rad=maximum));print('VERIFIED',role,name,sid,age,noise,maximum,flush=True)
                index+=1
        compare_rows(rows,pd.read_csv(RUN/'retrospective/metrics.csv'));probes=[];timings=[]
        for name,case in s['cases'].items():
            family=case['family'];axis=case['axis'];coords=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if k=='x' else 'Jr' for k in coords]
            for age in [0,5,15]:
                raw={key:np.zeros((1,401)) for key in [axis,'L']+coords+refs};raw['L'][:]=.4;result=api.forecast(raw,family=family,axis=axis,angle_age_samples=age);probes.append(dict(case=name,age_raw=age,bias_deg=float(np.abs(np.rad2deg(result['forecast_rad'])).max())))
            samples=[]
            for _ in range(3):
                start=time.perf_counter();api.forecast(raw,family=family,axis=axis,angle_age_samples=5);samples.append((time.perf_counter()-start)*1000)
            timings.append(dict(case=name,age_raw=5,repeats=3,median_ms=float(np.median(samples))))
        r.update(passed=True,forecast_comparisons=sum(x['windows']*x['methods'] for x in r['checks']),development_sha256=sha256(RUN/'development.json'),retrospective_sha256=sha256(RUN/'retrospective/provenance.json'),development_verification_sha256=sha256(OUT/'development_verification.json'),bundle_manifest_sha256=sha256(root/'manifest.json'),portable_inference_no_raw_or_parent_runs=True,all_retrospective_metrics_recomputed=True,zero_probes=probes,maximum_zero_bias_deg=max(x['bias_deg'] for x in probes),cpu_timing=timings,age_zero_identity_verified=True)
    except Exception as error:r['failure']=f'{type(error).__name__}: {error}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--development-only',action='store_true');development() if parser.parse_args().development_only else full()
