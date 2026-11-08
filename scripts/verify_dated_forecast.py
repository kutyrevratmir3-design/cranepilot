"""TRAIN refit, all VAL choices, raw/API replay and clock-availability audit."""
import argparse,json,time,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.dated_pilot import FrozenDatedPilot
from scripts.run_dated_forecast import RUN,OUT,setup,fit_case,reference_pool,candidates,choose,selection_from,nearest
from scripts.run_latency_observer import case_data,add_rows,balanced,current_packet,raw_history
from scripts.modal_history import load_allowed
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import write,verify_hashes
from scripts.verify_latency_observer import same,compare_rows,model_names
from scripts.verify_finetune_campaign import original_files
from scripts import dated_forecast as df,drive_conditioned as dc

def development():
    dest=OUT/'development_verification.json'
    if dest.exists():raise FileExistsError('Retain prior verification')
    r=dict(passed=False,checks=[],source_sha256=sha256(Path(__file__)),independent_confirmation=False)
    try:
        cfg,qc,pilot,task=setup();p=json.loads((RUN/'development.json').read_text());s=json.loads((RUN/'selection.json').read_text());assert p['status']=='complete' and not p['old_evaluation_read_during_development'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);index=0;scores={};refs={};aug={};rows=[];refits=[]
        for family,fold in task['families'].items():
            train=load_allowed(task,fold['train'],'train');val=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);new,n=fit_case(train,axis,family,task,qc,cal,cfg,index)
                for filename,m in new.items():
                    actual=dc.load(RUN/name/filename)
                    for key in ['mean','scale','coef']:np.testing.assert_allclose(m[key],actual[key],rtol=1e-12,atol=1e-12)
                refits.append(dict(case=name,train_windows=n,fits=len(new)));models=reference_pool(cfg,name,new);ref_scores={age:{} for age in cfg['train_ages_raw']};batches=[];per=[]
                for sid,age,noise,current,target,labels,raw,packet,dated in case_data(val,axis,family,task,qc,cal,cfg,index,cfg['train_ages_raw']):
                    pred,diag=df.predictions(raw,age,axis,pilot,name,cfg['methods']);ref=candidates(raw,age,axis,pilot,name,packet,dated,models,cfg);maximum=0.
                    with np.load(RUN/name/f'validation_s{sid}_a{age}_n{noise:g}.npz') as z:
                        np.testing.assert_array_equal(z['k'],current['k'])
                        for key,v in labels.items():np.testing.assert_array_equal(v,z[key])
                        for key,v in pred.items():maximum=max(maximum,same(v,z[key]))
                        for key,v in diag.items():maximum=max(maximum,same(np.asarray(v),z['diagnostic_'+key]))
                    if age==0:
                        original=pilot.forecast(raw,family=family,axis=axis)['forecast_rad']
                        for y in pred.values():np.testing.assert_array_equal(y,original)
                    else:assert len(diag['truth_indices'])==6 and np.min(diag['truth_indices'])>=0
                    for key,y in ref.items():ref_scores[age].setdefault(key,[]).append(balanced(y,target,labels['drive_active']))
                    for method,y in dict(pred,bare_dated_modal=dated).items():per.append(dict(method=method,age_raw=age,noise=noise,score=balanced(y,target,labels['drive_active'])))
                    add_rows(rows,dict(pred,bare_dated_modal=dated),target,labels,name,sid,'validation',task,qc,age,noise);batches.append((sid,age,noise,target,labels,ref));r['checks'].append(dict(case=name,age_raw=age,noise=noise,windows=len(target),methods=len(pred),maximum_error_rad=maximum))
                mean={a:{k:float(np.mean(v)) for k,v in pool.items()} for a,pool in ref_scores.items()};refs[name]=choose(mean);aug[name]=min(new,key=lambda k:np.mean([v['new/'+k] for v in mean.values()]))
                for sid,age,noise,target,labels,ref in batches:
                    pred=dict(matched_arx=ref[refs[name][str(age)]['key']],augmented_arx=ref['new/'+aug[name]])
                    with np.load(RUN/name/f'reference_validation_s{sid}_a{age}_n{noise:g}.npz') as z:
                        for key,v in pred.items():same(v,z[key])
                    for method,y in pred.items():per.append(dict(method=method,age_raw=age,noise=noise,score=balanced(y,target,labels['drive_active'])))
                    add_rows(rows,pred,target,labels,name,sid,'validation',task,qc,age,noise)
                scores[name]=per;index+=1;print('VERIFIED_VAL',name,flush=True)
        actual=selection_from(scores,refs,aug,cfg);assert actual==s;compare_rows(rows,pd.read_csv(RUN/'validation_metrics.csv'));grid=pd.read_csv(RUN/'arx_grid.csv')
        for row in grid.itertuples():np.testing.assert_allclose(row.validation_mae_deg,np.mean([v['scores']['new/'+row.file] for v in refs[row.case].values()]),rtol=1e-12,atol=1e-12)
        r.update(passed=True,development_sha256=sha256(RUN/'development.json'),all_120_train_fits_rebuilt=refits,all_30_conditional_reference_choices_reproduced=True,all_21_method_scores_recomputed=True,gate_reproduced=s['development_gate_passed'],age_zero_identity=True,all_validation_metrics_recomputed=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:write(dest,r)

def full():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain prior verification')
    r=dict(passed=False,checks=[],source_sha256=sha256(Path(__file__)),independent_confirmation=False,final_goal_achieved=False)
    try:
        cfg,qc,pilot,task=setup();p=json.loads((RUN/'development.json').read_text());s=json.loads((RUN/'selection.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());dev=json.loads((OUT/'development_verification.json').read_text());assert dev['passed'] and dev['development_sha256']==sha256(RUN/'development.json');assert p['status']==q['status']=='complete' and q['selection_sha256']==sha256(RUN/'selection.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');r['original_files']=original_files();assert r['original_files']['passed'];root=Path('artifacts/dated_bundle_20261108').resolve();np_load=np.load
        def restricted(path,*args,**kw):
            if not Path(path).resolve().is_relative_to(root):raise AssertionError('API read outside portable bundle')
            return np_load(path,*args,**kw)
        with patch('numpy.load',restricted):api=FrozenDatedPilot(root)
        rows=[];index=0;poison=[]
        for family,fold in task['families'].items():
            records=load_confirmation(task,fold['confirmation']);val=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration'])
                for role,source,ages in [('validation',val,cfg['train_ages_raw']),('old_evaluation',records,cfg['evaluation_ages_raw'])]:
                    for sid,age,noise,current,target,labels,raw,packet,dated in case_data(source,axis,family,task,qc,cal,cfg,index,ages):
                        with patch('numpy.load',restricted):result=api.forecast(raw,family=family,axis=axis,angle_age_samples=age,include_reference=True,include_candidates=True)
                        pred=dict(result['candidates']);pred.update(matched_arx=result['matched_arx_rad'],augmented_arx=result['augmented_arx_rad'],bare_dated_modal=result['bare_dated_modal_rad']);maximum=0.
                        file=RUN/name/f'validation_s{sid}_a{age}_n{noise:g}.npz' if role=='validation' else RUN/'retrospective'/f'{name}_s{sid}_a{age}_n{noise:g}.npz'
                        with np.load(file) as z:
                            np.testing.assert_array_equal(z['k'],current['k'])
                            for key,v in labels.items():np.testing.assert_array_equal(v,z[key])
                            for key,v in result['candidates'].items():maximum=max(maximum,same(v,z[key]))
                            for key,v in result['diagnostics'].items():maximum=max(maximum,same(np.asarray(v),z['diagnostic_'+key]))
                            if role=='old_evaluation':
                                maximum=max(maximum,same(result['forecast_rad'],z['selected']))
                                for key in ['matched_arx','augmented_arx','bare_dated_modal']:maximum=max(maximum,same(pred[key],z[key]))
                        if role=='validation':
                            with np.load(RUN/name/f'reference_validation_s{sid}_a{age}_n{noise:g}.npz') as z:
                                for key in ['matched_arx','augmented_arx']:maximum=max(maximum,same(pred[key],z[key]))
                        else:pred['selected']=result['forecast_rad'];add_rows(rows,pred,target,labels,name,sid,role,task,qc,age,noise)
                        if age==0:np.testing.assert_array_equal(result['forecast_rad'],pilot.forecast(raw,family=family,axis=axis)['forecast_rad'])
                        r['checks'].append(dict(role=role,case=name,scenario=int(sid),age_raw=age,noise=noise,windows=len(target),methods=len(pred),maximum_error_rad=maximum))
                # Mutate every unavailable selected-angle/future support sample in raw recordings.
                sid=next(iter(val));record=val[sid];current,_,_=current_packet(record,axis,task,qc,cal);k=current['k'][:1];cut=int(k[0])
                for age in [0,5,15,25]:
                    before=raw_history(record,k,axis,family,age,0,cfg,index);changed=copy.deepcopy(record)
                    for key,v in changed.signals.items():v[(cut-age+1 if key==axis else cut+1):]=1e6
                    after=raw_history(changed,k,axis,family,age,0,cfg,index)
                    for key in before:np.testing.assert_array_equal(before[key],after[key])
                    a=api.forecast(before,family=family,axis=axis,angle_age_samples=age)['forecast_rad'];b=api.forecast(after,family=family,axis=axis,angle_age_samples=age)['forecast_rad'];np.testing.assert_array_equal(a,b);poison.append(dict(case=name,age_raw=age,maximum_error_rad=0.))
                index+=1;print('VERIFIED_API',name,flush=True)
        compare_rows(rows,pd.read_csv(RUN/'retrospective/metrics.csv'));zero=[];timing=[]
        for name in s['cases']:
            family,axis=name.split('_');coords=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];raw={key:np.zeros((1,401)) for key in [axis,'L']+coords+['xr' if x=='x' else 'Jr' for x in coords]};raw['L'][:]=.4
            for age in [0,5,15,25]:
                y=api.forecast(raw,family=family,axis=axis,angle_age_samples=age)['forecast_rad'];zero.append(dict(case=name,age_raw=age,maximum_bias_deg=float(np.abs(np.rad2deg(y)).max())))
            samples=[]
            for _ in range(3):
                start=time.perf_counter();api.forecast(raw,family=family,axis=axis,angle_age_samples=15);samples.append((time.perf_counter()-start)*1000)
            timing.append(dict(case=name,age_raw=15,repeats=3,median_ms=float(np.median(samples))))
        r.update(passed=True,development_sha256=sha256(RUN/'development.json'),retrospective_sha256=sha256(RUN/'retrospective/provenance.json'),bundle_sha256=sha256(root/'manifest.json'),forecast_comparisons=sum(c['windows']*c['methods'] for c in r['checks']),all_retrospective_metrics_recomputed=True,portable_api_no_dataset_or_parent_runs=True,age_zero_identity=True,clock_poison_probes=poison,zero_probes=zero,cpu_timing=timing,maximum_forecast_error_rad=max(c['maximum_error_rad'] for c in r['checks']))
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:write(dest,r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development-only',action='store_true');development() if p.parse_args().development_only else full()
