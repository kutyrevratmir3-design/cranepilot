"""Reproduce TRAIN parameters, VAL selection and every stored old-record forecast."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts import drive_conditioned as dc
from scripts.modal_history import load_allowed,modal
from scripts.modal_backtest_gate import sliced
from scripts.run_drive_conditioned import RUN,OUT,setup,pack,candidate_predictions,score_groups,metric_rows,ref_prediction
from scripts.drive_attribution import controls
from scripts.run_modal_confirmation import load_confirmation
from scripts.active_qualification import qualification_packets
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

def same_model(actual,stored):
    assert set(actual)==set(stored)
    for key in actual:
        if np.asarray(actual[key]).dtype.kind in 'US':np.testing.assert_array_equal(actual[key],stored[key])
        else:np.testing.assert_allclose(actual[key],stored[key],rtol=1e-12,atol=1e-12)

def main():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain earlier verification')
    r=dict(passed=False,source_sha256=sha256(Path(__file__)),checks=[],independent_confirmation=False,final_goal_achieved=False)
    try:
        cfg,qc,pilot,task=setup();prov=json.loads((RUN/'development.json').read_text());ret=json.loads((RUN/'retrospective/provenance.json').read_text());attribution=json.loads((RUN/'attribution/provenance.json').read_text());selection=json.loads((RUN/'selection.json').read_text());assert prov['status']==ret['status']==attribution['status']=='complete' and not prov['old_evaluation_read_during_development'];verify_hashes(prov['guarded_hashes']);verify_hashes(prov['result_hashes'],RUN);verify_hashes(ret['result_hashes'],RUN/'retrospective');verify_hashes(attribution['source_hashes']);verify_hashes(attribution['result_hashes'],RUN/'attribution');assert not attribution['used_for_selection'] and attribution['selection_sha256']==sha256(RUN/'selection.json');assert ret['selection_sha256']==sha256(RUN/'selection.json') and ret['development_sha256']==sha256(RUN/'development.json');r['original_files']=original_files();assert r['original_files']['passed'];rows=[];case_scores={};fit_count=0
        for family,fold in task['families'].items():
            train=load_allowed(task,fold['train'],'train');validation=load_allowed(task,fold['validation'],'validation');records=load_confirmation(task,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.cases[name]['calibration']);case=selection['cases'][name];base=RUN/name;tr,yt,lt=pack(train,axis,task,qc,cal);prepared=dc.prepare(tr,task);prior=dc.fit_prior(prepared,tr,lt['drive_active'],cfg['prior_ridge']);same_model(prior,dc.load(base/'prior.npz'))
                for representation in cfg['representations']:
                    for form in cfg['forms']:
                        stats=dc.arx_stats(tr,yt,prepared['free'],lt['drive_active'],representation,form)
                        for alpha in cfg['ridge_grid']:
                            model=dc.arx_fit(stats,alpha);same_model(model,dc.load(base/f'arx_{representation}_{form}_{alpha:g}.npz'));fit_count+=1
                long_model=dc.load(base/case['long_model']);short=sliced(tr,0,61,cal);sf=modal(short,task,'two_mode')[0];short_model=dc.arx_fit(dc.arx_stats(short,yt,sf,lt['drive_active'],long_model['representation'],long_model['form']),long_model['alpha']);same_model(short_model,dc.load(base/'short_arx.npz'));fit_count+=1
                batches=[('validation',None,pack(validation,axis,task,qc,cal),base/'validation.npz')]+[('old_evaluation',sid,qualification_packets(s,axis,task,qc,cal),RUN/'retrospective'/f'{name}_s{sid}.npz') for sid,s in records.items()]
                for role,sid,(p,target,labels),file in batches:
                    assert set(p)<={'theta','length','omega','scenario','k','x','xr','J','Jr'} and all(value.shape[1]<=81 for value in p.values() if value.ndim==2);predictions,diag=candidate_predictions(p,cfg,task,pilot,name,cal,prior,long_model,short_model,case['reference'])
                    if role=='old_evaluation':predictions['selected']=predictions[selection['selected']].copy()
                    maximum=0.
                    with np.load(file) as z:
                        np.testing.assert_array_equal(p['k'],z['k'])
                        for key,value in labels.items():np.testing.assert_array_equal(value,z[key])
                        for key,value in dict(**predictions,**{'diagnostic_'+k:v for k,v in diag.items()}).items():
                            np.testing.assert_allclose(value,z[key],rtol=0,atol=1e-12);maximum=max(maximum,float(np.max(np.abs(value-z[key]))))
                    extra=controls(p,cfg,task,pilot,name,cal,prior,long_model,short_model,case['reference'])
                    for record in np.unique(p['scenario']):
                        mask=p['scenario']==record;rows+=metric_rows({k:v[mask] for k,v in predictions.items()},target[mask],{k:v[mask] for k,v in labels.items()},name,record,role,task,qc)
                        with np.load(RUN/'attribution'/f'{role}_{name}_s{record}.npz') as z:
                            np.testing.assert_array_equal(p['k'][mask],z['k'])
                            for key,value in extra.items():np.testing.assert_allclose(value[mask],z[key],rtol=0,atol=1e-12)
                        rows+=metric_rows({k:v[mask] for k,v in extra.items()},target[mask],{k:v[mask] for k,v in labels.items()},name,record,role,task,qc)
                    if role=='validation':
                        case_scores[name]={key:score_groups(y,target,labels) for key,y in predictions.items()}
                        grid=pd.read_csv(RUN/'arx_grid.csv');grid=grid[grid['case']==name];recomputed=[];fv=predictions['free_modal']
                        for row in grid.itertuples():
                            value=dc.balanced_score(dc.arx_predict(dc.load(base/row.model_file),p,fv),target,labels['drive_active']);np.testing.assert_allclose(value,row.validation_balanced_mae_deg,rtol=1e-12,atol=1e-12);recomputed.append((value,row.model_file))
                        assert min(recomputed,key=lambda pair:pair[0])[1]==case['long_model'];reference_scores={key:dc.balanced_score(ref_prediction(p,fv,key,long_model,short_model,pilot,name,cal,task),target,labels['drive_active']) for key in ['long','short','old']}
                        for key,value in reference_scores.items():np.testing.assert_allclose(value,case['reference_validation_scores'][key],rtol=1e-12,atol=1e-12)
                        assert min(reference_scores,key=reference_scores.get)==case['reference']
                        small={k:v[:2].copy() for k,v in p.items()};small['target']=np.full((2,7),1e6)
                        try:candidate_predictions(small,cfg,task,pilot,name,cal,prior,long_model,short_model,case['reference'])
                        except ValueError:pass
                        else:raise AssertionError('Inference accepted targets')
                    r['checks'].append(dict(case=name,scenario=None if sid is None else int(sid),role=role,windows=len(p['k']),methods=len(predictions)+len(extra),maximum_replay_error_rad=maximum));print('VERIFIED',name,role,sid,flush=True)
        expected=pd.DataFrame(rows);keys=['case','scenario','role','group','method','horizon_s'];actual=pd.concat([pd.read_csv(RUN/'validation_metrics.csv'),pd.read_csv(RUN/'retrospective/metrics.csv'),pd.read_csv(RUN/'attribution/metrics.csv')],ignore_index=True);pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),actual.sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        for case,scores in case_scores.items():
            for key,value in scores.items():
                for group,number in value.items():
                    if number is None:assert selection['case_scores'][case][key][group] is None
                    else:np.testing.assert_allclose(number,selection['case_scores'][case][key][group],rtol=1e-12,atol=1e-12)
        means={key:float(np.mean([c[key]['balanced'] for c in case_scores.values()])) for key in next(iter(case_scores.values()))};quiet={key:float(np.mean([c[key]['quiet'] for c in case_scores.values() if c[key]['quiet'] is not None])) for key in means};selected=min(means,key=means.get);assert selected==selection['selected'];wins=sum(c[selected]['balanced']<c['strong_arx']['balanced'] for c in case_scores.values());g=cfg['development_gate'];passed=means[selected]<=g['maximum_balanced_mae_ratio']*means['strong_arx'] and wins>=g['minimum_case_wins'] and quiet[selected]<=g['maximum_quiet_mae_ratio']*quiet['strong_arx'];assert bool(passed)==selection['development_gate_passed'];assert fit_count==selection['arx_fits']==125;r.update(passed=True,arx_fits_reproduced=fit_count,physical_priors_reproduced=5,all_metrics_recomputed=True,selection_reproduced=selected,development_gate_reproduced=bool(passed),forecast_comparisons=sum(x['windows']*x['methods'] for x in r['checks']),development_sha256=sha256(RUN/'development.json'),retrospective_sha256=sha256(RUN/'retrospective/provenance.json'))
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
