"""TRAIN/VAL development, then separately frozen retrospective replay. No future poses."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.benchmark import environment_record
from cranepilot.reliability_pilot import FrozenReliabilityPilot
from scripts.modal_history import load_allowed,concatenate,modal
from scripts.modal_reference_challenge import predict as old_arx
from scripts.modal_backtest_gate import sliced,compose
from scripts.active_qualification import qualification_packets,frozen_predictions
from scripts.run_active_qualification import groups
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import write,verify_hashes
from scripts import drive_conditioned as dc

CONFIG=Path('configs/drive_conditioned_20261107.json');RUN=Path('artifacts/runs/drive_conditioned_20261107');OUT=Path('results/drive_conditioned_20261107')
LOCKED=[CONFIG,Path('docs/DRIVE_CONDITIONED_DESIGN_20261107.md'),Path('scripts/drive_conditioned.py'),Path('scripts/run_drive_conditioned.py'),Path('tests/test_drive_conditioned.py')]

def setup():
    cfg=json.loads(CONFIG.read_text());qc=json.loads(Path(cfg['source_task']).read_text());pilot=FrozenReliabilityPilot(cfg['bundle']);return cfg,qc,pilot,pilot.cfg

def pack(records,axis,task,qc,cal):
    triples=[qualification_packets(s,axis,task,qc,cal) for s in records.values()]
    labels={key:np.concatenate([x[2][key] for x in triples]) for key in triples[0][2]}
    return concatenate([x[0] for x in triples]),np.concatenate([x[1] for x in triples]),labels

def score_groups(pred,target,labels):
    drive=labels['drive_active'];r={'balanced':dc.balanced_score(pred,target,drive)}
    for name,mask in [('quiet',~drive),('drive',drive)]:
        r[name]=float(np.abs(np.rad2deg(pred[mask]-target[mask])).mean()) if mask.any() else None
    return r

def ref_prediction(p,free,choice,long_model,short_model,pilot,name,cal,task):
    if choice=='old':return frozen_predictions(p,pilot,name)[0]['strong_arx']
    if choice=='short':
        short=sliced(p,0,61,cal);return dc.arx_predict(short_model,short,modal(short,task,'two_mode')[0])
    if choice=='long':return dc.arx_predict(long_model,p,free)
    raise ValueError('Unknown reference')

def candidate_predictions(p,cfg,task,pilot,name,cal,prior,long_model,short_model,choice):
    # Only past arrays reach this function. Realised future targets/poses are absent.
    if 'target' in p:raise ValueError('Targets must be separate')
    old,_=frozen_predictions(p,pilot,name);free=old['frozen_modal'];ref=ref_prediction(p,free,choice,long_model,short_model,pilot,name,cal,task)
    prefix=sliced(p,20,None,cal);prefix_free=modal(prefix,task,'two_mode')[0]
    back_ref=old_arx(pilot.short_models[name],prefix,task) if choice=='old' else dc.arx_predict(short_model,prefix,prefix_free)
    indices=np.arange(6);observed=p['theta'][:,np.array([18,16,13,10,5,0])];er=np.abs(back_ref[:,indices]-observed).mean(1)
    predictions=dict(strong_arx=ref,old_blend=old['frozen_blend'],free_modal=free);diagnostics=dict(reference_backtest=back_ref[:,indices],observed_backtest=observed,arx_backtest_mae=er)
    for continuation in cfg['continuations']:
        prepared=dc.prepare(p,task,continuation,prior['bounds']);past=dc.prepare(prefix,task,continuation,prior['bounds'])
        for strategy in cfg['strategies']:
            key=f'{continuation}_{strategy}';y,gamma=dc.predict(prepared,prior,strategy);back,_=dc.predict(past,prior,strategy);em=np.abs(back[:,indices]-observed).mean(1);blend,w=compose(y,ref,dict(modal_backtest_mae=em,arx_backtest_mae=er),'soft');predictions['physical_'+key]=y;predictions['blend_'+key]=blend;diagnostics['gamma_'+key]=gamma;diagnostics['weight_'+key]=w;diagnostics['backtest_'+key]=back[:,indices];diagnostics['backtest_mae_'+key]=em
    if not all(np.isfinite(x).all() for x in predictions.values()):raise ValueError('Retain nonfinite failed attempt')
    return predictions,diagnostics

def metric_rows(predictions,target,labels,name,sid,role,task,qc):
    rows=[]
    for group,mask in groups(labels,qc).items():
        for method,y in predictions.items():
            for j,h in enumerate(task['horizons_raw']):
                err=np.rad2deg(y[mask,j]-target[mask,j]);rows.append(dict(case=name,scenario=int(sid),role=role,group=group,method=method,horizon_s=h*.01,windows=int(mask.sum()),mae_deg=float(np.abs(err).mean()) if len(err) else np.nan,rmse_deg=float(np.sqrt((err**2).mean())) if len(err) else np.nan))
    return rows

def develop():
    cfg,qc,pilot,task=setup()
    for path in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit first {path}')
    if RUN.exists():raise FileExistsError('Retain every attempt')
    RUN.mkdir(parents=True);bundle=Path(cfg['bundle']);sources=set(pilot.manifest['source_hashes'])|set(pilot.reliability_manifest['source_hashes']);paths=LOCKED+[Path(cfg['source_task']),Path('scripts/active_qualification.py'),Path('scripts/run_active_qualification.py'),Path('scripts/run_modal_confirmation.py'),Path(qc['data'])/'dataset.npz',Path(qc['data'])/'manifest.json']+[Path(x) for x in sources]+[f for f in bundle.rglob('*') if f.is_file()];prov=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,qualification=qc,task=task,guarded_hashes={str(f):sha256(f) for f in paths},environment=environment_record(),old_evaluation_read_during_development=False,independent_confirmation=False,final_goal_achieved=False);write(RUN/'development.json',prov);choices={};case_scores={};grid_rows=[];metrics=[];fits=0
    try:
        for family,fold in task['families'].items():
            train=load_allowed(task,fold['train'],'train');val=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';out=RUN/name;out.mkdir();cal=LengthCalibration(**pilot.cases[name]['calibration']);tr,yt,lt=pack(train,axis,task,qc,cal);va,yv,lv=pack(val,axis,task,qc,cal);prepared=dc.prepare(tr,task);prior=dc.fit_prior(prepared,tr,lt['drive_active'],cfg['prior_ridge']);dc.save(out/'prior.npz',prior);fv=modal(va,task,'two_mode')[0];rank=[]
                gram=np.einsum('ntu,ntv->nuv',prepared['force']/prior['scale'],prepared['force']/prior['scale'])/prepared['force'].shape[1];condition=np.linalg.cond(gram+np.eye(gram.shape[-1])[None]);write(out/'prior_diagnostics.json',dict(channels=prior['channels'].tolist(),normalized_gamma=prior['gamma'].tolist(),physical_basis_gain=(prior['gamma']/prior['scale']).tolist(),scale=prior['scale'].tolist(),online_system_condition_median=float(np.median(condition)),online_system_condition_max=float(condition.max()),train_records=fold['train'],validation_records=fold['validation'],train_windows=len(yt),validation_windows=len(yv)))
                for representation in cfg['representations']:
                    for form in cfg['forms']:
                        stats=dc.arx_stats(tr,yt,prepared['free'],lt['drive_active'],representation,form)
                        for alpha in cfg['ridge_grid']:
                            model=dc.arx_fit(stats,alpha);filename=f'arx_{representation}_{form}_{alpha:g}.npz';dc.save(out/filename,model);pred=dc.arx_predict(model,va,fv);score=dc.balanced_score(pred,yv,lv['drive_active']);rank.append((score,filename,model));grid_rows.append(dict(case=name,representation=representation,form=form,alpha=alpha,validation_balanced_mae_deg=score,model_file=filename));fits+=1
                rank.sort(key=lambda item:item[0]);_,filename,long_model=rank[0];short_train=sliced(tr,0,61,cal);short_free=modal(short_train,task,'two_mode')[0];short_model=dc.arx_fit(dc.arx_stats(short_train,yt,short_free,lt['drive_active'],long_model['representation'],long_model['form']),long_model['alpha']);dc.save(out/'short_arx.npz',short_model);fits+=1
                ref_scores={key:dc.balanced_score(ref_prediction(va,fv,key,long_model,short_model,pilot,name,cal,task),yv,lv['drive_active']) for key in ['long','short','old']};choice=min(ref_scores,key=ref_scores.get);predictions,diagnostics=candidate_predictions(va,cfg,task,pilot,name,cal,prior,long_model,short_model,choice);case_scores[name]={key:score_groups(pred,yv,lv) for key,pred in predictions.items()};choices[name]=dict(family=family,axis=axis,long_model=filename,reference=choice,reference_validation_scores=ref_scores,train_records=fold['train'],validation_records=fold['validation']);np.savez_compressed(out/'validation.npz',k=va['k'],scenario=va['scenario'],**lv,**predictions,**{'diagnostic_'+k:v for k,v in diagnostics.items()})
                for sid in fold['validation']:
                    mask=va['scenario']==sid;metrics+=metric_rows({k:v[mask] for k,v in predictions.items()},yv[mask],{k:v[mask] for k,v in lv.items()},name,sid,'validation',task,qc)
                print('CASE',name,'REF',choice,ref_scores,'CANDIDATES',json.dumps(case_scores[name]),flush=True)
        means={key:float(np.mean([c[key]['balanced'] for c in case_scores.values()])) for key in next(iter(case_scores.values()))};quiet={key:float(np.mean([c[key]['quiet'] for c in case_scores.values() if c[key]['quiet'] is not None])) for key in means};selected=min(means,key=means.get);wins=sum(c[selected]['balanced']<c['strong_arx']['balanced'] for c in case_scores.values());gate=cfg['development_gate'];passed=means[selected]<=gate['maximum_balanced_mae_ratio']*means['strong_arx'] and wins>=gate['minimum_case_wins'] and quiet[selected]<=gate['maximum_quiet_mae_ratio']*quiet['strong_arx'];selection=dict(selected=selected,cases=choices,case_scores=case_scores,validation_means=means,validation_quiet_means=quiet,development_gate_passed=bool(passed),validation_case_wins=int(wins),validation_change_percent=100*(means[selected]/means['strong_arx']-1),arx_fits=fits,physical_prior_fits=len(choices),independent_confirmation=False,final_goal_achieved=False);write(RUN/'selection.json',selection);pd.DataFrame(grid_rows).to_csv(RUN/'arx_grid.csv',index=False);pd.DataFrame(metrics).to_csv(RUN/'validation_metrics.csv',index=False);prov.update(status='complete',selection_sha256=sha256(RUN/'selection.json'),result_hashes={str(f.relative_to(RUN)):sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name!='development.json'});print('SELECTED',selected,'MAE',means,'GATE',passed,'FITS',fits,flush=True)
    except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(RUN/'development.json',prov)

def evaluate():
    cfg,qc,pilot,task=setup();prov=json.loads((RUN/'development.json').read_text());assert prov['status']=='complete';verify_hashes(prov['guarded_hashes']);verify_hashes(prov['result_hashes'],RUN)
    # Numerical models and selection must be committed in a reviewable mirror first.
    for rel in ['development.json']+list(prov['result_hashes']):
        path=OUT/'frozen_development'/rel
        if path.read_bytes()!=(RUN/rel).read_bytes() or subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Freeze and commit development before replay: {path}')
    selection=json.loads((RUN/'selection.json').read_text());out=RUN/'retrospective'
    if out.exists():raise FileExistsError('Retain prior replay')
    out.mkdir();q=dict(status='running',selection_sha256=sha256(RUN/'selection.json'),development_sha256=sha256(RUN/'development.json'),scope='Previously opened old records; retrospective, not independent',independent_confirmation=False,final_goal_achieved=False);write(out/'provenance.json',q);metrics=[]
    try:
        for family,fold in task['families'].items():
            records=load_confirmation(task,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.cases[name]['calibration']);case=selection['cases'][name];prior=dc.load(RUN/name/'prior.npz');long_model=dc.load(RUN/name/case['long_model']);short_model=dc.load(RUN/name/'short_arx.npz')
                for sid,s in records.items():
                    packet,target,labels=qualification_packets(s,axis,task,qc,cal);predictions,diagnostics=candidate_predictions(packet,cfg,task,pilot,name,cal,prior,long_model,short_model,case['reference']);predictions['selected']=predictions[selection['selected']].copy();np.savez_compressed(out/f'{name}_s{sid}.npz',k=packet['k'],**labels,**predictions,**{'diagnostic_'+k:v for k,v in diagnostics.items()});metrics+=metric_rows(predictions,target,labels,name,sid,'old_evaluation',task,qc);print('RETROSPECTIVE',name,sid,flush=True)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);q.update(status='complete',result_hashes={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='provenance.json'})
    except Exception as e:q.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(out/'provenance.json',q)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');args=parser.parse_args();evaluate() if args.evaluate else develop()
