"""Precommitted dated-forecast grid and age-conditional strong ARX reference."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.benchmark import environment_record
from cranepilot.physics import LengthCalibration
from cranepilot.regime_pilot import FrozenRegimePilot
from scripts.modal_history import load_allowed,concatenate
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_latency_observer import case_data,add_rows,balanced
from scripts.run_external_benchmark import write,verify_hashes
from scripts.verify_latency_observer import model_names
from scripts import latency_observer as lo,drive_conditioned as dc,dated_forecast as df

CONFIG=Path('configs/dated_forecast_20261108.json');RUN=Path('artifacts/runs/dated_forecast_20261108');OUT=Path('results/dated_forecast_20261108')
LOCKED=[CONFIG,Path('docs/DATED_FORECAST_DESIGN_20261108.md')]+[Path(x) for x in ['scripts/dated_forecast.py','scripts/run_dated_forecast.py','src/cranepilot/dated_pilot.py','scripts/build_dated_bundle.py','scripts/verify_dated_forecast.py','tests/test_dated_forecast.py']]

def setup():
    cfg=json.loads(CONFIG.read_text());qc=json.loads(Path(cfg['qualification']).read_text());qc.update(origin_start_s=cfg['origin_start_s'],origin_stride_raw=cfg['origin_stride_raw']);pilot=FrozenRegimePilot(cfg['bundle']);return cfg,qc,pilot,pilot.pilot.cfg

def fit_case(records,axis,family,task,qc,cal,cfg,index):
    parts=[];ys=[];bs=[];drive=[]
    for sid,age,noise,current,target,labels,raw,p,dated in case_data(records,axis,family,task,qc,cal,cfg,index,cfg['train_ages_raw']):parts.append(p);ys.append(target);bs.append(dated);drive.append(labels['drive_active'])
    tr=concatenate(parts);target=np.concatenate(ys);dated=np.concatenate(bs);weights=dc.balanced_weights(tr['scenario'],np.concatenate(drive));models={}
    for representation in cfg['representations']:
        X=lo.arx_features(tr,representation)
        for form in cfg['forms']:
            base=np.repeat(tr['theta'][:,:1],7,axis=1) if form=='direct' else dated;stats=lo.fit_stats(X,target,base,weights,representation,form)
            for alpha in cfg['ridge_grid']:models[f'arx_{representation}_{form}_{alpha:g}.npz']=dc.arx_fit(stats,alpha)
    return models,len(target)

def reference_pool(cfg,name,new):
    models={'new/'+k:v for k,v in new.items()};models.update({'old/'+f:dc.load(Path(cfg['old_arx'])/name/f) for f in model_names(cfg)});return models

def candidates(raw,age,axis,pilot,name,p,dated,models,cfg):
    out={k:lo.arx_prediction(m,p,dated) for k,m in models.items()}
    for method in cfg['head_methods']:
        for expert in ['frozen_arx','active_arx']:out[f'aligned/{method}/{expert}']=df.arx(raw,age,axis,pilot,name,method,expert)
    return out

def choose(score):return {str(age):dict(key=min(s,key=s.get),scores=s) for age,s in score.items()}
def nearest(ages,age):return min(ages,key=lambda a:(abs(a-age),a))

def selection_from(scores,refs,aug,cfg):
    means={method:float(np.mean([np.mean([r['score'] for r in rows if r['method']==method]) for rows in scores.values()])) for method in cfg['methods']+['matched_arx','augmented_arx','bare_dated_modal']};selected=min(cfg['methods'],key=lambda m:means[m]);wins=sum(np.mean([r['score'] for r in rows if r['method']==selected])<np.mean([r['score'] for r in rows if r['method']=='matched_arx']) for rows in scores.values());byage={str(age):{m:float(np.mean([np.mean([r['score'] for r in rows if r['method']==m and r['age_raw']==age]) for rows in scores.values()])) for m in [selected,'parent_quad']} for age in cfg['train_ages_raw']};g=cfg['development_gate'];passed=means[selected]<=g['maximum_parent_ratio']*means['parent_quad'] and means[selected]<=g['maximum_arx_ratio']*means['matched_arx'] and wins>=g['minimum_arx_case_wins'] and all(v[selected]<=g['maximum_each_age_parent_ratio']*v['parent_quad'] for v in byage.values());return dict(selected=selected,validation_means=means,per_age=byage,arx_case_wins=int(wins),development_gate_passed=bool(passed),cases={name:dict(reference=refs[name],augmented_file=aug[name],family=name.split('_')[0],axis=name.split('_')[1]) for name in refs},arx_fits=120,case_scores=scores,independent_confirmation=False,final_goal_achieved=False)

def develop():
    cfg,qc,pilot,task=setup()
    for f in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{f}'])!=f.read_bytes():raise ValueError(f'Commit before fitting {f}')
    if RUN.exists():raise FileExistsError('Retain every attempt')
    RUN.mkdir(parents=True);sources=set(pilot.manifest['source_hashes'])|set(pilot.pilot.manifest['source_hashes'])|set(pilot.pilot.reliability_manifest['source_hashes']);paths=LOCKED+[Path(cfg['qualification']),Path('scripts/run_latency_observer.py'),Path('scripts/latency_observer.py'),Path(task['data'])/'dataset.npz',Path(task['data'])/'manifest.json']+[Path(x) for x in sources]+[f for base in [Path(cfg['bundle']),Path(cfg['old_arx'])] for f in base.rglob('*') if f.is_file()];prov=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,qualification=qc,task=task,guarded_hashes={str(f):sha256(f) for f in paths},environment=environment_record(),old_evaluation_read_during_development=False,independent_confirmation=False,final_goal_achieved=False);write(RUN/'development.json',prov);scores={};refs={};aug={};grid=[];rows=[];index=0
    try:
        for family,fold in task['families'].items():
            train=load_allowed(task,fold['train'],'train');val=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';out=RUN/name;out.mkdir();cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);new,ntrain=fit_case(train,axis,family,task,qc,cal,cfg,index)
                for filename,m in new.items():dc.save(out/filename,m)
                models=reference_pool(cfg,name,new);ref_scores={a:{} for a in cfg['train_ages_raw']};batches=[];per=[]
                for sid,age,noise,current,target,labels,raw,p,dated in case_data(val,axis,family,task,qc,cal,cfg,index,cfg['train_ages_raw']):
                    pred,diag=df.predictions(raw,age,axis,pilot,name,cfg['methods']);ref=candidates(raw,age,axis,pilot,name,p,dated,models,cfg)
                    for key,y in ref.items():ref_scores[age].setdefault(key,[]).append(balanced(y,target,labels['drive_active']))
                    for method,y in dict(pred,bare_dated_modal=dated).items():per.append(dict(method=method,age_raw=age,noise=noise,score=balanced(y,target,labels['drive_active'])))
                    file=f'validation_s{sid}_a{age}_n{noise:g}.npz';np.savez_compressed(out/file,k=current['k'],**labels,**pred,**{'diagnostic_'+k:v for k,v in diag.items()});batches.append((file,sid,age,noise,target,labels,ref,dated));add_rows(rows,dict(pred,bare_dated_modal=dated),target,labels,name,sid,'validation',task,qc,age,noise);print('VAL',name,age,noise,flush=True)
                mean={a:{k:float(np.mean(v)) for k,v in pool.items()} for a,pool in ref_scores.items()};refs[name]=choose(mean);best_aug=min(new,key=lambda key:np.mean([pool['new/'+key] for pool in mean.values()]));aug[name]=best_aug
                for file,sid,age,noise,target,labels,ref,dated in batches:
                    pred=dict(matched_arx=ref[refs[name][str(age)]['key']],augmented_arx=ref['new/'+best_aug]);np.savez_compressed(out/('reference_'+file),**pred)
                    for method,y in pred.items():per.append(dict(method=method,age_raw=age,noise=noise,score=balanced(y,target,labels['drive_active'])))
                    add_rows(rows,pred,target,labels,name,sid,'validation',task,qc,age,noise)
                scores[name]=per
                for key,m in new.items():grid.append(dict(case=name,file=key,representation=m['representation'],form=m['form'],alpha=m['alpha'],train_windows=ntrain,validation_mae_deg=float(np.mean([pool['new/'+key] for pool in mean.values()]))))
                index+=1
        selection=selection_from(scores,refs,aug,cfg);write(RUN/'selection.json',selection);pd.DataFrame(grid).to_csv(RUN/'arx_grid.csv',index=False);pd.DataFrame(rows).to_csv(RUN/'validation_metrics.csv',index=False);prov.update(status='complete',selection_sha256=sha256(RUN/'selection.json'),result_hashes={str(f.relative_to(RUN)):sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name!='development.json'});print('SELECTED',selection['selected'],'GATE',selection['development_gate_passed'],'MEANS',selection['validation_means'],flush=True)
    except Exception as error:prov.update(status='failed',failure=f'{type(error).__name__}: {error}');raise
    finally:write(RUN/'development.json',prov)

def reference(key,raw,age,axis,pilot,name,p,dated,cfg,new):
    source,rest=key.split('/',1)
    if source=='aligned':method,expert=rest.split('/');return df.arx(raw,age,axis,pilot,name,method,expert)
    model=new[rest] if source=='new' else dc.load(Path(cfg['old_arx'])/name/rest);return lo.arx_prediction(model,p,dated)

def evaluate():
    cfg,qc,pilot,task=setup();prov=json.loads((RUN/'development.json').read_text());verify_hashes(prov['guarded_hashes']);verify_hashes(prov['result_hashes'],RUN);assert prov['status']=='complete'
    for rel in ['development.json']+list(prov['result_hashes']):
        f=OUT/'frozen_development'/rel
        if f.read_bytes()!=(RUN/rel).read_bytes() or subprocess.check_output(['git','show',f'HEAD:{f}'])!=f.read_bytes():raise ValueError('Commit all models/selection before old replay')
    selection=json.loads((RUN/'selection.json').read_text());out=RUN/'retrospective'
    if out.exists():raise FileExistsError('Retain prior replay')
    out.mkdir();p=dict(status='running',selection_sha256=sha256(RUN/'selection.json'),independent_confirmation=False,final_goal_achieved=False);write(out/'provenance.json',p);rows=[];index=0
    try:
        for family,fold in task['families'].items():
            records=load_confirmation(task,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);case=selection['cases'][name];new={f:dc.load(RUN/name/f) for f in model_names(cfg)}
                for sid,age,noise,current,target,labels,raw,packet,dated in case_data(records,axis,family,task,qc,cal,cfg,index,cfg['evaluation_ages_raw']):
                    pred,diag=df.predictions(raw,age,axis,pilot,name,cfg['methods']);choice=case['reference'][str(nearest(cfg['train_ages_raw'],age))]['key'];pred.update(selected=pred[selection['selected']],matched_arx=reference(choice,raw,age,axis,pilot,name,packet,dated,cfg,new),augmented_arx=lo.arx_prediction(new[case['augmented_file']],packet,dated),bare_dated_modal=dated);np.savez_compressed(out/f'{name}_s{sid}_a{age}_n{noise:g}.npz',k=current['k'],**labels,**pred,**{'diagnostic_'+k:v for k,v in diag.items()});add_rows(rows,pred,target,labels,name,sid,'old_evaluation',task,qc,age,noise);print('REPLAY',name,sid,age,noise,flush=True)
                index+=1
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);p.update(status='complete',result_hashes={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='provenance.json'})
    except Exception as error:p.update(status='failed',failure=f'{type(error).__name__}: {error}');raise
    finally:write(out/'provenance.json',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true');evaluate() if p.parse_args().evaluate else develop()
