"""Matched known-age validation study and frozen retrospective/synthetic replay."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256,QUANTIZATION_RAD
from cranepilot.benchmark import environment_record
from cranepilot.physics import LengthCalibration
from cranepilot.regime_pilot import FrozenRegimePilot
from scripts.modal_history import load_allowed,concatenate
from scripts.run_modal_confirmation import load_confirmation
from scripts.active_qualification import qualification_packets
from scripts.run_drive_conditioned import metric_rows
from scripts.run_external_benchmark import write,verify_hashes
from scripts import drive_conditioned as dc
from scripts import latency_observer as lo

CONFIG=Path('configs/latency_observer_20261108.json');RUN=Path('artifacts/runs/latency_observer_20261108');OUT=Path('results/latency_observer_20261108')
LOCKED=[CONFIG,Path('docs/LATENCY_OBSERVER_DESIGN_20261108.md'),Path('scripts/latency_observer.py'),Path('scripts/run_latency_observer.py'),Path('tests/test_latency_observer.py'),Path('src/cranepilot/latency_pilot.py')]

def setup():
    cfg=json.loads(CONFIG.read_text());qc=json.loads(Path(cfg['qualification']).read_text());qc.update(origin_start_s=cfg['origin_start_s'],origin_stride_raw=cfg['origin_stride_raw']);pilot=FrozenRegimePilot(cfg['bundle']);return cfg,qc,pilot,pilot.pilot.cfg

def current_packet(s,axis,task,qc,cal):return qualification_packets(s,axis,task,qc,cal)

def raw_history(s,k,axis,family,age,noise,cfg,case_index):
    coordinates=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if x=='x' else 'Jr' for x in coordinates];raw=k[:,None]-np.arange(401);history={key:s.signals[key][raw].copy() for key in ['L']+coordinates+refs};index=raw-age
    if index.min()<0:raise ValueError('Genuine dated prehistory required')
    history[axis]=s.signals[axis][index].copy()
    if noise:
        rng=np.random.default_rng(np.random.SeedSequence([cfg['noise_seed'],s.sid,case_index,age,int(noise*1000)]));history[axis]+=rng.uniform(-noise*QUANTIZATION_RAD,noise*QUANTIZATION_RAD,history[axis].shape)
    return history

def balanced(y,target,drive):return dc.balanced_score(y,target,drive)

def case_data(records,axis,family,task,qc,cal,cfg,index,ages):
    for sid,s in records.items():
        current,target,labels=current_packet(s,axis,task,qc,cal)
        for age in ages:
            for noise in cfg['noise_quantum_multipliers']:
                raw=raw_history(s,current['k'],axis,family,age,noise,cfg,index);p=lo.packet(raw,axis,cal,age);p['scenario']=np.full(len(target),sid);dated=lo.dated_modal(p['theta'],p['omega_sensor'],p['age_raw'],task);yield sid,age,noise,current,target,labels,raw,p,dated

def add_rows(rows,pred,target,labels,name,sid,role,task,qc,age,noise):
    current=metric_rows(pred,target,labels,name,sid,role,task,qc)
    for row in current:row.update(age_ms=age*10,noise_quantum=noise)
    rows+=current

def ref_forecast(choice,raw,age,axis,pilot,name,p,dated,models,cache):
    if choice['type']=='augmented':return lo.arx_prediction(models[choice['file']],p,dated)
    method=choice['method']
    if method not in cache:cache[method]=lo.predict(raw,age,axis,method,pilot,name)
    return cache[method][1][choice['expert']]

def develop():
    cfg,qc,pilot,task=setup()
    for f in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{f}'])!=f.read_bytes():raise ValueError(f'Commit before fitting {f}')
    if RUN.exists():raise FileExistsError('Retain every attempt')
    RUN.mkdir(parents=True);bundle=Path(cfg['bundle']);sources=set(pilot.manifest['source_hashes'])|set(pilot.pilot.manifest['source_hashes'])|set(pilot.pilot.reliability_manifest['source_hashes']);paths=LOCKED+[Path(cfg['qualification']),Path('scripts/run_drive_conditioned.py'),Path(task['data'])/'dataset.npz',Path(task['data'])/'manifest.json']+[Path(x) for x in sources]+[f for f in bundle.rglob('*') if f.is_file()];prov=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,qualification=qc,task=task,guarded_hashes={str(f):sha256(f) for f in paths},environment=environment_record(),old_evaluation_read_during_development=False,independent_confirmation=False,final_goal_achieved=False);write(RUN/'development.json',prov);scores={};references={};grid=[];rows=[];index=0
    try:
        for family,fold in task['families'].items():
            train=load_allowed(task,fold['train'],'train');val=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';out=RUN/name;out.mkdir();cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);parts=[];targets=[];bases=[];drives=[]
                for sid,age,noise,current,target,labels,raw,p,dated in case_data(train,axis,family,task,qc,cal,cfg,index,cfg['train_ages_raw']):parts.append(p);targets.append(target);bases.append(dated);drives.append(labels['drive_active'])
                tr=concatenate(parts);yt=np.concatenate(targets);bt=np.concatenate(bases);drive=np.concatenate(drives);weight=dc.balanced_weights(tr['scenario'],drive);models={}
                for representation in cfg['representations']:
                    X=lo.arx_features(tr,representation)
                    for form in cfg['forms']:
                        base=np.repeat(tr['theta'][:,:1],7,axis=1) if form=='direct' else bt;stats=lo.fit_stats(X,yt,base,weight,representation,form)
                        for alpha in cfg['ridge_grid']:
                            filename=f'arx_{representation}_{form}_{alpha:g}.npz';model=dc.arx_fit(stats,alpha);dc.save(out/filename,model);models[filename]=model
                del tr,yt,bt,parts,targets,bases,drives,X
                per=[];ref_scores={key:[] for key in models};candidate_refs={};val_batches=[]
                for sid,age,noise,current,target,labels,raw,p,dated in case_data(val,axis,family,task,qc,cal,cfg,index,cfg['train_ages_raw']):
                    cache={};pred={};weights={};available={}
                    for method in cfg['methods']:
                        if age==0 and cache:
                            y,diag=next(iter(cache.values()));cache[method]=(y.copy(),{k:v.copy() for k,v in diag.items()})
                        else:cache[method]=lo.predict(raw,age,axis,method,pilot,name)
                        y,diag=cache[method];pred[method]=y;weights['weight_'+method]=diag['weights'];per.append(dict(method=method,age_raw=age,noise=noise,score=balanced(y,target,labels['drive_active'])))
                        for expert in ['frozen_arx','active_arx']:
                            key=method+'__'+expert;available[key]=diag[expert];candidate_refs[key]=dict(type='aligned',method=method,expert=expert);ref_scores.setdefault(key,[]).append(balanced(diag[expert],target,labels['drive_active']))
                    for filename,model in models.items():ref_scores[filename].append(balanced(lo.arx_prediction(model,p,dated),target,labels['drive_active']))
                    file=f'validation_s{sid}_a{age}_n{noise:g}.npz';np.savez_compressed(out/file,k=current['k'],**labels,**pred,**weights,**{'ref_'+k:v for k,v in available.items()});val_batches.append((file,age,noise,target,labels,p,dated,raw,current,sid));add_rows(rows,pred,target,labels,name,sid,'validation',task,qc,age,noise);print('VAL',name,age,noise,flush=True)
                mean_ref={key:float(np.mean(value)) for key,value in ref_scores.items()};best_aug=min(models,key=lambda key:mean_ref[key]);best_ref=min(mean_ref,key=mean_ref.get);choice=dict(type='augmented',file=best_ref) if best_ref in models else candidate_refs[best_ref];references[name]=dict(reference=choice,augmented_file=best_aug,validation_reference_scores=mean_ref,family=family,axis=axis);scores[name]={method:dict(all=float(np.mean([r['score'] for r in per if r['method']==method])),positive=float(np.mean([r['score'] for r in per if r['method']==method and r['age_raw']>0]))) for method in cfg['methods']};scores[name]['matched_arx']=dict(all=mean_ref[best_ref]);scores[name]['augmented_arx']=dict(all=mean_ref[best_aug])
                for filename,model in models.items():grid.append(dict(case=name,file=filename,representation=model['representation'],form=model['form'],alpha=model['alpha'],validation_mae_deg=mean_ref[filename]))
                for file,age,noise,target,labels,p,dated,raw,current,sid in val_batches:
                    with np.load(out/file) as z:cached={key[4:]:z[key].copy() for key in z.files if key.startswith('ref_')}
                    ref=lo.arx_prediction(models[best_ref],p,dated) if choice['type']=='augmented' else cached[choice['method']+'__'+choice['expert']];aug=lo.arx_prediction(models[best_aug],p,dated);add_rows(rows,dict(matched_arx=ref,augmented_arx=aug,dated_modal=dated),target,labels,name,sid,'validation',task,qc,age,noise);np.savez_compressed(out/('reference_'+file),k=current['k'],matched_arx=ref,augmented_arx=aug,dated_modal=dated)
                index+=1
        means={method:float(np.mean([c[method]['all'] for c in scores.values()])) for method in cfg['methods']+['matched_arx','augmented_arx']};positive={method:float(np.mean([c[method]['positive'] for c in scores.values()])) for method in cfg['methods']};selected=min(cfg['methods'],key=lambda key:means[key]);wins=sum(c[selected]['all']<c['matched_arx']['all'] for c in scores.values());g=cfg['development_gate'];passed=positive[selected]<=g['maximum_unaware_positive_age_ratio']*positive['unaware'] and positive[selected]<=g['maximum_hold_positive_age_ratio']*positive['hold'] and means[selected]<=g['maximum_matched_arx_ratio']*means['matched_arx'] and wins>=g['minimum_arx_case_wins'];selection=dict(selected=selected,cases=references,case_scores=scores,validation_means=means,positive_age_means=positive,arx_case_wins=int(wins),development_gate_passed=bool(passed),arx_fits=len(grid),independent_confirmation=False,final_goal_achieved=False);write(RUN/'selection.json',selection);pd.DataFrame(grid).to_csv(RUN/'arx_grid.csv',index=False);pd.DataFrame(rows).to_csv(RUN/'validation_metrics.csv',index=False);prov.update(status='complete',selection_sha256=sha256(RUN/'selection.json'),result_hashes={str(f.relative_to(RUN)):sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name!='development.json'});print('SELECTED',selected,'MEANS',means,'GATE',passed,'ARX_WINS',wins,flush=True)
    except Exception as error:prov.update(status='failed',failure=f'{type(error).__name__}: {error}');raise
    finally:write(RUN/'development.json',prov)

def evaluate():
    cfg,qc,pilot,task=setup();p=json.loads((RUN/'development.json').read_text());assert p['status']=='complete';verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN)
    for rel in ['development.json']+list(p['result_hashes']):
        f=OUT/'frozen_development'/rel
        if f.read_bytes()!=(RUN/rel).read_bytes() or subprocess.check_output(['git','show',f'HEAD:{f}'])!=f.read_bytes():raise ValueError('Commit frozen selection first')
    selection=json.loads((RUN/'selection.json').read_text());out=RUN/'retrospective'
    if out.exists():raise FileExistsError('Retain prior replay')
    out.mkdir();q=dict(status='running',selection_sha256=sha256(RUN/'selection.json'),development_sha256=sha256(RUN/'development.json'),independent_confirmation=False,final_goal_achieved=False);write(out/'provenance.json',q);rows=[];index=0
    try:
        for family,fold in task['families'].items():
            records=load_confirmation(task,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);case=selection['cases'][name];model_names={case['augmented_file']}
                if case['reference']['type']=='augmented':model_names.add(case['reference']['file'])
                models={filename:dc.load(RUN/name/filename) for filename in model_names}
                for sid,age,noise,current,target,labels,raw,p,dated in case_data(records,axis,family,task,qc,cal,cfg,index,cfg['evaluation_ages_raw']):
                    cache={method:lo.predict(raw,age,axis,method,pilot,name) for method in dict.fromkeys([selection['selected'],'hold','unaware'])};ref=ref_forecast(case['reference'],raw,age,axis,pilot,name,p,dated,models,cache);pred=dict(selected=cache[selection['selected']][0],timestamp_hold=cache['hold'][0],unaware=cache['unaware'][0],matched_arx=ref,augmented_arx=lo.arx_prediction(models[case['augmented_file']],p,dated),dated_modal=dated);diag=cache[selection['selected']][1];np.savez_compressed(out/f'{name}_s{sid}_a{age}_n{noise:g}.npz',k=current['k'],**labels,**pred,**{'diagnostic_'+key:value for key,value in diag.items()});add_rows(rows,pred,target,labels,name,sid,'old_evaluation',task,qc,age,noise);print('REPLAY',name,sid,age,noise,flush=True)
                index+=1
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);q.update(status='complete',result_hashes={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='provenance.json'})
    except Exception as error:q.update(status='failed',failure=f'{type(error).__name__}: {error}');raise
    finally:write(out/'provenance.json',q)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');evaluate() if parser.parse_args().evaluate else develop()
