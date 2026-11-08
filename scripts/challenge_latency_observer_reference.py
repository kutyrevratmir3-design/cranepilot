"""Late age-conditioned ARX comparator; primary frozen selection never changes."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import load_allowed,modal
from scripts.modal_backtest_gate import sliced
from scripts.modal_reference_challenge import predict as old_predict
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_latency_observer import RUN,OUT,setup,case_data,current_packet,raw_history,add_rows,balanced
from scripts.verify_latency_observer import model_names,compare_rows,same
from scripts.run_external_benchmark import write,verify_hashes
from scripts import latency_observer as lo
from scripts import drive_conditioned as dc

DEST=OUT/'reference_challenge';DESIGN=Path('docs/LATENCY_REFERENCE_CHALLENGE_20261108.md')

def choose_reference(cfg,pilot,task,qc):
    choices={};index=0
    for family,fold in task['families'].items():
        records=load_allowed(task,fold['validation'],'validation')
        for axis in fold['axes']:
            name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration']);models={f:dc.load(RUN/name/f) for f in model_names(cfg)};scores={age:{f:[] for f in models} for age in cfg['train_ages_raw']}
            for sid,age,noise,current,target,labels,raw,p,dated in case_data(records,axis,family,task,qc,cal,cfg,index,cfg['train_ages_raw']):
                for filename,m in models.items():scores[age][filename].append(balanced(lo.arx_prediction(m,p,dated),target,labels['drive_active']))
                with np.load(RUN/name/f'validation_s{sid}_a{age}_n{noise:g}.npz') as z:
                    for method in cfg['methods']:
                        for expert in ['frozen_arx','active_arx']:
                            key=method+'__'+expert;scores[age].setdefault(key,[]).append(balanced(z['ref_'+key],target,labels['drive_active']))
            choices[name]={}
            for age,candidates in scores.items():
                means={key:float(np.mean(value)) for key,value in candidates.items()};best=min(means,key=means.get);choice=dict(type='augmented',file=best) if best in models else dict(type='aligned',method=best.split('__')[0],expert=best.split('__')[1]);choices[name][str(age)]=dict(choice=choice,validation_scores=means)
            index+=1;print('REFERENCE_VAL',name,{age:c['choice'] for age,c in choices[name].items()},flush=True)
    return choices

def reference_prediction(raw,age,axis,pilot,name,choice,dated):
    cal=LengthCalibration(**pilot.pilot.cases[name]['calibration'])
    if choice['type']=='augmented':return lo.arx_prediction(dc.load(RUN/name/choice['file']),lo.packet(raw,axis,cal,age),dated)
    aligned=lo.complete_head(raw,age,axis,choice['method'],pilot,name);p=lo.packet(aligned,axis,cal);p.pop('age_raw');p.pop('omega_sensor');task=pilot.pilot.cfg
    def legacy():
        c=pilot.pilot.reliability_manifest['cases'][name]
        if c['reference']=='short':return old_predict(pilot.pilot.short_models[name],sliced(p,0,61,cal),task)
        return old_predict(pilot.pilot.cases[name]['reference'],p,task)
    if choice['expert']=='frozen_arx' or pilot.models[name]['reference']=='old':return legacy()
    m=pilot.models[name]['short'] if pilot.models[name]['reference']=='short' else pilot.models[name]['long'];part=sliced(p,0,61,cal) if pilot.models[name]['reference']=='short' else p;free=modal(part,task,'two_mode')[0] if m['form']=='modal_residual' else np.zeros((len(p['theta']),7));return dc.arx_predict(m,part,free)

def develop():
    if DEST.exists():raise FileExistsError('Retain earlier challenge')
    DEST.mkdir();cfg,qc,pilot,task=setup();choices=choose_reference(cfg,pilot,task,qc);write(DEST/'selection.json',dict(cases=choices,source_sha256=sha256(Path(__file__)),design_sha256=sha256(DESIGN),parent_selection_sha256=sha256(RUN/'selection.json'),late_posthoc_reference_challenge=True,designed_after_original_old_curve_seen=True,primary_model_unchanged=True,independent_confirmation=False))

def batches(cfg,qc,pilot,task,choices):
    index=0
    for family,fold in task['families'].items():
        records=load_confirmation(task,fold['confirmation'])
        for axis in fold['axes']:
            name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration'])
            for sid,s in records.items():
                current,target,labels=current_packet(s,axis,task,qc,cal)
                for age in cfg['evaluation_ages_raw']:
                    selected_age=min(cfg['train_ages_raw'],key=lambda a:(abs(a-age),a));choice=choices[name][str(selected_age)]['choice']
                    for noise in cfg['noise_quantum_multipliers']:
                        raw=raw_history(s,current['k'],axis,family,age,noise,cfg,index)
                        with np.load(RUN/'retrospective'/f'{name}_s{sid}_a{age}_n{noise:g}.npz') as z:dated=z['dated_modal'].copy()
                        y=reference_prediction(raw,age,axis,pilot,name,choice,dated);yield name,sid,age,noise,current,target,labels,y
            index+=1

def evaluate():
    cfg,qc,pilot,task=setup();selection=json.loads((DEST/'selection.json').read_text());f=DEST/'selection.json'
    if subprocess.check_output(['git','show',f'HEAD:{f}'])!=f.read_bytes() or selection['source_sha256']!=sha256(Path(__file__)) or selection['design_sha256']!=sha256(DESIGN):raise ValueError('Commit challenge choices/code first')
    if selection['parent_selection_sha256']!=sha256(RUN/'selection.json'):raise ValueError('Primary selection changed')
    out=DEST/'retrospective'
    if out.exists():raise FileExistsError('Retain prior challenge replay')
    out.mkdir();rows=[]
    for name,sid,age,noise,current,target,labels,y in batches(cfg,qc,pilot,task,selection['cases']):
        np.savez_compressed(out/f'{name}_s{sid}_a{age}_n{noise:g}.npz',k=current['k'],conditional_arx=y);add_rows(rows,dict(conditional_arx=y),target,labels,name,sid,'old_evaluation',task,qc,age,noise);print('REFERENCE_REPLAY',name,sid,age,noise,flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);write(DEST/'replay.json',dict(status='complete',selection_sha256=sha256(f),result_hashes={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file()},independent_confirmation=False))

def verify():
    f=DEST/'verification.json'
    if f.exists():raise FileExistsError('Retain prior verification')
    r=dict(passed=False,source_sha256=sha256(Path(__file__)),checks=[],independent_confirmation=False)
    try:
        cfg,qc,pilot,task=setup();selection=json.loads((DEST/'selection.json').read_text());replay=json.loads((DEST/'replay.json').read_text());assert replay['status']=='complete' and replay['selection_sha256']==sha256(DEST/'selection.json');assert selection['parent_selection_sha256']==sha256(RUN/'selection.json');verify_hashes(replay['result_hashes'],DEST/'retrospective');new=choose_reference(cfg,pilot,task,qc)
        for name,ages in new.items():
            for age,c in ages.items():
                old=selection['cases'][name][age];assert c['choice']==old['choice']
                for key,value in c['validation_scores'].items():np.testing.assert_allclose(value,old['validation_scores'][key],rtol=1e-12,atol=1e-12)
        rows=[]
        for name,sid,age,noise,current,target,labels,y in batches(cfg,qc,pilot,task,selection['cases']):
            with np.load(DEST/'retrospective'/f'{name}_s{sid}_a{age}_n{noise:g}.npz') as z:error=same(y,z['conditional_arx']);np.testing.assert_array_equal(current['k'],z['k'])
            add_rows(rows,dict(conditional_arx=y),target,labels,name,sid,'old_evaluation',task,qc,age,noise);r['checks'].append(dict(case=name,scenario=int(sid),age_raw=age,noise=noise,windows=len(y),maximum_error_rad=error))
        compare_rows(rows,pd.read_csv(DEST/'retrospective/metrics.csv'));r.update(passed=True,validation_choices_reproduced=20,forecast_comparisons=sum(x['windows'] for x in r['checks']),all_metrics_recomputed=True,primary_selection_unchanged=True)
    except Exception as error:r['failure']=f'{type(error).__name__}: {error}';raise
    finally:write(f,r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args();verify() if a.verify else evaluate() if a.evaluate else develop()
