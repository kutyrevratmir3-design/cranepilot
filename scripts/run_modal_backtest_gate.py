"""Train/VAL reliability development and a separate, labelled retrospective replay."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.benchmark import environment_record
from scripts.modal_history import load_allowed,packets,concatenate,score
from scripts.modal_reference_challenge import statistics,fit,predict,save,load
from scripts.modal_backtest_gate import sliced,backtest,compose
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_modal_reference_challenge import prediction as long_prediction
from scripts.run_external_benchmark import write,verify_hashes

CONFIG=Path('configs/modal_backtest_gate_20261107.json');RUN=Path('artifacts/runs/modal_backtest_gate_20261107')
LOCKED=[CONFIG,Path('docs/MODAL_BACKTEST_GATE_DESIGN_20261107.md'),Path('scripts/modal_backtest_gate.py'),Path('scripts/run_modal_backtest_gate.py'),Path('tests/test_modal_backtest_gate.py')]

def parents():
    g=json.loads(CONFIG.read_text());base=Path(g['parent_development']);reference=Path(g['parent_reference']);confirmation=Path(g['parent_confirmation']);p=json.loads((base/'provenance.json').read_text());q=json.loads((reference/'development.json').read_text());r=json.loads((confirmation/'provenance.json').read_text())
    for prov,root in [(p,base),(q,reference),(r,confirmation)]:assert prov['status']=='complete';verify_hashes(prov['guarded_hashes']);verify_hashes(prov['result_hashes'],root)
    return g,p['config'],base,reference,confirmation,json.loads((reference/'selection.json').read_text())

def references(case,p,cfg,cal,base,reference,prior,short):
    long=long_prediction(prior[case],p,cfg,reference/case,base/case);short_y=predict(short,sliced(p,0,61,cal),cfg)
    return long,short_y

def develop():
    g,cfg,base,reference,confirmation,prior=parents()
    for path in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit first {path}')
    if RUN.exists():raise FileExistsError('Preserve all attempts')
    RUN.mkdir(parents=True);paths=LOCKED+[base/'provenance.json',reference/'development.json',reference/'selection.json',confirmation/'provenance.json',Path('scripts/modal_history.py'),Path('scripts/modal_reference_challenge.py')]
    p=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=g,task=cfg,guarded_hashes={str(f):sha256(f) for f in paths},environment=environment_record(),holdout_read_during_development=False,all_old_records_previously_opened=True,independent_confirmation=False);write(RUN/'development.json',p);rows=[];choices={};case_scores={}
    try:
        for family,fold in cfg['families'].items():
            train=load_allowed(cfg,fold['train'],'train');val=load_allowed(cfg,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';out=RUN/name;out.mkdir();cal=LengthCalibration(**json.loads((base/name/'calibration.json').read_text()));tr=concatenate([packets(s,axis,cfg,cal)[0] for s in train.values()]);va=concatenate([packets(s,axis,cfg,cal)[0] for s in val.values()]);short_train=sliced(tr,0,61,cal);short_train['target']=tr['target'].copy();old=prior[name];assert old['source']=='challenge'
                m=fit(statistics(short_train,cfg,old['representation'],old['form']),old['alpha']);save(out/'short_arx.npz',m);long,short=references(name,va,cfg,cal,base,reference,prior,m);np.testing.assert_array_equal(short,predict(load(out/'short_arx.npz'),sliced(va,0,61,cal),cfg));scores=dict(long_arx=score(long,va),short_arx=score(short,va));selected_ref='short' if scores['short_arx']<scores['long_arx'] else 'long';yr=short if selected_ref=='short' else long
                with np.load(base/name/'validation_forecasts.npz') as f:np.testing.assert_array_equal(va['k'],f['k']);ym=f['selected_modal']
                evidence=backtest(va,cfg,g,cal,m);preds=dict(long_arx=long,short_arx=short);weights={}
                for variant in g['variants']:
                    y,w=compose(ym,yr,evidence,variant);preds[variant]=y;weights['weight_'+variant]=w;scores[variant]=score(y,va)
                    for sid in fold['validation']:
                        mask=va['scenario']==sid
                        for j,h in enumerate(cfg['horizons_raw']):rows.append(dict(case=name,scenario=sid,method=variant,horizon_s=h*.01,mae_deg=float(np.abs(np.rad2deg(y[mask,j]-va['target'][mask,j])).mean())))
                np.savez_compressed(out/'validation.npz',k=va['k'],scenario=va['scenario'],**preds,**weights,**evidence);choices[name]=dict(family=family,axis=axis,reference=selected_ref,reference_validation_scores={k:scores[k] for k in ['long_arx','short_arx']},short_model_sha256=sha256(out/'short_arx.npz'),short_model_representation=old['representation'],short_model_form=old['form'],short_model_alpha=old['alpha']);case_scores[name]=scores
                print('CASE',name,json.dumps(scores),flush=True)
        means={variant:float(np.mean([s[variant] for s in case_scores.values()])) for variant in g['variants']};selected=min(means,key=means.get);ref=means['reference'];win=sum(s[selected]<s['reference'] for s in case_scores.values());change=100*(means[selected]/ref-1);gate=bool(change<=-10 and win>=4);choice=dict(selected=selected,cases=choices,case_scores=case_scores,validation_means=means,validation_change_percent=change,validation_case_wins=win,development_gate_passed=gate,independent_confirmation=False,final_goal_achieved=False)
        pd.DataFrame(rows).to_csv(RUN/'validation_metrics.csv',index=False);write(RUN/'selection.json',choice);p.update(status='complete',selection_sha256=sha256(RUN/'selection.json'),result_hashes={str(f.relative_to(RUN)):sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name!='development.json'});print('SELECTED',selected,means,'CHANGE%',change,'WINS',win,'GATE',gate,flush=True)
    except Exception as e:p.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(RUN/'development.json',p)

def evaluate():
    g,cfg,base,reference,confirmation,prior=parents();p=json.loads((RUN/'development.json').read_text());assert p['status']=='complete';verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);selection=json.loads((RUN/'selection.json').read_text());out=RUN/'retrospective'
    if out.exists():raise FileExistsError('Retain prior evaluation')
    out.mkdir();q=dict(status='running',selection_sha256=sha256(RUN/'selection.json'),development_sha256=sha256(RUN/'development.json'),independent_confirmation=False,scope='Retrospective development on opened public records');write(out/'provenance.json',q);rows=[]
    try:
        for family,fold in cfg['families'].items():
            records=load_confirmation(cfg,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**json.loads((base/name/'calibration.json').read_text()));m=load(RUN/name/'short_arx.npz');case=selection['cases'][name]
                for sid,s in records.items():
                    part,_=packets(s,axis,cfg,cal);long,short=references(name,part,cfg,cal,base,reference,prior,m);yr=short if case['reference']=='short' else long
                    with np.load(confirmation/f'{name}_s{sid}.npz') as f:np.testing.assert_array_equal(part['k'],f['k']);ym=f['selected_modal']
                    evidence=backtest(part,cfg,g,cal,m);predictions=dict(long_arx=long,short_arx=short);weights={}
                    for variant in g['variants']:
                        y,w=compose(ym,yr,evidence,variant);predictions[variant]=y;weights['weight_'+variant]=w
                    predictions['selected']=predictions[selection['selected']].copy();np.savez_compressed(out/f'{name}_s{sid}.npz',k=part['k'],**predictions,**weights,**evidence)
                    for method,y in predictions.items():
                        for j,h in enumerate(cfg['horizons_raw']):rows.append(dict(case=name,scenario=sid,method=method,horizon_s=h*.01,mae_deg=float(np.abs(np.rad2deg(y[:,j]-part['target'][:,j])).mean())))
                    print('RETROSPECTIVE',name,sid,flush=True)
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);q.update(status='complete',result_hashes={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='provenance.json'})
    except Exception as e:q.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(out/'provenance.json',q)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');args=parser.parse_args();evaluate() if args.evaluate else develop()
