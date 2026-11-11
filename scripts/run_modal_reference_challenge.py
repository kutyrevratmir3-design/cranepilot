"""Train/VAL-only reference challenge, followed by explicitly retrospective evaluation."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.benchmark import environment_record
from scripts.modal_history import load_allowed,packets,concatenate,score,load_arx,predict_arx
from scripts.modal_reference_challenge import statistics,fit,predict,save,load
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import write,verify_hashes

CONFIG=Path('configs/modal_reference_challenge_20261107.json');RUN=Path('artifacts/runs/modal_reference_challenge_20261107')
LOCKED=[CONFIG,Path('docs/MODAL_REFERENCE_CHALLENGE_DESIGN_20261107.md'),Path('scripts/modal_reference_challenge.py'),Path('scripts/run_modal_reference_challenge.py'),Path('tests/test_modal_reference_challenge.py')]

def parents():
    c=json.loads(CONFIG.read_text());dev=Path(c['parent_development']);conf=Path(c['parent_confirmation']);p=json.loads((dev/'provenance.json').read_text());q=json.loads((conf/'provenance.json').read_text())
    for prov,root in [(p,dev),(q,conf)]:
        assert prov['status']=='complete';verify_hashes(prov['guarded_hashes']);verify_hashes(prov['result_hashes'],root)
    return c,p['config'],dev,conf

def prediction(case,p,cfg,out,parent):
    if case['source']=='original':return predict_arx(load_arx(parent/'strong_arx.npz'),p,cfg)
    return predict(load(out/'selected.npz'),p,cfg)

def develop():
    c,cfg,parent,conf=parents()
    for path in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit first {path}')
    if RUN.exists():raise FileExistsError('Retain previous challenge')
    RUN.mkdir(parents=True);prov=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=c,task=cfg,guarded_hashes={str(p):sha256(p) for p in LOCKED+[parent/'provenance.json',parent/'selection.json',conf/'provenance.json']},environment=environment_record(),stage='Post-hoc comparator robustness; old holdout already observed',holdout_read_during_fitting=False,independent_confirmation=False);write(RUN/'development.json',prov);selection={};rows=[]
    try:
        for family,fold in cfg['families'].items():
            tr=load_allowed(cfg,fold['train'],'train');va=load_allowed(cfg,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';out=RUN/name;out.mkdir();(out/'candidates').mkdir();cal=LengthCalibration(**json.loads((parent/name/'calibration.json').read_text()));train=concatenate([packets(s,axis,cfg,cal)[0] for s in tr.values()]);val=concatenate([packets(s,axis,cfg,cal)[0] for s in va.values()]);original=predict_arx(load_arx(parent/name/'strong_arx.npz'),val,cfg);best=dict(source='original',validation_mae_deg=score(original,val))
                for representation in c['representations']:
                    for form in c['forms']:
                        stats=statistics(train,cfg,representation,form)
                        for alpha in c['ridge_grid']:
                            m=fit(stats,alpha);file=out/'candidates'/f'{representation}_{form}_{alpha:g}.npz';save(file,m);y=predict(m,val,cfg);np.testing.assert_array_equal(y,predict(load(file),val,cfg));value=score(y,val);rows.append(dict(case=name,representation=representation,form=form,alpha=alpha,validation_mae_deg=value,file=str(file.relative_to(RUN))))
                            if value<best['validation_mae_deg']:best=dict(source='challenge',validation_mae_deg=value,file=str(file.relative_to(RUN)),representation=representation,form=form,alpha=alpha)
                if best['source']=='challenge':save(out/'selected.npz',load(RUN/best['file']));best['checkpoint_sha256']=sha256(out/'selected.npz')
                y=prediction(best,val,cfg,out,parent/name);np.savez_compressed(out/'validation.npz',k=val['k'],scenario=val['scenario'],challenged_arx=y,original_arx=original);selection[name]=best;print('CHALLENGE',name,json.dumps(best),flush=True)
        pd.DataFrame(rows).to_csv(RUN/'candidate_validation.csv',index=False);write(RUN/'selection.json',selection);prov.update(status='complete',selection_sha256=sha256(RUN/'selection.json'),result_hashes={str(p.relative_to(RUN)):sha256(p) for p in RUN.rglob('*') if p.is_file() and p.name!='development.json'})
    except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(RUN/'development.json',prov)

def evaluate():
    c,cfg,parent,conf=parents();p=json.loads((RUN/'development.json').read_text());assert p['status']=='complete';verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);selection=json.loads((RUN/'selection.json').read_text());out=RUN/'retrospective'
    if out.exists():raise FileExistsError('Retain previous evaluation')
    # This explicit freeze precedes read access for this stage. Records were seen earlier.
    out.mkdir();prov=dict(status='running',development_sha256=sha256(RUN/'development.json'),selection_sha256=sha256(RUN/'selection.json'),independent_confirmation=False,scope='Retrospective robustness challenge; never a new holdout')
    write(out/'provenance.json',prov);rows=[]
    try:
        for family,fold in cfg['families'].items():
            records=load_confirmation(cfg,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**json.loads((parent/name/'calibration.json').read_text()))
                for sid,seq in records.items():
                    part,_=packets(seq,axis,cfg,cal)
                    if not len(part['k']):raise ValueError('Empty case retained as failure')
                    y=prediction(selection[name],part,cfg,RUN/name,parent/name)
                    with np.load(conf/f'{name}_s{sid}.npz') as f:
                        np.testing.assert_array_equal(f['k'],part['k']);old={k:f[k] for k in ['selected_modal','strong_arx','local_ar']}
                    np.savez_compressed(out/f'{name}_s{sid}.npz',k=part['k'],challenged_arx=y)
                    for method,forecast in dict(challenged_arx=y,**old).items():
                        for j,h in enumerate(cfg['horizons_raw']):rows.append(dict(case=name,scenario=sid,method=method,horizon_s=h*.01,mae_deg=float(np.abs(np.rad2deg(forecast[:,j]-part['target'][:,j])).mean())))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);prov.update(status='complete',result_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
    except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(out/'provenance.json',prov)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    evaluate() if args.evaluate else develop()
