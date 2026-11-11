"""One conditional-holdout evaluation after the complete method/reference freeze."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import Scenario,sha256
from cranepilot.physics import LengthCalibration
from cranepilot.benchmark import environment_record
from scripts.modal_history import packets,modal,local_ar,predict_arx,load_arx
from scripts.run_external_benchmark import write,verify_hashes

CONFIG=Path('configs/modal_confirmation_20261107.json');RUN=Path('artifacts/runs/modal_confirmation_20261107')

def load_confirmation(cfg,ids):
    if not ids or not set(ids)<=set(cfg['sealed_confirmation_ids']):raise ValueError('Only preregistered conditional holdouts allowed')
    root=Path(cfg['data']);manifest=json.loads((root/'manifest.json').read_text());meta={m['sid']:m for m in manifest['scenarios']}
    with np.load(root/'dataset.npz',allow_pickle=False) as z:
        time=z['time'].copy()
        return {sid:Scenario(sid,time.copy(),{name:z[f's{sid}_{name}'].copy() for name in meta[sid]['signals']},meta[sid]) for sid in ids}

def forecasts(p,cfg,case,parent):
    y,diag=modal(p,cfg,case['modal'])
    output=dict(selected_modal=y,strong_arx=predict_arx(load_arx(parent/'strong_arx.npz'),p,cfg),local_ar=local_ar(p,cfg,**case['local_ar']),fixed=modal(p,cfg,'fixed')[0],one_mode=modal(p,cfg,'one_mode')[0],persistence=np.repeat(p['theta'][:,:1],len(cfg['horizons_raw']),axis=1))
    return output,diag

def main():
    freeze=json.loads(CONFIG.read_text());parent=Path(freeze['parent_run']);verify_hashes(freeze['frozen_parent_hashes']);p=json.loads((parent/'provenance.json').read_text());selection=json.loads((parent/'selection.json').read_text());cfg=p['config'];v=json.loads(Path(freeze['development_verification']).read_text())
    assert p['status']=='complete' and v['passed'] and selection['development_gate_passed'] and v['development_gate_reproduced'] and not p['confirmation_records_read']
    verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],parent)
    locked=[CONFIG,Path('scripts/run_modal_confirmation.py'),Path('docs/MODAL_CONFIRMATION_DESIGN_20261107.md')]
    for path in locked:
        if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit before opening holdout: {path}')
    if RUN.exists():raise FileExistsError('One evaluation only; retain prior attempt')
    RUN.mkdir(parents=True);prov=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),freeze=freeze,config=cfg,guarded_hashes={str(f):sha256(f) for f in locked},environment=environment_record(),records_read=[],independently_collected_confirmation=False,qualification=cfg['confirmation_status'])
    write(RUN/'provenance.json',prov);coverage=[];metrics=[];empty=[]
    try:
        for family,fold in cfg['families'].items():
            records=load_confirmation(cfg,fold['confirmation']);prov['records_read']+=fold['confirmation']
            for axis in fold['axes']:
                name=f'{family}_{axis}';case=selection['cases'][name];cal=LengthCalibration(**json.loads((parent/name/'calibration.json').read_text()))
                for sid,seq in records.items():
                    part,c=packets(seq,axis,cfg,cal);coverage.append(dict(case=name,**c))
                    if not len(part['k']):empty.append(dict(case=name,scenario=sid));continue
                    y,diag=forecasts(part,cfg,case,parent/name);np.savez_compressed(RUN/f'{name}_s{sid}.npz',k=part['k'],**y,**{'diagnostic_'+k:v for k,v in diag.items()})
                    for method,forecast in y.items():
                        for j,h in enumerate(cfg['horizons_raw']):
                            err=np.rad2deg(forecast[:,j]-part['target'][:,j]);metrics.append(dict(case=name,scenario=sid,axis=axis,method=method,horizon_s=h*.01,mae_deg=float(np.abs(err).mean()),rmse_deg=float(np.sqrt((err**2).mean())),windows=len(part['k'])))
                    print('SCORED',name,sid,'origins',len(part['k']),flush=True)
        pd.DataFrame(metrics).to_csv(RUN/'metrics.csv',index=False);pd.DataFrame(coverage).to_csv(RUN/'coverage.csv',index=False);write(RUN/'empty_cases.json',empty)
        prov.update(status='complete',records_read=sorted(set(prov['records_read'])),empty_cases=empty,selection_unchanged_sha256=sha256(parent/'selection.json'),result_hashes={str(f.relative_to(RUN)):sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name!='provenance.json'})
    except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(RUN/'provenance.json',prov)

if __name__=='__main__':main()
