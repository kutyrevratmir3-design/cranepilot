"""Replay fixed attribution controls after the registered selection is frozen."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import load_allowed
from scripts.run_drive_conditioned import RUN,OUT,setup,pack,metric_rows
from scripts.drive_attribution import controls
from scripts import drive_conditioned as dc
from scripts.run_modal_confirmation import load_confirmation
from scripts.active_qualification import qualification_packets
from scripts.run_external_benchmark import write,verify_hashes

LOCKED=[Path('docs/DRIVE_ATTRIBUTION_DESIGN_20261107.md'),Path('scripts/drive_attribution.py'),Path('scripts/run_drive_attribution.py')]

def main():
    for path in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit diagnostic first: {path}')
    cfg,qc,pilot,task=setup();dev=json.loads((RUN/'development.json').read_text());ret=json.loads((RUN/'retrospective/provenance.json').read_text());assert dev['status']==ret['status']=='complete';verify_hashes(dev['guarded_hashes']);verify_hashes(dev['result_hashes'],RUN);verify_hashes(ret['result_hashes'],RUN/'retrospective');selection=json.loads((RUN/'selection.json').read_text());out=RUN/'attribution'
    if out.exists():raise FileExistsError('Retain all attribution attempts')
    out.mkdir();p=dict(status='running',source_hashes={str(f):sha256(f) for f in LOCKED},selection_sha256=sha256(RUN/'selection.json'),development_sha256=sha256(RUN/'development.json'),late_diagnostic_after_trolley_validation=True,used_for_selection=False,independent_confirmation=False,final_goal_achieved=False);write(out/'provenance.json',p);rows=[]
    try:
        for family,fold in task['families'].items():
            validation=load_allowed(task,fold['validation'],'validation');records=load_confirmation(task,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.cases[name]['calibration']);case=selection['cases'][name];base=RUN/name;prior=dc.load(base/'prior.npz');long_model=dc.load(base/case['long_model']);short_model=dc.load(base/'short_arx.npz');batches=[('validation',pack(validation,axis,task,qc,cal))]+[('old_evaluation',qualification_packets(s,axis,task,qc,cal)) for s in records.values()]
                for role,(packet,target,labels) in batches:
                    predictions=controls(packet,cfg,task,pilot,name,cal,prior,long_model,short_model,case['reference'])
                    for sid in np.unique(packet['scenario']):
                        mask=packet['scenario']==sid;np.savez_compressed(out/f'{role}_{name}_s{sid}.npz',k=packet['k'][mask],**{k:v[mask] for k,v in predictions.items()});rows+=metric_rows({k:v[mask] for k,v in predictions.items()},target[mask],{k:v[mask] for k,v in labels.items()},name,sid,role,task,qc);print('ATTRIBUTION',role,name,int(sid),flush=True)
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);p.update(status='complete',result_hashes={str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file() and f.name!='provenance.json'})
    except Exception as e:p.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(out/'provenance.json',p)

if __name__=='__main__':main()
