"""Copy a validation-frozen routing model without copying training measurements."""
import json,shutil
from pathlib import Path
from cranepilot.data import sha256
from scripts.run_regime_fusion import RUN,OUT,setup
from scripts.run_external_benchmark import write,verify_hashes

DEST=Path('artifacts/regime_bundle_20261107')
def main():
    if DEST.exists():raise FileExistsError('Preserve prior bundle')
    cfg,_,_,_,_,_=setup();p=json.loads((RUN/'development.json').read_text());s=json.loads((RUN/'selection.json').read_text());v=json.loads((OUT/'development_verification.json').read_text());assert p['status']=='complete' and v['passed'] and v['development_sha256']==sha256(RUN/'development.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);DEST.mkdir(parents=True);shutil.copytree(cfg['bundle'],DEST/'legacy');cases={}
    for name,case in s['parent_cases'].items():
        (DEST/name).mkdir();parent=Path(cfg['parent'])/name;names=dict(long=case['long_model'],short='short_arx.npz',prior='prior.npz');files={}
        for key,source in names.items():
            target=DEST/name/(key+'.npz');shutil.copyfile(parent/source,target);files[key]=str(target.relative_to(DEST))
        cases[name]=dict(reference=case['reference'],**files)
    write(DEST/'selection.json',s);sources=['src/cranepilot/regime_pilot.py','scripts/regime_fusion.py','scripts/active_qualification.py','scripts/drive_conditioned.py'];write(DEST/'manifest.json',dict(selected=s['selected'],cases=cases,files={str(f.relative_to(DEST)):sha256(f) for f in DEST.rglob('*') if f.is_file()},source_hashes={f:sha256(Path(f)) for f in sources},scope='Separate experimental history-only4s100Hz predictor with causal regime routing. Retrospective development only. Future support is assumed, no future input-plan/counterfactual native-actuator/control/hardware guarantee; independent recordings still required.',development_sha256=sha256(RUN/'development.json'),selection_sha256=sha256(RUN/'selection.json'),development_verification_sha256=sha256(OUT/'development_verification.json'),training_data_required=False,independent_confirmation=False,final_goal_achieved=False));print('FROZEN_BUNDLE',sum(f.is_file() for f in DEST.rglob('*')),sum(f.stat().st_size for f in DEST.rglob('*') if f.is_file()))
if __name__=='__main__':main()
