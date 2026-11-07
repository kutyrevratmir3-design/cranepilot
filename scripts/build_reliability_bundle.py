"""Package frozen development-selected blend; no training or old-data selection."""
import json,shutil
from pathlib import Path
from cranepilot.data import sha256
from scripts.run_external_benchmark import write,verify_hashes

OUT=Path('artifacts/reliability_bundle_20261107')
def main():
    if OUT.exists():raise FileExistsError('Preserve earlier bundle')
    run=Path('artifacts/runs/modal_backtest_gate_20261107');p=json.loads((run/'development.json').read_text());s=json.loads((run/'selection.json').read_text());v=json.loads(Path('results/modal_backtest_gate_20261107/verification.json').read_text());assert p['status']=='complete' and v['passed'] and v['development_sha256']==sha256(run/'development.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],run)
    OUT.mkdir(parents=True);shutil.copytree('artifacts/modal_bundle_20261107',OUT/'base');write(OUT/'gate.json',p['config']);cases={}
    for name,case in s['cases'].items():
        target=OUT/name;target.mkdir();shutil.copyfile(run/name/'short_arx.npz',target/'short_arx.npz');cases[name]=dict(reference=case['reference'],short_model_file=f'{name}/short_arx.npz')
    sources=['src/cranepilot/reliability_pilot.py','src/cranepilot/physics.py','scripts/modal_backtest_gate.py','scripts/modal_reference_challenge.py'];manifest=dict(selected=s['selected'],cases=cases,scope='Experimental causal past-error blend, same4s settled-input task. Retrospective development only; no independent new-data, active-controller or hardware success established.',files={str(f.relative_to(OUT)):sha256(f) for f in OUT.rglob('*') if f.is_file()},source_hashes={f:sha256(Path(f)) for f in sources},development_sha256=sha256(run/'development.json'),selection_sha256=sha256(run/'selection.json'),verification_sha256=sha256(Path('results/modal_backtest_gate_20261107/verification.json')),no_raw_data_or_training_required=True,final_goal_achieved=False)
    write(OUT/'manifest.json',manifest);print('BUNDLE',len(manifest['files'])+1,'files',sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),'bytes')
if __name__=='__main__':main()
