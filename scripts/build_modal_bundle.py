"""Package only frozen parameters/code hashes; never fit a model or copy raw data."""
import json,shutil
from pathlib import Path
from cranepilot.data import sha256
from scripts.run_external_benchmark import write,verify_hashes

OUT=Path('artifacts/modal_bundle_20261107')
def main():
    if OUT.exists():raise FileExistsError('Retain earlier frozen bundle')
    dev=Path('artifacts/runs/modal_history_20261107');ref=Path('artifacts/runs/modal_reference_challenge_20261107');p=json.loads((dev/'provenance.json').read_text());q=json.loads((ref/'development.json').read_text());verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],dev);verify_hashes(q['guarded_hashes']);verify_hashes(q['result_hashes'],ref)
    for stem in ['modal_history','modal_confirmation','modal_reference_challenge']:assert json.loads(Path(f'results/{stem}_20261107/verification.json').read_text())['passed']
    selected=json.loads((dev/'selection.json').read_text());references=json.loads((ref/'selection.json').read_text());OUT.mkdir(parents=True);write(OUT/'task.json',p['config']);cases={}
    for name,case in selected['cases'].items():
        assert references[name]['source']=='challenge';target=OUT/name;target.mkdir();shutil.copyfile(dev/name/'calibration.json',target/'calibration.json');shutil.copyfile(ref/name/'selected.npz',target/'reference_arx.npz');cases[name]=dict(variant=case['modal'],calibration_file=f'{name}/calibration.json',reference_file=f'{name}/reference_arx.npz',reference_validation_mae_deg=references[name]['validation_mae_deg'])
    sources=['src/cranepilot/modal_pilot.py','src/cranepilot/data.py','scripts/modal_history.py','scripts/modal_reference_challenge.py']
    manifest=dict(scope='Experimental settled-input residual-sway forecast with4s history. No active-control/hardware/independent-new-data success claim. Previously scored public results are retrospective; frozen simple ARX retained as reference.',cases=cases,files={str(f.relative_to(OUT)):sha256(f) for f in OUT.rglob('*') if f.is_file()},source_hashes={s:sha256(Path(s)) for s in sources},source_development_sha256=sha256(dev/'provenance.json'),source_reference_development_sha256=sha256(ref/'development.json'),no_raw_data_or_training_required=True,final_goal_achieved=False)
    write(OUT/'manifest.json',manifest);print('BUNDLE',len(manifest['files'])+1,'files',sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),'bytes')
if __name__=='__main__':main()
