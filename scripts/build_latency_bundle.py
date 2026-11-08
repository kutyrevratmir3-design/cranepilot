"""Freeze a known-age adapter and its matched reference without dataset dependencies."""
import json,shutil
from pathlib import Path
from cranepilot.data import sha256
from scripts.run_latency_observer import RUN,OUT,setup
from scripts.run_external_benchmark import write,verify_hashes

DEST=Path('artifacts/latency_bundle_20261108')
def main():
    if DEST.exists():raise FileExistsError('Preserve earlier bundle')
    cfg,_,_,_=setup();p=json.loads((RUN/'development.json').read_text());s=json.loads((RUN/'selection.json').read_text());v=json.loads((OUT/'development_verification.json').read_text());assert v['passed'] and v['development_sha256']==sha256(RUN/'development.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);DEST.mkdir(parents=True);shutil.copytree(cfg['bundle'],DEST/'regime');cases={}
    for name,case in s['cases'].items():
        (DEST/name).mkdir();ref=case['reference'].copy()
        if ref['type']=='augmented':shutil.copyfile(RUN/name/ref['file'],DEST/name/'reference.npz');ref['file']=name+'/reference.npz'
        shutil.copyfile(RUN/name/case['augmented_file'],DEST/name/'augmented.npz');cases[name]=dict(reference=ref,augmented=name+'/augmented.npz',family=case['family'],axis=case['axis'])
    write(DEST/'selection.json',s);sources=['src/cranepilot/latency_pilot.py','scripts/latency_observer.py'];write(DEST/'manifest.json',dict(selected=s['selected'],cases=cases,files={str(f.relative_to(DEST)):sha256(f) for f in DEST.rglob('*') if f.is_file()},source_hashes={f:sha256(Path(f)) for f in sources},scope='Experimental forecast with known scalar angle age0..15samples100Hz. Dated angle history and current-clock support histories, no future measured inputs. Missing head estimates are excluded from backtest truth. Unknown-delay identification and closed-loop/hardware superiority not established; retrospective/synthetic development only.',development_gate_passed=s['development_gate_passed'],development_sha256=sha256(RUN/'development.json'),selection_sha256=sha256(RUN/'selection.json'),independent_confirmation=False,final_goal_achieved=False));print('BUNDLE',sum(f.is_file() for f in DEST.rglob('*')),sum(f.stat().st_size for f in DEST.rglob('*') if f.is_file()))
if __name__=='__main__':main()
