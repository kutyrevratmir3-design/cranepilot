import json,shutil
from pathlib import Path
from cranepilot.data import sha256
from scripts.run_dated_forecast import RUN,OUT,setup
from scripts.run_external_benchmark import write,verify_hashes

DEST=Path('artifacts/dated_bundle_20261108')
def main():
    if DEST.exists():raise FileExistsError('Preserve earlier bundle')
    cfg,_,_,_=setup();p=json.loads((RUN/'development.json').read_text());s=json.loads((RUN/'selection.json').read_text());v=json.loads((OUT/'development_verification.json').read_text());assert v['passed'] and v['development_sha256']==sha256(RUN/'development.json');verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);DEST.mkdir(parents=True);shutil.copytree(cfg['bundle'],DEST/'regime');cases={}
    for name,case in s['cases'].items():
        needed={'new/'+case['augmented_file']}|{c['key'] for c in case['reference'].values() if not c['key'].startswith('aligned/')};models={};(DEST/name).mkdir()
        for key in sorted(needed):
            source,file=key.split('/',1);src=(RUN if source=='new' else Path(cfg['old_arx']))/name/file;rel=name+'/'+source+'_'+file;shutil.copyfile(src,DEST/rel);models[key]=rel
        cases[name]=dict(reference={age:dict(key=c['key']) for age,c in case['reference'].items()},models=models,augmented_file=case['augmented_file'])
    write(DEST/'selection.json',s);sources=['src/cranepilot/dated_pilot.py','scripts/dated_forecast.py','scripts/latency_observer.py'];write(DEST/'manifest.json',dict(selected=s['selected'],methods=cfg['methods'],train_ages_raw=cfg['train_ages_raw'],cases=cases,files={str(f.relative_to(DEST)):sha256(f) for f in DEST.rglob('*') if f.is_file()},source_hashes={f:sha256(Path(f)) for f in sources},scope=cfg['scope'],development_gate_passed=s['development_gate_passed'],development_sha256=sha256(RUN/'development.json'),selection_sha256=sha256(RUN/'selection.json'),independent_confirmation=False,final_goal_achieved=False));print('BUNDLE',sum(f.is_file() for f in DEST.rglob('*')),sum(f.stat().st_size for f in DEST.rglob('*') if f.is_file()))
if __name__=='__main__':main()
