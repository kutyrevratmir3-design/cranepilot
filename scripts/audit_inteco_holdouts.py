"""Metadata-only audit for previously unscored Inteco scenarios; never open signal arrays."""
import csv,json,re
from pathlib import Path
from cranepilot.data import sha256

CANDIDATES={9,13,21,22,23,32,33}
EXTERNAL=('external_benchmark','matched_adaptation','shared_response')
OUT=Path('results/inteco_usage_audit_20261107')


def main():
    if OUT.exists():raise FileExistsError('Retain prior usage audit')
    OUT.mkdir(parents=True);hits=[];errors=[];hashes={};csv_count=0;json_count=0;metadata_hits=[]
    for base in [Path('results'),Path('artifacts/runs'),Path('configs')]:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.suffix not in ['.json','.csv'] or OUT in p.parents or 'frozen_run' in p.parts or any(s in str(p) for s in EXTERNAL) or p.name=='archive_manifest.json':continue
            try:
                if p.suffix=='.csv':
                    with p.open(newline='') as f:
                        reader=csv.DictReader(f);columns=set(reader.fieldnames or [])&{'scenario','scenario_id','sid'}
                        if not columns:continue
                        csv_count+=1;hashes[str(p)]=sha256(p);ids=set()
                        for row in reader:
                            for key in columns:
                                try:
                                    v=float(row[key]);sid=int(v)
                                    if v==sid and sid in CANDIDATES:ids.add(sid)
                                except (ValueError,TypeError):pass
                        if ids:hits.append(dict(path=str(p),kind='score_table',ids=sorted(ids)))
                else:
                    data=json.loads(p.read_text());json_count+=1;hashes[str(p)]=sha256(p)
                    def walk(o,where=''):
                        if isinstance(o,dict):
                            for key,value in o.items():
                                if key in {'train','validation','test','evaluation','load_stress','train_ids','validation_ids','test_ids','scenario_ids','real_scenarios'} and isinstance(value,list) and all(isinstance(v,int) and not isinstance(v,bool) for v in value):
                                    ids=CANDIDATES&set(value)
                                    if ids:hits.append(dict(path=str(p),kind='explicit_role',key=where+'.'+key,ids=sorted(ids)))
                                walk(value,where+'.'+key)
                            for key in ['sid','scenario']:
                                if isinstance(o.get(key),(int,float)) and o[key] in CANDIDATES:metadata_hits.append(dict(path=str(p),key=where+'.'+key,sid=o[key]))
                        elif isinstance(o,list):
                            for i,v in enumerate(o):walk(v,where+f'[{i}]')
                    walk(data)
            except Exception as e:errors.append(dict(path=str(p),error=str(e)))
    predictions=[]
    for p in Path('artifacts/runs').rglob('predictions_s*.npz'):
        match=re.fullmatch(r'predictions_s(\d+)\.npz',p.name)
        if match and int(match[1]) in CANDIDATES:predictions.append(str(p))
    manifest=Path('data/processed/manifest.json');meta=json.loads(manifest.read_text());rows=[{k:r[k] for k in ['sid','file_name','family','fixed_length','L_min_m','L_max_m','mass_g','shape']} for r in meta['scenarios'] if r['sid'] in CANDIDATES]
    info=dict(passed=not(hits or errors or predictions or metadata_hits),candidate_ids=sorted(CANDIDATES),csv_tables_checked=csv_count,json_files_checked=json_count,explicit_role_or_score_hits=hits,scalar_id_hits=metadata_hits,prediction_file_hits=predictions,errors=errors,
        candidate_metadata=rows,signal_arrays_opened=False,manifest_sha256=sha256(manifest),source_sha256=sha256(Path(__file__)),scanned_file_hashes=hashes,
        interpretation='No saved fitting/validation/score use found in audited main-project artifacts; initial dataset import/metadata and possible unrecorded human access are not ruled out. This is not a new recording or an independently collected experiment.')
    (OUT/'audit.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps({k:v for k,v in info.items() if k!='scanned_file_hashes'},indent=2))

if __name__=='__main__':main()
