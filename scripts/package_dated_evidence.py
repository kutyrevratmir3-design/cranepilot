"""Archive this stage's public evidence, then verify every entry and live hash."""
import json,zipfile
from pathlib import Path
from cranepilot.data import sha256
from scripts.run_dated_forecast import RUN,OUT,LOCKED
from scripts.run_external_benchmark import write,verify_hashes

def main():
    archive=OUT/'dated_forecast_evidence.zip';manifest=OUT/'EVIDENCE_MANIFEST.json';dest=OUT/'package_verification.json'
    if archive.exists() or manifest.exists() or dest.exists():raise FileExistsError('Retain prior evidence package')
    v=json.loads((OUT/'verification.json').read_text());assert v['passed'];p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective')
    for base,mirror,rel in [(RUN,OUT/'frozen_development',['development.json']+list(p['result_hashes'])),(RUN/'retrospective',OUT/'frozen_replay',['provenance.json']+list(q['result_hashes']))]:
        for f in rel:
            if (base/f).read_bytes()!=(mirror/f).read_bytes():raise ValueError(f'Mirror differs: {mirror/f}')
    extra=[Path(x) for x in ['scripts/report_dated_forecast.py','scripts/audit_dated_length.py','scripts/package_dated_evidence.py','docs/DATED_FORECAST_2026_11_08_RU.md','docs/DATED_BUNDLE_20261108.md','docs/DELAY_CONTROL_LITERATURE_20261108_RU.md','docs/DATA_ACCESS_RECHECK_20261108.md','docs/CI_FIXTURE_20261108.md','docs/AUTHOR_DATA_INTAKE_20261107_RU.md','RESEARCH_STATUS_2026_11_07.md','artifacts/runs/external_benchmark_20261107/selection.json']];files=sorted(set(LOCKED+extra+[f for base in [OUT,Path('artifacts/dated_bundle_20261108')] for f in base.rglob('*') if f.is_file()]),key=str);hashes={str(f):sha256(f) for f in files};write(manifest,dict(files=hashes,file_bytes={str(f):f.stat().st_size for f in files},scope='This stage public source, weights, forecasts, metrics and checks; no original raw/processed dataset or private correspondence. Full scientific replay also uses previously published source/model files and original data; the portable API requires only the bundle and main source/dependencies.',independent_confirmation=False,final_goal_achieved=False));files.append(manifest)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,str(f))
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;assert set(z.namelist())==set(map(str,files))
        for f in files:
            if hashlib.sha256(z.read(str(f))).hexdigest()!=sha256(f):raise ValueError(f'Archive mismatch: {f}')
    verify_hashes(hashes);write(dest,dict(passed=True,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive),entries=len(files),manifest_sha256=sha256(manifest),all_entry_crc_and_sha256_verified=True,all_manifest_live_files_verified=True,development_and_replay_mirrors_byte_identical=True,package_report_excluded_from_its_own_archive=True));print('PACKAGE',archive.stat().st_size,'bytes',len(files),'files',sha256(archive))
if __name__=='__main__':main()
