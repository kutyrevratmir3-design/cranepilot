"""Archive own matched/shared development evidence, without the external raw data."""
import json,shutil
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256
from scripts.run_external_benchmark import verify_hashes

def main():
    for stage in ['matched_adaptation','shared_response']:
        stem=stage+'_20261107';out=Path('results')/stem;run=Path('artifacts/runs')/stem;dest=out/'frozen_run'
        assert json.loads((out/'verification.json').read_text())['passed']
        p=json.loads((run/'provenance.json').read_text());verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],run)
        r=json.loads((out/'report.json').read_text());assert sha256(Path(r['document']))==r['document_sha256']
        if dest.exists() or (out/'evidence.zip').exists():raise FileExistsError('Preserve earlier archive')
        shutil.copytree(run,dest)
        files=[f for f in out.rglob('*') if f.is_file()]
        for pattern in [f'scripts/*{stage}*.py',f'tests/test_{stage}.py',f'configs/{stem}.json',f'docs/{stage.upper()}*.md','scripts/package_development_stages.py']:
            files.extend(f for f in Path('.').glob(pattern) if f.is_file())
        files=sorted(set(files));manifest=out/'archive_manifest.json'
        manifest.write_text(json.dumps(dict(scope='Own development code, checkpoints, derived forecasts and metrics; no external raw MAT/source files; parent repository stages and verified source download required.',files={str(f):dict(bytes=f.stat().st_size,sha256=sha256(f)) for f in files}),indent=2)+'\n');files.append(manifest)
        target=out/'evidence.zip'
        with ZipFile(target,'x',ZIP_DEFLATED,compresslevel=6) as z:
            for f in files:z.write(f,str(f))
        with ZipFile(target) as z:assert z.testzip() is None and set(z.namelist())==set(map(str,files))
        info=dict(bytes=target.stat().st_size,files=len(files),sha256=sha256(target),manifest_sha256=sha256(manifest),crc_verified=True,external_raw_files_included=False)
        (out/'archive.json').write_text(json.dumps(info,indent=2)+'\n');print(stem,json.dumps(info))
if __name__=='__main__':main()
