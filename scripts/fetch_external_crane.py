"""Download pinned public MAT files, verify Git object hashes, never parse test observations."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
from scripts.external_crane_data import config,blob_hash

def main():
    cfg=config();cache=Path(cfg['cache']);dest=cache/'download_manifest.json'
    if dest.exists():raise FileExistsError('Retain downloaded source snapshot')
    def fetch(spec):
        p=cache/spec['path'];url=f"https://raw.githubusercontent.com/ikhemakhem/BehavioralTrajGenCrane/{cfg['commit']}/{spec['path']}"
        if p.exists():content=p.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=30) as response:content=response.read()
            if len(content)!=spec['bytes'] or blob_hash(content)!=spec['git_blob_sha1']:raise ValueError('External download integrity failure')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        assert len(content)==spec['bytes'] and blob_hash(content)==spec['git_blob_sha1']
        return dict(**spec,sha256=hashlib.sha256(content).hexdigest(),url=url)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(fetch,cfg['files']))
    value=dict(repository=cfg['repository'],commit=cfg['commit'],files=files,parsed_test_records=False,redistributed_raw_files=False)
    cache.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(value,indent=2)+'\n');print('Verified',len(files),'files',sum(f['bytes'] for f in files),'bytes')

if __name__=='__main__':main()
