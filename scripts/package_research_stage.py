"""Compact local evidence archive; no external publication operation."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json
from cranepilot.data import sha256


def main():
    root=Path.cwd();output=Path('results/latest/research_20261103_evidence.zip')
    if output.exists():raise FileExistsError('Preserve earlier archive; choose a new stage filename')
    runs=['forceaware_campaign','forceaware_campaign_v2','phase_normalized_campaign',
          'measurement_consistent_campaign','cross_length_campaign','zero_prior_gp_campaign']
    files=[]
    for name in runs:
        files.extend(p for p in (Path('artifacts/runs')/name).rglob('*') if p.is_file() and p.suffix!='.joblib')
    actor=Path('artifacts/runs/actuator_development')
    files.extend(p for p in actor.iterdir() if p.is_file())
    for name in ['src/cranepilot','scripts','configs','tests','docs','examples']:
        files.extend(p for p in Path(name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.zip'])
    files.extend(Path(name) for name in ['README.md','LICENSE','CITATION.cff','pyproject.toml','requirements-lock.txt'])
    files.extend(p for p in Path('results/latest').iterdir() if p.is_file() and p.suffix in ['.csv','.json','.txt'])
    files.extend(p for p in Path('paper/figures').glob('*20261103*') if p.is_file())
    files=sorted(set(files))
    manifest={str(p).replace('\\','/'):sha256(p) for p in files}
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p).replace('\\','/'))
        z.writestr('archive_manifest.json',json.dumps(manifest,indent=2))
        z.writestr('ARCHIVE_README.txt','Local research evidence, 3 November 2026. Extract into project root. Includes all new neural weights, fold-specific actors, prediction traces, metrics, source, tests and configurations. GP joblib checkpoints are retained locally but excluded from this compact archive; reproduce them with the supplied scripts. Raw data are separate: DOI 10.17632/x69hyyv3zw.2, CC BY 4.0; cite the original dataset article DOI 10.1016/j.dib.2025.111822. PyTorch checkpoints are state_dict tensors; use weights_only=True. The reference model card describes scope and limitations. No new hardware or universal best-model claim.')
    with ZipFile(output) as z:
        if z.testzip() is not None:raise RuntimeError('Archive integrity failure')
        for path,expected in manifest.items():
            import hashlib
            if hashlib.sha256(z.read(path)).hexdigest()!=expected:raise RuntimeError('Archive content mismatch')
    print(json.dumps({'archive':str(output),'files':len(files),'bytes':output.stat().st_size,'sha256':sha256(output)},indent=2))


if __name__=='__main__':main()
