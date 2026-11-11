"""Package frozen local fine-tuning evidence without overwriting an archive."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from cranepilot.data import sha256
from scripts.report_finetune_campaign import verify_evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/runs/finetune_campaign_20261107'))
    parser.add_argument('--output',type=Path,default=Path('results/finetune_campaign_20261107'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];run=root/args.run;out=root/args.output
    _,counts=verify_evidence(root,run)
    report=json.loads((out/'report.json').read_text())
    if sha256(run/'provenance.json')!=report['source_provenance_sha256']:raise ValueError('Report provenance changed')
    if sha256(root/report['document'])!=report['document_sha256']:raise ValueError('Report document changed')
    if sha256(root/'scripts/report_finetune_campaign.py')!=report['report_source_sha256']:raise ValueError('Report generator changed')
    if 'inference_latency_sha256' in report and sha256(out/'inference_latency.json')!=report['inference_latency_sha256']:
        raise ValueError('Inference latency evidence changed')
    excluded={'evidence.zip','archive_manifest.json','archive.json'}
    for name in excluded:
        if (out/name).exists():raise FileExistsError(f'Preserve existing archive: {out/name}')
    paths=[p for p in run.rglob('*') if p.is_file()]
    paths.extend(p for p in out.rglob('*') if p.is_file() and p.name not in excluded)
    for pattern in ['scripts/*finetune*.py','scripts/audit_deepc_excitation.py','tests/test_finetune*.py','configs/finetune*.json','docs/FINETUNE*.md',
                    'docs/FINETUNING*.md','docs/CONTROL_METHOD_SELECTION_RESEARCH_20261107.md',
                    'docs/CONTROL_DATA_AND_REGULARIZATION_20261107.md']:
        paths.extend(root.glob(pattern))
    paths.append(root/report['document']);paths=sorted(set(paths))
    manifest=dict(scope='This local fine-tuning stage; full replay requires restored upstream code/data/weights.',
        source_provenance_sha256=sha256(run/'provenance.json'),verified_hash_counts=counts,
        files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in paths})
    path=out/'archive_manifest.json'
    with path.open('x') as handle:json.dump(manifest,handle,indent=2,ensure_ascii=False);handle.write('\n')
    paths.append(path);destination=out/'evidence.zip'
    with ZipFile(destination,'x',ZIP_DEFLATED,compresslevel=6) as archive:
        for p in paths:archive.write(p,str(p.relative_to(root)))
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('Archive CRC failure')
        if set(archive.namelist())!={str(p.relative_to(root)) for p in paths}:raise ValueError('Archive member mismatch')
    info=dict(file=destination.name,files=len(paths),bytes=destination.stat().st_size,sha256=sha256(destination),
              manifest_sha256=sha256(path),crc_verified=True)
    with (out/'archive.json').open('x') as handle:json.dump(info,handle,indent=2);handle.write('\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
