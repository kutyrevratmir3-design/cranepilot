"""Single-packet wall-clock diagnostic on fixed development samples, no tuning."""
import json,time
from pathlib import Path
import numpy as np
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import load_allowed,packets,concatenate,modal
from scripts.run_external_benchmark import verify_hashes

def main():
    dest=Path('results/modal_history_20261107/latency.json')
    if dest.exists():raise FileExistsError('Retain earlier diagnostic')
    run=Path('artifacts/runs/modal_history_20261107');p=json.loads((run/'provenance.json').read_text());cfg=p['config'];selected=json.loads((run/'selection.json').read_text());verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],run);rows=[]
    for name,case in selected['cases'].items():
        fold=cfg['families'][case['family']];records=load_allowed(cfg,fold['validation'],'validation');cal=LengthCalibration(**json.loads((run/name/'calibration.json').read_text()));data=concatenate([packets(s,case['axis'],cfg,cal)[0] for s in records.values()]);samples=[]
        for i in range(10):
            part={k:v[i:i+1].copy() for k,v in data.items() if k!='target'};start=time.perf_counter();modal(part,cfg,case['modal']);elapsed=1000*(time.perf_counter()-start)
            if i>=2:samples.append(elapsed)
        rows.append(dict(case=name,variant=case['modal'],single_packet_ms=samples,median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95))))
    report=dict(source_sha256=sha256(Path(__file__)),provenance_sha256=sha256(run/'provenance.json'),selection_sha256=sha256(run/'selection.json'),timing_environment=p['environment'],rows=rows,scope='Observed single-process wall time on first10 development windows per case, first2warmups discarded; includes all modal grid searches. Not a real-time scheduling or hardware guarantee.')
    dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='single_packet_ms'} for r in rows],indent=2))
if __name__=='__main__':main()
