"""History-only input for the frozen forced-adaptation study; restored workspace required."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from cranepilot.rollout import HORIZON_INDEX,HORIZONS_S
from scripts.predict_cranepilot import build_packet
from scripts.forced_adaptive_candidate import ForcedAdaptiveCandidate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('artifacts/runs/forced_adaptive_20261107'))
    p.add_argument('--fold',choices=['primary','short_transfer','long_transfer'],default='primary')
    p.add_argument('--variant',default='common',help='common, selected, or an explicit evaluated method')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose fresh output')
    prov=json.loads((a.run/'provenance.json').read_text())
    if prov['status']!='complete':raise ValueError('Completed study required')
    if sha256(a.run/'selection.json')!=prov['selection_sha256']:raise ValueError('Selection changed')
    for name,digest in prov['guarded_hashes'].items():
        if sha256(Path(name))!=digest:raise ValueError(f'Dependency changed: {name}')
    choice=json.loads((a.run/'selection.json').read_text())
    name=choice['common']['model'] if a.variant=='common' else choice['folds'][a.fold]['model'] if a.variant=='selected' else a.variant
    torch.set_num_threads(4);predictor=ForcedAdaptiveCandidate(prov['config'],a.fold)
    with np.load(a.input,allow_pickle=False) as packet:
        raw=build_packet(packet['theta'],packet['x'],packet['reference'],packet['length_m'],packet['future_reference'],predictor.calibration)
    pred,metadata,_=predictor.forecast(raw)
    if name not in pred:raise ValueError('Unknown method')
    y=np.rad2deg(pred[name][:,HORIZON_INDEX])
    frame=pd.DataFrame({f'angle_deg_{h:g}s':y[:,i] for i,h in enumerate(HORIZONS_S)});frame['method']=name
    if name in metadata:
        for key in ['frequency_ratio','damping_per_s','history_rms_rad','score_gap_rad2']:frame[key]=metadata[name][key]
    a.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(a.output,index=False)
    print(f'Saved {name}. New adaptive variants are experimental and did not pass shared validation selection.')


if __name__=='__main__':main()
