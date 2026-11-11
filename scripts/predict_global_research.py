"""Forecast using frozen offline global fits and a history-only NPZ packet."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.rollout import HORIZON_INDEX, HORIZONS_S
from scripts.run_global_research import verify
from scripts.predict_cranepilot import build_packet
from scripts.global_research_candidate import GlobalResearchCandidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/runs/global_research_20261107'))
    parser.add_argument('--fold',choices=['primary','short_transfer','long_transfer'],default='primary')
    parser.add_argument('--variant',default='common')
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve retained forecasts')
    prov=json.loads((args.run/'provenance.json').read_text())
    if prov['status']!='complete':raise ValueError('Completed frozen study required')
    verify(prov['guarded_hashes']);verify(prov['checkpoint_hashes'],args.run)
    verify({'selection.json':prov['selection_sha256']},args.run)
    selection=json.loads((args.run/'selection.json').read_text())
    name=selection['common']['model'] if args.variant=='common' else selection['folds'][args.fold]['model'] if args.variant=='selected' else args.variant
    torch.set_num_threads(4);predictor=GlobalResearchCandidate(prov['config'],args.fold,args.run/args.fold)
    with np.load(args.input,allow_pickle=False) as packet:
        raw=build_packet(packet['theta'],packet['x'],packet['reference'],packet['length_m'],packet['future_reference'],predictor.calibration)
    predictions,_=predictor.forecast(raw)
    if name not in predictions:raise ValueError('Unknown frozen model')
    y=np.rad2deg(predictions[name][:,HORIZON_INDEX])
    frame=pd.DataFrame({f'angle_deg_{h:g}s':y[:,i] for i,h in enumerate(HORIZONS_S)});frame['method']=name
    args.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output,index=False)
    print(f'Saved {len(frame)} history-only forecasts with {name}.')


if __name__=='__main__':main()
