"""Use one frozen fine-tune ensemble from a history-only input packet."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.broad_strategies import make_model
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import HORIZON_INDEX,HORIZONS_S
from scripts.run_global_research import verify
from scripts.run_temporal import load_actor
from scripts.run_broad_search import inputs,predict
from scripts.predict_cranepilot import build_packet
from scripts.finetune_candidate import FinetuneCandidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/runs/finetune_campaign_20261107'))
    parser.add_argument('--fold',choices=['primary','short_transfer','long_transfer'],default='primary')
    parser.add_argument('--variant',default='selected')
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous output')
    prov=json.loads((args.run/'provenance.json').read_text())
    if prov['status']!='complete':raise ValueError('Completed frozen study required')
    cfg=prov['config'];verify(prov['guarded_hashes']);verify(prov['checkpoint_hashes'],args.run)
    parent=Path(cfg['parent_run']);parent_prov=json.loads((parent/'provenance.json').read_text())
    verify(parent_prov['checkpoint_hashes'],parent)
    verify({'selection.json':prov['selection_sha256']},args.run)
    selection=json.loads((args.run/'selection.json').read_text())
    name=selection['folds'][args.fold]['model'] if args.variant=='selected' else selection['common']['model'] if args.variant=='common' else args.variant
    info=json.loads((Path(cfg['source_run'])/'provenance.json').read_text())['inputs'][args.fold]
    calibration=LengthCalibration(**info['calibration']);torch.set_num_threads(4)
    with np.load(args.input,allow_pickle=False) as packet:
        raw=build_packet(packet['theta'],packet['x'],packet['reference'],packet['length_m'],packet['future_reference'],calibration)
    if name in cfg['variants']:
        actor=load_actor(dict(info,actor=info['actor'].replace('\\','/')));batch=inputs(raw,actor)
        ys=[]
        for seed in cfg['seeds']:
            model=make_model({'symmetry_mix':.5});model.load_state_dict(torch.load(args.run/args.fold/f'{name}_seed{seed}.pt',weights_only=True))
            ys.append(predict(model,batch))
        prediction=np.mean(ys,axis=0)
    else:
        predictions,_,_=FinetuneCandidate(cfg,args.fold).forecast(raw)
        if name not in predictions:raise ValueError('Unknown frozen method')
        prediction=predictions[name]
    y=np.rad2deg(prediction[:,HORIZON_INDEX])
    frame=pd.DataFrame({f'angle_deg_{h:g}s':y[:,i] for i,h in enumerate(HORIZONS_S)});frame['method']=name
    args.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output,index=False)
    print(f'Saved {len(frame)} forecasts using {name}.')


if __name__=='__main__':main()
