"""Forecast with the frozen experimental combination from history only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from cranepilot.rollout import HORIZON_INDEX, HORIZONS_S
from scripts.combined_candidate import CombinedCandidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('artifacts/runs/combined_stress_20261107_replayfix'))
    p.add_argument('--fold', choices=['primary','short_transfer','long_transfer'], default='primary')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('Choose a fresh output file')
    provenance = json.loads((a.run/'provenance.json').read_text())
    if provenance['status'] != 'complete': raise ValueError('Completed evaluation required')
    for name,digest in provenance['guarded_hashes'].items():
        if name.endswith(('.py','.json')) and not name.startswith('data/'):
            if sha256(Path(name)) != digest: raise ValueError(f'Inference dependency changed: {name}')
    torch.set_num_threads(4)
    predictor = CombinedCandidate(provenance['config'], a.fold)
    with np.load(a.input, allow_pickle=False) as packet:
        means, seeds, gate = predictor.from_packet(packet)
    values = np.rad2deg(means['combined'][:,HORIZON_INDEX])
    spread = np.rad2deg(seeds['combined'][:,:,HORIZON_INDEX].std(0))
    frame = pd.DataFrame({f'angle_deg_{h:g}s':values[:,j] for j,h in enumerate(HORIZONS_S)})
    for j,h in enumerate(HORIZONS_S): frame[f'seed_std_deg_{h:g}s'] = spread[:,j]
    frame['network_weight'] = gate
    a.output.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(a.output,index=False)
    print('Saved experimental combination. Gate and seed spread are not calibrated probabilities or confidence intervals.')


if __name__ == '__main__': main()
