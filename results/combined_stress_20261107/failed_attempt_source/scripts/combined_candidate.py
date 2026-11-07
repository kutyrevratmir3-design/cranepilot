"""Fixed soft symmetry + discrepancy gate; no fitting and no selection."""
import json
from pathlib import Path
import numpy as np
import torch
from cranepilot.broad_strategies import make_model
from cranepilot.data import sha256
from cranepilot.forced_observer import refine_windows, analytic_rollout
from cranepilot.physics import LengthCalibration
from scripts.run_temporal import load_actor
from scripts.run_broad_search import inputs, predict
from scripts.predict_cranepilot import build_packet
from scripts.reliability_blend import blend


class CombinedCandidate:
    def __init__(self, config, fold):
        self.config, self.fold = config, fold
        source = Path(config['source_run']); broad = Path(config['broad_run'])
        old = json.loads((source / 'provenance.json').read_text())
        previous = json.loads((broad / 'provenance.json').read_text())
        for provenance, base in [(old, source), (previous, broad)]:
            if provenance['status'] != 'complete': raise ValueError('Completed parent run required')
            for key, prefix in [('guarded_hashes', Path('.')), ('checkpoint_hashes', base)]:
                for name, digest in provenance[key].items():
                    if sha256(prefix / name.replace('\\', '/')) != digest:
                        raise ValueError(f'Parent dependency changed: {name}')
        info = old['inputs'][fold]
        self.calibration = LengthCalibration(**info['calibration'])
        self.actor = load_actor(dict(info, actor=info['actor'].replace('\\', '/')))
        self.models = {}
        for name, choice in [('incumbent', {}), ('soft_symmetry', previous['config']['variants']['soft_symmetry'])]:
            models = []
            for i, seed in enumerate(config['seeds']):
                path = Path(info['weights'][i].replace('\\', '/')) if name == 'incumbent' else broad / fold / f'soft_symmetry_seed{seed}.pt'
                model = make_model(choice); model.load_state_dict(torch.load(path, weights_only=True)); model.eval(); models.append(model)
            self.models[name] = models

    def from_packet(self, packet, autonomous=False):
        raw = build_packet(packet['theta'], packet['x'], packet['reference'], packet['length_m'], packet['future_reference'], self.calibration)
        return self.from_raw(raw, autonomous)

    def from_raw(self, raw, autonomous=False):
        # Unknown targets, scenario identifiers and true simulation parameters are unused.
        batch = inputs(raw, None if autonomous else self.actor)
        refined = refine_windows(raw)
        physical = analytic_rollout(refined['initial_q'], refined['initial_p'], raw['omega'], batch['acceleration'].numpy())
        rms = np.sqrt(np.mean(refined['forced_residual']**2, axis=1))
        seeds = {name: np.stack([predict(m, batch) for m in models]) for name, models in self.models.items()}
        for parent, name in [('incumbent', 'incumbent_gate'), ('soft_symmetry', 'combined')]:
            seeds[name] = np.stack([blend(y, physical, rms, self.config['tau_bins'])[0] for y in seeds[parent]])
        seeds['fixed_half'] = .5 * seeds['soft_symmetry'] + .5 * physical[None]
        means = {name: y.mean(0) for name, y in seeds.items()}
        means['physical'] = physical
        weight = blend(seeds['incumbent'][0], physical, rms, self.config['tau_bins'])[1]
        return means, seeds, weight
