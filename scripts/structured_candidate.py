"""New structured models plus exact controls retained from the preceding stage."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from cranepilot.broad_strategies import make_model
from scripts.finetune_candidate import FinetuneCandidate
from scripts.run_broad_search import inputs, predict

CONTROLS = ['incumbent', 'soft_symmetry', 'physical', 'previous_local']


def make_structured_model(config, fold, seed, choice):
    from scripts.residual_adapter import make_adapter
    from scripts.stable_latent import make_stable_latent
    base = None
    if choice['family'] == 'adapter' or choice.get('mode') == 'residual':
        base = make_model({'symmetry_mix': .5})
        source = Path(config['broad_run']) / fold / f'soft_symmetry_seed{seed}.pt'
        base.load_state_dict(torch.load(source, map_location='cpu', weights_only=True))
    if choice['family'] == 'adapter':
        return make_adapter(base, choice)
    if choice['family'] == 'latent':
        return make_stable_latent(choice, base=base)
    raise ValueError('Unknown preregistered model family')


def fit_structured(model, *args):
    from scripts.residual_adapter import ResidualAdapter, fit_adapter
    from scripts.stable_latent import fit_stable
    return (fit_adapter if isinstance(model, ResidualAdapter) else fit_stable)(model, *args)


class StructuredCandidate:
    def __init__(self, config, fold, directory=None):
        self.config = config
        parent = Path(config['parent_run'])
        parent_cfg = copy.deepcopy(json.loads((parent/'provenance.json').read_text())['config'])
        self.parent_selected = json.loads((parent/'selection.json').read_text())['folds'][fold]['model']
        # Only the retained parent winner needs its new fine-tune ensemble loaded.
        parent_cfg['variants'] = {n: c for n, c in parent_cfg['variants'].items() if n == self.parent_selected}
        self.parent = FinetuneCandidate(parent_cfg, fold, parent/fold)
        self.actor = self.parent.actor
        self.calibration = self.parent.calibration
        self.ensembles = {}
        if directory is not None:
            for name, choice in config['variants'].items():
                models = []
                for seed in config['seeds']:
                    model = make_structured_model(config, fold, seed, choice)
                    model.load_state_dict(torch.load(Path(directory)/f'{name}_seed{seed}.pt', weights_only=True))
                    models.append(model.eval())
                self.ensembles[name] = models

    def forecast(self, raw, prescribed_acceleration=None):
        prior, _, acceleration = self.parent.forecast(raw, prescribed_acceleration)
        predictions = {name: prior[name] for name in CONTROLS if name != 'previous_local'}
        predictions['previous_local'] = prior[self.parent_selected]
        batch = inputs(raw)
        batch['acceleration'] = torch.as_tensor(acceleration, dtype=torch.float32)
        seeds = {name: np.stack([predict(m, batch) for m in models]) for name, models in self.ensembles.items()}
        predictions.update({name: value.mean(0) for name, value in seeds.items()})
        if not all(y.shape == (len(raw['omega']), 30) and np.isfinite(y).all() for y in predictions.values()):
            raise ValueError('Invalid structured forecast')
        return predictions, seeds, acceleration
