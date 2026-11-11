"""Offline identified physics and direct delay lifts; fixed history-only deployment."""
import json
from pathlib import Path
import numpy as np
import torch
from cranepilot.rollout import tensors
from cranepilot.forced_observer import refine_windows, analytic_rollout
from scripts.combined_candidate import CombinedCandidate
from scripts.run_broad_search import inputs, predict
from scripts.reliability_blend import blend
from scripts.global_weak_physics import forecast_global
from scripts.global_delay_lift import load_model, predict_residual

CONTROLS = ['incumbent', 'soft_symmetry', 'combined', 'physical']


def control_forecasts(predictor, raw, acceleration=None):
    if acceleration is None:
        with torch.no_grad(): acceleration = predictor.actor(tensors(raw))[1].numpy()
    acceleration = np.asarray(acceleration, dtype=float)
    if acceleration.shape != (len(raw['omega']), 30) or not np.isfinite(acceleration).all():
        raise ValueError('Finite N by 30 acceleration required')
    batch = inputs(raw)
    batch['acceleration'] = torch.as_tensor(acceleration, dtype=torch.float32)
    predictions = {name: np.stack([predict(m, batch) for m in models]).mean(0)
                   for name, models in predictor.models.items()}
    refined = refine_windows(raw)
    predictions['physical'] = analytic_rollout(refined['initial_q'], refined['initial_p'], raw['omega'], acceleration)
    rms = np.sqrt(np.mean(refined['forced_residual']**2, axis=1))
    predictions['combined'] = blend(predictions['soft_symmetry'], predictions['physical'], rms, predictor.config['tau_bins'])[0]
    return predictions, acceleration


class GlobalResearchCandidate(CombinedCandidate):
    def __init__(self, config, fold, directory):
        super().__init__(config, fold)
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/'models.json').read_text())
        self.fitted = {}
        for name, spec in self.manifest.items():
            path = self.directory/spec['file']
            self.fitted[name] = json.loads(path.read_text()) if spec['family'] == 'weak' else load_model(path)

    def forecast(self, raw, prescribed_acceleration=None):
        predictions, acceleration = control_forecasts(self, raw, prescribed_acceleration)
        theta = raw['angular_context'][:, :41]*.1
        x = raw['actuator_context'][:, :41]*.1
        for name, spec in self.manifest.items():
            model = self.fitted[name]
            if spec['family'] == 'weak':
                predictions[name] = forecast_global(model, theta, x, raw['omega'], acceleration,
                                                    velocity=raw['initial_velocity'])
            else:
                predictions[name] = predictions['physical'] + predict_residual(model, raw, acceleration)
        if not all(y.shape == (len(theta), 30) and np.isfinite(y).all() for y in predictions.values()):
            raise ValueError('Invalid global forecast')
        return predictions, acceleration
