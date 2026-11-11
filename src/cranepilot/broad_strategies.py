"""Matched continuation strategies for the incumbent; differentiable compiled flow."""
import numpy as np
import torch
from torch.nn import functional as F
from .innovation import InnovationForecaster
from .data import QUANTIZATION_RAD


def mirrored(batch):
    """Reverse signed angular/command/history quantities, keeping frequency and RMS.

    This is an angular-model projection given acceleration. It does not assert
    that the separately learned real actuator obeys the same reflection.
    """
    signed = {'angular_context', 'initial_q', 'initial_p', 'future_u', 'acceleration'}
    return {key: -value if key in signed else value for key, value in batch.items()}


class StrategyForecaster(InnovationForecaster):
    def __init__(self, angle_bound=False, symmetry_mix=0.0):
        super().__init__(driven=True)
        self.angle_bound = angle_bound
        self.symmetry_mix = symmetry_mix

    def flow(self, batch, acceleration):
        context = self.encoder(batch['angular_context'])
        correction = torch.tanh(self.initial(context))
        adjustment = torch.tanh(self.observer(context))
        bound = (batch['observer_residual_rms'].clamp(max=.02)
                 if self.angle_bound else .02)
        delta_q = bound * adjustment[:, 0]
        q = batch['initial_q'] + delta_q
        omega = batch['omega'] * (1 + .15 * adjustment[:, 1])
        p = batch['initial_p'] * batch['omega'] / omega + .2 * correction[:, 0]
        budget = .5 * torch.sqrt(q.square() + p.square() + QUANTIZATION_RAD**2)
        memory = budget[:, None] * correction[:, 1:] / np.sqrt(4)
        state = torch.cat([q[:, None], p[:, None], memory], -1)
        damping, coupling, rates = self.cell.parameters_from_context(context)
        C = self.cell.closure_matrix(damping, coupling, rates)
        n, dim = state.shape
        eye = torch.eye(dim, dtype=state.dtype, device=state.device).expand(n, -1, -1)
        phase = omega * .05
        half = .25 * phase[:, None, None] * C
        D = torch.linalg.solve(eye - half, eye + half)
        cosine, sine = torch.cos(phase), torch.sin(phase)
        R = eye.clone()
        R[:, 0, 0] = cosine; R[:, 1, 1] = cosine
        R[:, 0, 1] = sine; R[:, 1, 0] = -sine
        transition = D @ R @ D
        port = state.new_zeros(n, dim)
        port[:, 0] = 2 * torch.sin(.5 * phase)**2
        port[:, 1] = sine
        port = (D @ port[..., None])[..., 0]
        hidden = torch.tanh(self.hidden_initial(context))
        ys, forces = [], []
        for k in range(acceleration.shape[1]):
            inputs = torch.stack([state[:, 0] / .1, state[:, 1] / .1,
                                  batch['future_u'][:, k] / 2, acceleration[:, k], omega / 5], -1)
            hidden = self.innovation(inputs, hidden)
            force = .1 * torch.tanh(self.force_output(hidden)[:, 0])
            state = ((transition @ state[..., None])[..., 0]
                     + port * (-acceleration[:, k] / 9.81 + force)[:, None])
            ys.append(state[:, 0]); forces.append(force)
        return torch.stack(ys, 1), {'initial_memory': memory,
                                  'innovation_force': torch.stack(forces, 1), 'angle_shift': delta_q}

    def forward(self, batch, acceleration):
        positive, diag = self.flow(batch, acceleration)
        if self.symmetry_mix:
            negative, other = self.flow(mirrored(batch), -acceleration)
            projected = .5 * (positive - negative)
            positive = (1 - self.symmetry_mix) * positive + self.symmetry_mix * projected
            # Penalize both internal paths, rather than cancelling their regularizers.
            for key in ('initial_memory', 'innovation_force'):
                diag[key] = torch.cat([diag[key], other[key]], 0)
        return positive, diag


def fit_loss(prediction, target, active, balanced=False, beta=.1):
    errors = F.smooth_l1_loss((prediction - target) / .1,
                             torch.zeros_like(prediction), beta=beta, reduction='none').mean(1)
    if not balanced:
        return errors.mean()
    groups = [errors[mask].mean() for mask in (active, ~active) if mask.any()]
    return torch.stack(groups).mean()


def make_model(choice):
    return StrategyForecaster(choice.get('angle_bound', False), choice.get('symmetry_mix', 0.0))
