"""Small zero-initialized corrections to a permanently frozen soft forecaster.

This is an offline residual-learning experiment inspired by hybrid dynamics
models such as KNODE (https://proceedings.mlr.press/v211/chee23a.html), not a
reproduction of that controller or its closed-loop stability theorem. The
optional correction filter alone has stable homogeneous dynamics. Its bounded
learned forcing is independent of the residual state; the base neural forecast
is not covered by that stability statement.

Inputs retain the established complete known-command plan. No future measured
angle/position or target is read. Caching is an explicit training optimization;
ordinary forward remains differentiable with respect to differentiable inputs.
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cranepilot.broad_strategies import mirrored
from cranepilot.rollout import HORIZON_INDEX
from scripts.finetune_strategies import phase_space_terms


def oscillator_transition(omega, damping_ratio=.05, step=.05):
    """Exact ZOH flow of q'=omega*p, p'=omega*(-q-2*zeta*p+force)."""
    if omega.ndim != 1 or not torch.isfinite(omega).all() or torch.any(omega <= 0):
        raise ValueError('Finite positive per-window omega required')
    if not np.isfinite(damping_ratio) or damping_ratio <= 0 or not np.isfinite(step) or step <= 0:
        raise ValueError('Positive finite damping and step required')
    generator = omega.new_tensor([[0., 1.], [-1., -2.*damping_ratio]])
    transition = torch.matrix_exp(omega[:, None, None]*step*generator)
    # Constant force has equilibrium (q,p)=(force,0).
    port = -transition[:, :, 0].clone()
    port[:, 0] = port[:, 0]+1.
    return transition, port


class ResidualAdapter(nn.Module):
    def __init__(self, base, hidden=8, correction='force', symmetry_mix=.5):
        super().__init__()
        if hidden not in (8, 16) or correction not in ('force', 'direct'):
            raise ValueError('Preregistered hidden widths8/16 and force/direct correction required')
        if symmetry_mix != .5:
            raise ValueError('This experiment fixes soft symmetry at .5')
        self.base = base
        self.hidden = int(hidden)
        self.correction = correction
        self.symmetry_mix = float(symmetry_mix)
        self.damping_ratio = .05
        self.bound = .05 if correction == 'force' else .02
        reference = next(base.parameters())
        self.adapter = nn.Sequential(nn.Linear(38, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.adapter.to(device=reference.device, dtype=reference.dtype)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self._freeze_base()

    def _freeze_base(self):
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def train(self, mode=True):
        super().train(mode)
        self._freeze_base()
        return self

    def _path_features(self, batch, acceleration):
        prediction, _ = self.base(batch, acceleration)
        n, steps = prediction.shape
        if steps != 30 or acceleration.shape != prediction.shape:
            raise ValueError('Adapter requires30-step matching acceleration and forecast')
        context = self.base.encoder(batch['angular_context'])
        if context.shape != (n, 32):
            raise ValueError('Frozen encoder must supply32 history features')
        previous_q = torch.cat([batch['initial_q'][:, None], prediction[:, :-1]], dim=1)
        previous_p = torch.cat([batch['initial_p'][:, None],
            torch.diff(previous_q, dim=1)/(batch['omega'][:, None]*.05)], dim=1)
        clock = torch.arange(30, dtype=prediction.dtype, device=prediction.device)/30.
        dynamic = torch.stack([previous_q/.1, previous_p/.1, batch['future_u']/2.,
                               acceleration, batch['omega'][:, None].expand(-1, 30)/5.,
                               clock[None].expand(n, -1)], dim=-1)
        return prediction, torch.cat([context[:, None, :].expand(-1, 30, -1), dynamic], dim=-1)

    def prepare(self, batch, acceleration):
        """Compute frozen features; callers may use no_grad to cache for training."""
        positive, features = self._path_features(batch, acceleration)
        _, negative_features = self._path_features(mirrored(batch), -acceleration)
        result = {'base_prediction': positive, 'features': features,
                  'negative_features': negative_features, 'omega': batch['omega']}
        if self.correction == 'force':
            transition, port = oscillator_transition(batch['omega'], self.damping_ratio)
            result.update(transition=transition, port=port)
        return result

    def _filter(self, force, prepared):
        if self.correction == 'direct':
            return force
        state = force.new_zeros(len(force), 2)
        ys = []
        for k in range(force.shape[1]):
            state = ((prepared['transition'] @ state[..., None])[..., 0]
                     + prepared['port']*force[:, k, None])
            ys.append(state[:, 0])
        return torch.stack(ys, dim=1)

    def forward_prepared(self, prepared):
        positive = self.bound*torch.tanh(self.adapter(prepared['features'])[..., 0])
        negative = self.bound*torch.tanh(self.adapter(prepared['negative_features'])[..., 0])
        positive_correction = self._filter(positive, prepared)
        negative_correction = self._filter(negative, prepared)
        correction = ((1.-.5*self.symmetry_mix)*positive_correction
                      -.5*self.symmetry_mix*negative_correction)
        prediction = prepared['base_prediction']+correction
        return prediction, {'adapter_correction': correction,
                            'base_prediction': prepared['base_prediction'],
                            'initial_memory': prediction.new_zeros(len(prediction), 1),
                            'innovation_force': torch.cat([positive, negative], dim=0)}

    def forward(self, batch, acceleration):
        return self.forward_prepared(self.prepare(batch, acceleration))


def make_adapter(base, choice):
    return ResidualAdapter(base, hidden=int(choice.get('hidden', 8)),
                           correction=choice.get('correction', 'force'))


@torch.no_grad()
def cache_frozen_features(model, batch, chunk=512):
    """Cache detached TRAIN/validation features separately; never cache targets."""
    if chunk < 1 or not len(batch['omega']):
        raise ValueError('Nonempty batch and positive chunk required')
    parts = []
    for start in range(0, len(batch['omega']), chunk):
        local = {key: value[start:start+chunk] for key, value in batch.items()}
        parts.append(model.prepare(local, local['acceleration']))
    return {key: torch.cat([part[key] for part in parts]).detach() for key in parts[0]}


@torch.no_grad()
def _validation(model, prepared, target, active, chunk=512):
    predictions = []
    for start in range(0, len(target), chunk):
        local = {key: value[start:start+chunk] for key, value in prepared.items()}
        predictions.append(model.forward_prepared(local)[0])
    error = torch.rad2deg(torch.cat(predictions)[:, HORIZON_INDEX]-target[:, HORIZON_INDEX]).abs()
    if not active.any() or active.all():
        raise ValueError('Both validation regimes required')
    return {'all': float(error.mean()), 'active': float(error[active].mean()),
            'quiet': float(error[~active].mean())}


def _criterion(values, starting):
    return .5*(values['all']/max(starting['all'], 1e-12)
               +values['active']/max(starting['active'], 1e-12))


def fit_adapter(model, trainbatch, target, active, scenario, validationbatch,
                valtarget, valactive, config, choice, seed):
    """Return(state, logs, metadata), restoring selected model, epoch-1 eligible.

    Only adapter parameters optimize. The frozen base's old memory/force
    regularizers are constants and omitted. Add .01 mean squared bounded adapter
    head output. Optional phase_space adds .25 rate + .25 terminal surrogates.
    Batching and validation checkpoint criterion match the prior fine-tune study.
    ``active`` and ``scenario`` are checked for alignment, not used as gradients.
    """
    if not isinstance(model, ResidualAdapter):
        raise TypeError('A ResidualAdapter is required')
    model._freeze_base()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    reference = next(model.parameters())
    dtype, device = reference.dtype, reference.device
    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=device).detach()
    target, valtarget = tensor(target), tensor(valtarget)
    active = torch.as_tensor(active, dtype=torch.bool, device=device)
    valactive = torch.as_tensor(valactive, dtype=torch.bool, device=device)
    trainbatch = {key: tensor(value) for key, value in trainbatch.items()}
    validationbatch = {key: tensor(value) for key, value in validationbatch.items()}
    n = len(target)
    if n < 1 or target.shape != (n, 30) or active.shape != (n,) or np.asarray(scenario).shape != (n,):
        raise ValueError('Aligned nonempty N by30 training target and labels required')
    if valtarget.shape != (len(valactive), 30) or not len(valtarget):
        raise ValueError('Aligned nonempty validation targets required')
    if any(len(value) != n for value in trainbatch.values()) or any(len(value) != len(valtarget) for value in validationbatch.values()):
        raise ValueError('Input/target row mismatch')
    if not torch.isfinite(target).all() or not torch.isfinite(valtarget).all():
        raise ValueError('Finite targets required')
    epochs, patience, batch_size = [int(config[key]) for key in ('epochs', 'patience', 'batch_size')]
    beta = float(config.get('smooth_l1_beta', .1))
    if epochs < 0 or patience < 1 or batch_size < 1 or not np.isfinite(beta) or beta <= 0:
        raise ValueError('Invalid training schedule')
    phase = bool(choice.get('phase_space', False))
    name = choice.get('name', f'{model.correction}{model.hidden}'+('_phase' if phase else ''))
    parameters = list(model.adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(config['learning_rate']),
                                  weight_decay=float(config['weight_decay']))
    frozen = {key: value.detach().clone() for key, value in model.base.state_dict().items()}
    start = time.perf_counter()
    prepared = cache_frozen_features(model, trainbatch)
    validation = cache_frozen_features(model, validationbatch)
    cache_seconds = time.perf_counter()-start
    starting = _validation(model, validation, valtarget, valactive)
    best_score = _criterion(starting, starting)
    best_values = starting.copy()
    best_epoch, stale, executed, optimizer_steps = -1, 0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    logs = [{'epoch': -1, 'training_loss': None, 'selection_score': best_score,
             'best_score': best_score, 'checkpoint_improved': True, 'elapsed_s': cache_seconds,
             **{f'validation_{key}': value for key, value in starting.items()}}]
    for epoch in range(epochs):
        model.train()
        losses, sizes = [], []
        for ids_np in np.array_split(rng.permutation(n), int(np.ceil(n/batch_size))):
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
            local = {key: value[ids] for key, value in prepared.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction, diag = model.forward_prepared(local)
            loss = F.smooth_l1_loss((prediction-target[ids])/.1,
                                    torch.zeros_like(prediction), beta=beta)
            loss = loss+.01*diag['innovation_force'].square().mean()
            if phase:
                rate, terminal = phase_space_terms(prediction, target[ids], local['omega'], beta)
                loss = loss+.25*rate+.25*terminal
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite adapter loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Nonfinite adapter gradient')
            torch.nn.utils.clip_grad_norm_(parameters, float(config.get('gradient_clip',2.)))
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach())); sizes.append(len(ids_np))
        model.eval()
        values = _validation(model, validation, valtarget, valactive)
        score = _criterion(values, starting)
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite adapter validation score')
        improved = score < best_score-1e-8
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            best_values, best_state = values.copy(), copy.deepcopy(model.state_dict())
        else:
            stale += 1
        logs.append({'epoch': epoch, 'training_loss': float(np.average(losses, weights=sizes)),
                     'selection_score': score, 'best_score': best_score,
                     'checkpoint_improved': improved, 'elapsed_s': time.perf_counter()-start,
                     **{f'validation_{key}': value for key, value in values.items()}})
        executed += 1
        if epoch % 10 == 0:
            print(f'  {name} seed={seed} epoch={epoch} selection={score:.6f} best={best_score:.6f}', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    for key, value in model.base.state_dict().items():
        if not torch.equal(value, frozen[key]):
            raise AssertionError(f'Frozen base changed: {key}')
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    metadata = {'strategy': name, 'seed': int(seed), 'best_epoch': best_epoch,
                'selected_epoch': best_epoch, 'epochs_executed': executed,
                'optimizer_steps': optimizer_steps, 'backward_passes': optimizer_steps,
                'optimizer_closure_evaluations': optimizer_steps,
                'fit_seconds': time.perf_counter()-start, 'cache_seconds': cache_seconds,
                'selection_score': best_score, 'validation': best_values,
                'starting_validation': starting,
                'trainable_parameter_names': [name for name, p in model.named_parameters() if p.requires_grad],
                'trainable_parameter_count': sum(p.numel() for p in parameters),
                'total_parameter_count': sum(p.numel() for p in model.parameters()),
                'epoch_minus_one_eligible': True, 'validation_gradients': False,
                'base_frozen_exact': True, 'base_regularizers_omitted_as_constant': True,
                'adapter_head_regularizer': .01, 'phase_space_weight': .25 if phase else 0.,
                'hidden': model.hidden, 'correction': model.correction,
                'bounded_head_radians': model.bound, 'damping_ratio': model.damping_ratio,
                'symmetry_mix': model.symmetry_mix, 'optimizer': 'AdamW'}
    return state, logs, metadata
