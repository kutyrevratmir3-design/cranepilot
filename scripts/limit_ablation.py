"""Inference interventions on a frozen InnovationForecaster, without weight edits."""
from contextlib import ExitStack
from unittest.mock import patch
import torch
from cranepilot.fast_inference import fast_prediction

COMPONENTS = frozenset({'angle', 'momentum', 'frequency', 'memory', 'closure', 'force'})


@torch.no_grad()
def predict_disabled(model, batch, acceleration, disabled=()):
    disabled = frozenset(disabled)
    if not disabled <= COMPONENTS:
        raise ValueError(f'Unknown components: {disabled - COMPONENTS}')
    if model.training or model.measurement_anchor:
        raise ValueError('Audit requires the unanchored incumbent in eval mode')

    def mask_observer(module, inputs, output):
        result = output.clone()
        if 'angle' in disabled:
            result[:, 0] = 0
        if 'frequency' in disabled:
            result[:, 1] = 0
        return result

    def mask_initial(module, inputs, output):
        result = output.clone()
        if 'momentum' in disabled:
            result[:, 0] = 0
        if 'memory' in disabled:
            result[:, 1:] = 0
        return result

    original_coefficients = model.cell.parameters_from_context

    def no_closure(context):
        return tuple(torch.zeros_like(x) for x in original_coefficients(context))

    with ExitStack() as stack:
        for layer, hook in [(model.observer, mask_observer), (model.initial, mask_initial)]:
            handle = layer.register_forward_hook(hook)
            stack.callback(handle.remove)
        if 'closure' in disabled:
            stack.enter_context(patch.object(model.cell, 'parameters_from_context', no_closure))
        if 'force' in disabled:
            stack.enter_context(patch.object(model, 'driven', False))
        return fast_prediction(model, batch, acceleration)
