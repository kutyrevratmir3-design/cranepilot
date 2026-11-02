"""Causal observer refinement and explicitly driven latent innovation.

The passive core retains its storage identity. Learned forcing is an additional
energy port, not evidence of physical passivity or identified external force.
"""
import numpy as np
import torch
from torch import nn
from .rollout import PassiveForecaster, STEP
from .data import QUANTIZATION_RAD


class InnovationForecaster(PassiveForecaster):
    def __init__(self, driven=False):
        super().__init__(memory_dim=4)
        self.driven = driven
        self.observer = nn.Linear(32, 2)
        nn.init.zeros_(self.observer.weight)
        nn.init.zeros_(self.observer.bias)
        if driven:
            self.hidden_initial = nn.Linear(32, 12)
            self.innovation = nn.GRUCell(5, 12)
            self.force_output = nn.Linear(12, 1)
            nn.init.zeros_(self.force_output.weight)
            nn.init.zeros_(self.force_output.bias)

    def forward(self, batch, acceleration):
        context = self.encoder(batch['angular_context'])
        correction = torch.tanh(self.initial(context))
        adjustment = torch.tanh(self.observer(context))
        q = batch['initial_q'] + .02*adjustment[:, 0]
        # A bounded local effective-frequency correction; not identified geometry.
        omega = batch['omega']*(1. + .15*adjustment[:, 1])
        p = batch['initial_p']*batch['omega']/omega + .2*correction[:, 0]
        budget = .5*torch.sqrt(q.square()+p.square()+QUANTIZATION_RAD**2)
        memory = budget[:, None]*correction[:, 1:]/np.sqrt(4)
        state = torch.cat([q[:, None], p[:, None], memory], -1)
        coefficients = self.cell.parameters_from_context(context)
        if self.driven:
            hidden = torch.tanh(self.hidden_initial(context))
        outputs, balances, forces = [], [], []
        for k in range(acceleration.shape[1]):
            force = torch.zeros_like(q)
            if self.driven:
                inputs = torch.stack([state[:, 0]/.1, state[:, 1]/.1,
                                      batch['future_u'][:, k]/2.,
                                      acceleration[:, k], omega/5.], -1)
                hidden = self.innovation(inputs, hidden)
                force = .1*torch.tanh(self.force_output(hidden)[:, 0])
            # Core force is -a/g: subtracting g*innovation adds +innovation.
            state, diag = self.cell.step(state, omega,
                acceleration[:, k]-9.81*force, STEP, coefficients)
            outputs.append(state[:, 0]); forces.append(force)
            balances.append(diag['balance_residual'])
        return torch.stack(outputs, 1), {
            'balance':torch.stack(balances, 1), 'initial_memory':memory,
            'innovation_force':torch.stack(forces, 1),
            'frequency_ratio':omega/batch['omega']}
