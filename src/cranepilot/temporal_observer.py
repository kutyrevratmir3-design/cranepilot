"""Causal dilated-history models for the fixed-information forecasting task.

Independent implementation of standard TCN building blocks (Bai et al., 2018).
This adaptation omits weight normalization and uses channel dropout. It is not
an exact reproduction of the paper's task-specific tuned benchmarks.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .innovation import InnovationForecaster


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=.1):
        super().__init__()
        self.left_pad = 2*dilation
        self.filters = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, dilation=dilation) for _ in range(2)])
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, sequence):
        residual = sequence
        for conv in self.filters:
            sequence = self.dropout(F.relu(conv(F.pad(sequence, (self.left_pad, 0)))))
        return F.relu(sequence+residual)


class TemporalHistoryEncoder(nn.Module):
    """41 newest-first samples become a chronological, 61-sample receptive field."""
    def __init__(self):
        super().__init__()
        self.project = nn.Conv1d(3, 16, 1)
        self.blocks = nn.Sequential(*[CausalBlock(16, d) for d in (1, 2, 4, 8)])
        self.combine = nn.Sequential(nn.Linear(18, 32), nn.Tanh())

    def forward(self, context):
        if context.ndim != 2 or context.shape[1] != 125:
            raise ValueError('Expected 41 angle/x/reference samples and two observer features')
        history = context[:, :123].reshape(-1, 3, 41).flip(-1)
        encoded = self.blocks(F.relu(self.project(history)))[:, :, -1]
        return self.combine(torch.cat([encoded, context[:, -2:]], dim=1))


class TemporalPhysicsForecaster(InnovationForecaster):
    """Same driven physics core; replace only the dense historical encoder."""
    def __init__(self):
        super().__init__(driven=True)
        self.encoder = TemporalHistoryEncoder()


class TCNResidualForecaster(nn.Module):
    """Causal convolutional decoder over the forced-physics forecast baseline."""
    def __init__(self):
        super().__init__()
        self.encoder = TemporalHistoryEncoder()
        self.project = nn.Conv1d(40, 16, 1)
        self.decoder = nn.Sequential(*[CausalBlock(16, d) for d in (1, 2, 4)])
        self.output = nn.Conv1d(16, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, batch, acceleration, dt=.05):
        count = acceleration.shape[1]
        phase = batch['omega'][:, None]*dt*torch.arange(
            1, count+1, device=acceleration.device, dtype=acceleration.dtype)
        features = torch.stack([
            batch['future_u']/2, acceleration/9.81, batch['physical_base']/.1,
            torch.sin(phase), torch.cos(phase),
            batch['omega'][:, None].expand_as(phase)/5,
            batch['initial_x'][:, None].expand_as(phase)/.5,
            batch['initial_velocity'][:, None].expand_as(phase)/.2], dim=1)
        history = self.encoder(batch['angular_context'])[:, :, None].expand(-1, -1, count)
        latent = self.decoder(F.relu(self.project(torch.cat([history, features], dim=1))))
        return batch['physical_base']+.1*self.output(latent)[:, 0], {}


def temporal_model(name):
    if name == 'dense_horizon':
        return InnovationForecaster(driven=True)
    if name in ('temporal_uniform', 'temporal_horizon'):
        return TemporalPhysicsForecaster()
    if name == 'tcn_reference':
        return TCNResidualForecaster()
    raise ValueError(f'Unknown temporal campaign model: {name}')
