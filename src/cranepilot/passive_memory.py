"""Phase-exact passive-memory surrogate with a discrete storage balance.

This module describes a fixed-length, small-angle angular surrogate driven by
prescribed trolley acceleration. It does not identify the actuator or certify
a crane controller. Batch inputs: state (batch, 2 + memory_dim), omega (batch,).
"""
import torch
from torch import nn


class PassiveMemoryCell(nn.Module):
    def __init__(self, memory_dim=4, context_dim=8, hidden=32):
        super().__init__()
        if memory_dim < 0:
            raise ValueError("memory_dim must be nonnegative")
        self.memory_dim = memory_dim
        self.coefficients = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1 + 2 * memory_dim))
        # Small initial memory exchange and low initial damping.
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)
        with torch.no_grad():
            self.coefficients[-1].bias[0] = -4.0
            # Nonzero exchange avoids a dead gradient when latent state starts at zero.
            self.coefficients[-1].bias[1:1 + memory_dim].fill_(0.02)

    def parameters_from_context(self, context):
        """Context must use observations through the forecast origin only."""
        raw = self.coefficients(context)
        m = self.memory_dim
        damping = 0.5 * torch.sigmoid(raw[:, 0])
        coupling = 2.0 * torch.tanh(raw[:, 1:1 + m])
        rates = 0.01 + 5.0 * torch.sigmoid(raw[:, 1 + m:])
        return damping, coupling, rates

    @staticmethod
    def closure_matrix(damping, coupling, rates):
        batch, m = coupling.shape
        dim = m + 2
        matrix = coupling.new_zeros(batch, dim, dim)
        matrix[:, 1, 1] = -damping
        matrix[:, 1, 2:] = coupling
        matrix[:, 2:, 1] = -coupling
        index = torch.arange(m, device=coupling.device)
        matrix[:, index + 2, index + 2] = -rates
        return matrix

    @staticmethod
    def dissipative_step(state, matrix, damping, rates, duration):
        """Cayley update; duration is dimensionless and nonnegative."""
        dim = state.shape[-1]
        eye = torch.eye(dim, dtype=state.dtype, device=state.device)
        h = duration[:, None, None]
        rhs = (eye + 0.5 * h * matrix) @ state[..., None]
        updated = torch.linalg.solve(eye - 0.5 * h * matrix, rhs)[..., 0]
        midpoint = 0.5 * (state + updated)
        dissipation = duration * (
            damping * midpoint[:, 1].square()
            + (rates * midpoint[:, 2:].square()).sum(-1))
        return updated, dissipation

    @staticmethod
    def forced_rotation(state, phase, force):
        """Exact solution of q' = p, p' = -q + force at constant force."""
        q, p = state[:, 0], state[:, 1]
        cosine, sine = torch.cos(phase), torch.sin(phase)
        # Stable evaluation at very small phase.
        one_minus_cosine = 2.0 * torch.sin(0.5 * phase).square()
        q_next = cosine * q + sine * p + one_minus_cosine * force
        p_next = -sine * q + cosine * p + sine * force
        updated = torch.cat((q_next[:, None], p_next[:, None], state[:, 2:]), -1)
        supplied_work = force * (q_next - q)
        return updated, supplied_work

    def step(self, state, omega, acceleration, dt, coefficients):
        """Coefficients are fixed from causal context for the forecast window."""
        damping, coupling, rates = coefficients
        phase = omega * dt
        matrix = self.closure_matrix(damping, coupling, rates)
        left, loss_left = self.dissipative_step(
            state, matrix, damping, rates, 0.5 * phase)
        middle, work = self.forced_rotation(left, phase, -acceleration / 9.81)
        final, loss_right = self.dissipative_step(
            middle, matrix, damping, rates, 0.5 * phase)
        energy_before = 0.5 * state.square().sum(-1)
        energy_after = 0.5 * final.square().sum(-1)
        balance_residual = energy_after - energy_before + loss_left + loss_right - work
        return final, {
            "energy": energy_after,
            "dissipation": loss_left + loss_right,
            "work": work,
            "balance_residual": balance_residual,
        }

    def rollout(self, initial_state, omega, future_acceleration, dt, context):
        """Future acceleration must come from a candidate action/actuator model.

        Supplying future measured acceleration is an oracle diagnostic only.
        No observation of future sway is consumed or assimilated here.
        """
        coefficients = self.parameters_from_context(context)
        state = initial_state
        outputs, balances = [], []
        for k in range(future_acceleration.shape[1]):
            state, diagnostics = self.step(
                state, omega, future_acceleration[:, k], dt, coefficients)
            outputs.append(state[:, 0])
            balances.append(diagnostics["balance_residual"])
        return torch.stack(outputs, 1), torch.stack(balances, 1)


def interval_squared_error(prediction, recorded_angle, bin_width):
    """Optional training loss, after verifying the encoder-bin interpretation."""
    outside = torch.relu((prediction - recorded_angle).abs() - 0.5 * bin_width)
    return outside.square().mean()


def property_checks():
    """Quick analytic/numerical checks; broader independent tests are in tests/."""
    torch.manual_seed(7)
    cell = PassiveMemoryCell().double()
    state = torch.randn(32, 6, dtype=torch.float64) * 0.1
    omega = torch.linspace(2.0, 8.0, 32, dtype=torch.float64)
    context = torch.randn(32, 8, dtype=torch.float64)
    coefficients = cell.parameters_from_context(context)
    before = 0.5 * state.square().sum(-1)

    after, diagnostics = cell.step(
        state, omega, torch.zeros_like(omega), 0.05, coefficients)
    assert torch.all(diagnostics["energy"] <= before + 1e-11)
    assert diagnostics["balance_residual"].abs().max() < 1e-11

    forced, diagnostics = cell.step(
        state, omega, torch.randn_like(omega), 0.05, coefficients)
    assert diagnostics["balance_residual"].abs().max() < 1e-11

    zero = (torch.zeros(32, dtype=torch.float64),
            torch.zeros(32, 4, dtype=torch.float64),
            torch.ones(32, 4, dtype=torch.float64))
    pure = state.clone()
    pure[:, 2:] = 0.0
    updated, _ = cell.step(
        pure, omega, torch.zeros_like(omega), 0.05, zero)
    phase = omega * 0.05
    expected = torch.cos(phase) * pure[:, 0] + torch.sin(phase) * pure[:, 1]
    assert torch.allclose(updated[:, 0], expected, atol=1e-12, rtol=1e-12)

    loss = forced.square().mean()
    loss.backward()
    gradients = [p.grad for p in cell.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)

    # Further required tests: independent scalar reference, finite-difference
    # parameter gradients, dt convergence, future-observation mutation,
    # batch independence, and actuator coupling on independently generated data.

