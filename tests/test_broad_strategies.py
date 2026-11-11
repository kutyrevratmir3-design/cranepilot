import copy
import unittest
import torch
from cranepilot.innovation import InnovationForecaster
from cranepilot.broad_strategies import StrategyForecaster, mirrored, fit_loss


class BroadStrategyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7319)
        self.model = StrategyForecaster().double()
        with torch.no_grad():
            for layer in [self.model.initial, self.model.observer, self.model.force_output]:
                layer.weight.normal_(0, .08); layer.bias.normal_(0, .1)
        self.batch = {'angular_context': torch.randn(4, 125, dtype=torch.double) * .2,
                      'initial_q': torch.randn(4, dtype=torch.double) * .1,
                      'initial_p': torch.randn(4, dtype=torch.double) * .1,
                      'omega': torch.linspace(3, 9, 4, dtype=torch.double),
                      'future_u': torch.randn(4, 30, dtype=torch.double),
                      'observer_residual_rms': torch.tensor([0., .0001, .01, .1], dtype=torch.double)}
        self.a = torch.randn(4, 30, dtype=torch.double) * .3

    def test_differentiable_compilation_matches_reference_outputs_and_gradients(self):
        old = InnovationForecaster(True).double()
        old.load_state_dict(self.model.state_dict())
        y, _ = self.model(self.batch, self.a)
        reference, _ = old(self.batch, self.a)
        torch.testing.assert_close(y, reference, rtol=1e-10, atol=1e-12)
        y.square().mean().backward(); reference.square().mean().backward()
        for (key, p), (_, q) in zip(self.model.named_parameters(), old.named_parameters()):
            self.assertIsNotNone(p.grad, key)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-11)

    def test_bound_vanishes_at_consistent_history_and_respects_rms(self):
        self.model.angle_bound = True
        _, diag = self.model(self.batch, self.a)
        self.assertEqual(float(diag['angle_shift'][0].detach()), 0)
        self.assertTrue(torch.all(diag['angle_shift'].abs() <= self.batch['observer_residual_rms'].clamp(max=.02)))

    def test_hard_projection_is_odd_and_has_zero_equilibrium(self):
        self.model.symmetry_mix = 1.
        y, _ = self.model(self.batch, self.a)
        z, _ = self.model(mirrored(self.batch), -self.a)
        torch.testing.assert_close(y, -z, rtol=0, atol=0)
        rest = {k: v if k == 'omega' else torch.zeros_like(v) for k, v in self.batch.items()}
        zero, _ = self.model(rest, torch.zeros_like(self.a))
        torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)

    def test_balanced_loss_is_invariant_to_duplication_within_a_group(self):
        y = torch.tensor([[.1], [.02]], dtype=torch.double)
        target = torch.zeros_like(y); active = torch.tensor([True, False])
        first = fit_loss(y, target, active, balanced=True)
        idx = torch.tensor([0, 1, 1, 1, 1])
        second = fit_loss(y[idx], target[idx], active[idx], balanced=True)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(fit_loss(y[:1], target[:1], active[:1], balanced=True)))

    def test_targets_do_not_enter_strategy_prediction_and_gradients_stay_finite(self):
        other = dict(self.batch, target_q=torch.randn(4, 30) * 100, target_x=torch.randn(4, 30) * 100)
        for bound, mix in [(True, 0), (False, .5), (True, 1)]:
            model = copy.deepcopy(self.model); model.angle_bound = bound; model.symmetry_mix = mix
            y, _ = model(self.batch, self.a); z, _ = model(other, self.a)
            torch.testing.assert_close(y, z, rtol=0, atol=0)
            y.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_ensemble_selection_rejects_active_regression_despite_mean_gain(self):
        from scripts.run_broad_search import eligible
        baseline = {'all': .2, 'active': .8, 'quiet': .15}
        rule = {'all_max_ratio': 1., 'active_max_ratio': 1.01, 'quiet_max_ratio': 1.01}
        self.assertFalse(eligible({'all': .19, 'active': .82, 'quiet': .14}, baseline, rule))
        self.assertTrue(eligible(baseline, baseline, rule))

    def test_preparation_strips_targets_and_never_uses_future_labels(self):
        import numpy as np
        from scripts.run_broad_search import inputs
        from scripts.predict_cranepilot import build_packet
        from cranepilot.physics import LengthCalibration
        theta = np.sin(np.arange(41)[None, :] * .04) * .03
        raw = build_packet(theta, np.zeros_like(theta), np.zeros_like(theta),
                           np.array([.4]), np.zeros((1, 30)), LengthCalibration())
        first = inputs(raw)
        raw['target_q'] = np.full((1, 30), 1e10); raw['target_x'] = np.full((1, 30), -1e10)
        second = inputs(raw)
        self.assertFalse(any(k.startswith('target') for k in second))
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
