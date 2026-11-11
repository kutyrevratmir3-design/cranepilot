import unittest
import numpy as np
import torch
from cranepilot.temporal_observer import CausalBlock, TemporalPhysicsForecaster, TCNResidualForecaster
from cranepilot.fast_inference import fast_prediction
from cranepilot.physics import LengthCalibration
from scripts.run_cleanroom import simulate_cases


class TemporalTests(unittest.TestCase):
    def batch(self):
        return {'angular_context': torch.randn(3, 125)*.2,
            'initial_q': torch.randn(3)*.1, 'initial_p': torch.randn(3)*.1,
            'omega': torch.tensor([3., 5., 7.]), 'initial_x': torch.zeros(3),
            'initial_velocity': torch.zeros(3), 'future_u': torch.randn(3, 30),
            'physical_base': torch.randn(3, 30)*.01}

    def test_causal_convolution_ignores_later_samples(self):
        torch.manual_seed(6)
        block = CausalBlock(4, 4).eval()
        x = torch.randn(2, 4, 41)
        changed = x.clone()
        changed[:, :, 20:] += 100
        torch.testing.assert_close(block(x)[:, :, :20], block(changed)[:, :, :20])

    def test_decoder_cannot_see_later_prescribed_inputs(self):
        torch.manual_seed(7)
        model = TCNResidualForecaster().eval()
        with torch.no_grad():
            model.output.weight.normal_()
        b = self.batch()
        a = torch.randn(3, 30)
        future = {k: v.clone() for k, v in b.items()}
        future['future_u'][:, 15:] += 100
        future['physical_base'][:, 15:] += 100
        a2 = a.clone()
        a2[:, 15:] += 100
        with torch.no_grad():
            y = model(b, a)[0]
            y2 = model(future, a2)[0]
        torch.testing.assert_close(y[:, :15], y2[:, :15])

    def test_no_target_dependence_and_finite_training_gradient(self):
        torch.manual_seed(8)
        b = self.batch()
        a = torch.randn(3, 30)
        for model in (TemporalPhysicsForecaster(), TCNResidualForecaster()):
            model.eval()
            y = model(b, a)[0]
            other = dict(b, target_q=torch.ones(3, 30)*1e8, target_x=torch.randn(3, 30))
            torch.testing.assert_close(y, model(other, a)[0])
            model.train()
            loss = (model(b, a)[0]-torch.randn(3, 30)*.1).square().mean()
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_temporal_physics_retains_compiled_flow_equivalence(self):
        torch.manual_seed(9)
        model = TemporalPhysicsForecaster().eval()
        b = self.batch()
        a = torch.randn(3, 30)
        with torch.no_grad():
            model.force_output.weight.normal_(0, .1)
            expected = model(b, a)[0]
            actual = fast_prediction(model, b, a)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_cleanroom_small_amplitude_matches_analytic_pendulum(self):
        cfg = {'length_m': [.5], 'amplitude_deg': [.001], 'phase_rad': [.7],
               'damping_per_s': [0.], 'measurement': ['exact']}
        cal = LengthCalibration()
        metadata, theta, target = simulate_cases(cfg, cal)
        w = cal.omega(.5)
        expected = np.deg2rad(.001)*np.cos(.7+w*(.4+np.arange(1, 31)*.05))
        np.testing.assert_allclose(target[0], expected, atol=1e-12)
        self.assertEqual(theta.shape, (1, 41))
