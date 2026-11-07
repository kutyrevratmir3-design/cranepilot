import copy
import io
import unittest

import numpy as np
import torch

from cranepilot.broad_strategies import make_model
from scripts.residual_adapter import (make_adapter, fit_adapter,
                                     cache_frozen_features, oscillator_transition)


class ResidualAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)

    def batch(self, n=6):
        return {'angular_context': torch.randn(n, 125, dtype=torch.double)*.2,
                'initial_q': torch.randn(n, dtype=torch.double)*.02,
                'initial_p': torch.randn(n, dtype=torch.double)*.02,
                'omega': torch.linspace(3., 7., n, dtype=torch.double),
                'future_u': torch.randn(n, 30, dtype=torch.double)*.1,
                'observer_residual_rms': torch.full((n,), .002, dtype=torch.double),
                'acceleration': torch.randn(n, 30, dtype=torch.double)*.05}

    def model(self, **choice):
        return make_adapter(make_model({'symmetry_mix': .5}).double(), choice)

    def test_zero_initialization_exactly_retains_base_and_mode_freezing(self):
        batch = self.batch()
        for correction in ('force', 'direct'):
            model = self.model(correction=correction, hidden=8)
            model.train()
            self.assertTrue(model.training)
            self.assertTrue(model.adapter.training)
            self.assertFalse(model.base.training)
            self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
            expected = model.base(batch, batch['acceleration'])[0]
            prediction, diagnostics = model(batch, batch['acceleration'])
            torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
            self.assertEqual(float(diagnostics['adapter_correction'].abs().max().detach()), 0.)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 321)

    def test_exact_oscillator_semigroup_equilibrium_and_free_energy(self):
        omega = torch.tensor([.2, 4., 20.], dtype=torch.double)
        transition, port = oscillator_transition(omega)
        double_transition, double_port = oscillator_transition(omega, step=.1)
        torch.testing.assert_close(double_transition, transition @ transition, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(double_port, (transition @ port[..., None])[..., 0]+port,
                                   atol=1e-13, rtol=1e-13)
        equilibrium = torch.tensor([[.1, 0.]], dtype=torch.double).repeat(3, 1)
        torch.testing.assert_close((transition @ equilibrium[..., None])[..., 0]+port*.1,
                                   equilibrium, atol=1e-14, rtol=1e-14)
        state = torch.randn(3, 2, dtype=torch.double)
        for _ in range(100):
            previous = state.square().sum(1)
            state = (transition @ state[..., None])[..., 0]
            self.assertTrue(torch.all(state.square().sum(1) <= previous+1e-14))
        self.assertTrue(torch.all(torch.linalg.eigvals(transition).abs() < 1.))

    def test_cached_and_uncached_nonzero_adapter_agree_and_ignore_targets(self):
        batch = self.batch()
        for correction in ('force', 'direct'):
            model = self.model(correction=correction, hidden=16)
            with torch.no_grad():
                model.adapter[-1].weight.normal_(0, .1)
                model.adapter[-1].bias.fill_(.12)
            expected, _ = model(batch, batch['acceleration'])
            prepared = cache_frozen_features(model, batch, chunk=2)
            self.assertTrue(all(not value.requires_grad for value in prepared.values()))
            actual, diagnostics = model.forward_prepared(prepared)
            torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
            self.assertLessEqual(float(diagnostics['innovation_force'].abs().max().detach()), model.bound)
            poisoned = dict(batch, target_q=torch.full((6, 30), float('nan'), dtype=torch.double),
                            target_x=torch.full((6, 30), float('inf'), dtype=torch.double))
            torch.testing.assert_close(model(poisoned, batch['acceleration'])[0], expected, rtol=0, atol=0)

    def test_command_acceleration_prefix_and_input_gradients(self):
        batch = self.batch(3)
        model = self.model(correction='force')
        with torch.no_grad():
            model.adapter[-1].weight.normal_(0, .1)
        original = model(batch, batch['acceleration'])[0]
        modified = {key: value.clone() for key, value in batch.items()}
        modified['future_u'][:, 10:] += 2.
        modified['acceleration'][:, 10:] -= .3
        changed = model(modified, modified['acceleration'])[0]
        torch.testing.assert_close(changed[:, :10], original[:, :10], rtol=0, atol=0)
        self.assertFalse(torch.equal(changed[:, 10:], original[:, 10:]))
        acceleration = batch['acceleration'].clone().requires_grad_()
        model(batch, acceleration)[0].square().mean().backward()
        self.assertIsNotNone(acceleration.grad)
        self.assertTrue(torch.isfinite(acceleration.grad).all())
        self.assertGreater(float(acceleration.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_train_reload_all_choices_and_preserve_every_base_tensor(self):
        train, validation = self.batch(8), self.batch(4)
        active, valactive = np.array([False, True]*4), np.array([False, True]*2)
        config = {'epochs': 1, 'patience': 1, 'batch_size': 4,
                  'learning_rate': .001, 'weight_decay': .0001}
        for choice in ({'correction': 'force', 'hidden': 8},
                       {'correction': 'force', 'hidden': 16},
                       {'correction': 'direct', 'hidden': 8},
                       {'correction': 'force', 'hidden': 8, 'phase_space': True}):
            model = self.model(**choice)
            before = copy.deepcopy(model.base.state_dict())
            with torch.no_grad():
                target = model(train, train['acceleration'])[0]+.015
                valtarget = model(validation, validation['acceleration'])[0]+.015
            state, logs, metadata = fit_adapter(model, train, target, active,
                np.array([1]*4+[2]*4), validation, valtarget, valactive, config, choice, 42)
            self.assertEqual(metadata['optimizer_steps'], 2)
            self.assertEqual(metadata['backward_passes'], 2)
            self.assertEqual(metadata['epochs_executed'], 1)
            self.assertTrue(metadata['base_frozen_exact'])
            self.assertEqual(logs[0]['epoch'], -1)
            for key, value in model.base.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer.seek(0)
            reloaded = self.model(**choice)
            reloaded.load_state_dict(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(reloaded(validation, validation['acceleration'])[0],
                                       model(validation, validation['acceleration'])[0], rtol=0, atol=0)

    def test_validation_regression_rolls_back_zero_head_after_real_updates(self):
        one = self.batch(1)
        batch = {key: value.repeat(4, *([1]*(value.ndim-1))) for key, value in one.items()}
        choice = {'correction': 'direct', 'hidden': 8}
        model = self.model(**choice)
        original = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            base = model(batch, batch['acceleration'])[0]
        config = {'epochs': 4, 'patience': 2, 'batch_size': 4,
                  'learning_rate': .01, 'weight_decay': 0.}
        selected, logs, metadata = fit_adapter(model, batch, base+.03, np.array([False, True]*2),
            np.array([1, 1, 2, 2]), batch, base-.03, np.array([False, True]*2), config, choice, 19)
        self.assertEqual(metadata['selected_epoch'], -1)
        self.assertEqual(metadata['optimizer_steps'], 2)
        self.assertTrue(all(row['selection_score'] > 1 for row in logs[1:]))
        for key, value in selected.items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
