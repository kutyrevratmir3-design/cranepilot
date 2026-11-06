import unittest
import numpy as np
import torch
from cranepilot.innovation import InnovationForecaster
from cranepilot.forced_observer import analytic_rollout
from scripts.limit_ablation import COMPONENTS, predict_disabled


class LimitInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)
        self.model = InnovationForecaster(True).double().eval()
        with torch.no_grad():
            for layer in [self.model.initial, self.model.observer, self.model.force_output]:
                layer.weight.normal_(0, .05)
                layer.bias.normal_(0, .1)
        self.batch = {'angular_context': torch.randn(5, 125, dtype=torch.double) * .2,
                      'initial_q': torch.randn(5, dtype=torch.double) * .1,
                      'initial_p': torch.randn(5, dtype=torch.double) * .1,
                      'omega': torch.linspace(3, 10, 5, dtype=torch.double),
                      'future_u': torch.randn(5, 30, dtype=torch.double)}
        self.acceleration = torch.randn(5, 30, dtype=torch.double) * .3

    def test_unchanged_intervention_matches_original_reference(self):
        with torch.no_grad():
            expected = self.model(self.batch, self.acceleration)[0]
        actual = predict_disabled(self.model, self.batch, self.acceleration)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)

    def test_all_off_recovers_analytic_forced_physics(self):
        b = self.batch
        expected = analytic_rollout(b['initial_q'].numpy(), b['initial_p'].numpy(),
                                    b['omega'].numpy(), self.acceleration.numpy())
        actual = predict_disabled(self.model, b, self.acceleration, COMPONENTS).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-11)

    def test_interventions_restore_weights_hooks_and_behavior(self):
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        expected = predict_disabled(self.model, self.batch, self.acceleration)
        for disabled in [COMPONENTS, {'force'}, {'closure'}, {'memory'}]:
            predict_disabled(self.model, self.batch, self.acceleration, disabled)
            self.assertTrue(self.model.driven)
            self.assertFalse(self.model.initial._forward_hooks)
            self.assertFalse(self.model.observer._forward_hooks)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        torch.testing.assert_close(predict_disabled(self.model, self.batch, self.acceleration),
                                   expected, atol=0, rtol=0)

    def test_target_values_do_not_enter_any_intervention(self):
        b = dict(self.batch, target_q=torch.full((5, 30), 1e8), target_x=torch.full((5, 30), -1e8))
        for disabled in [set(), COMPONENTS, {'force'}, {'frequency'}]:
            torch.testing.assert_close(predict_disabled(self.model, b, self.acceleration, disabled),
                                       predict_disabled(self.model, self.batch, self.acceleration, disabled),
                                       atol=0, rtol=0)

    def test_training_mode_and_unknown_interventions_are_rejected(self):
        with self.assertRaises(ValueError):
            predict_disabled(self.model, self.batch, self.acceleration, {'unknown'})
        with self.assertRaises(ValueError):
            predict_disabled(self.model.train(), self.batch, self.acceleration)
