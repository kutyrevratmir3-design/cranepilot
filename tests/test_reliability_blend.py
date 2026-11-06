import unittest
import numpy as np
from scripts.reliability_blend import blend


class ReliabilityBlendTests(unittest.TestCase):
    def test_exact_consistency_recovers_physics_and_large_residual_recovers_network(self):
        neural = np.ones((2, 30)); physical = np.zeros((2, 30))
        y, weight = blend(neural, physical, np.array([0., 1.]), .25)
        np.testing.assert_array_equal(y[0], physical[0])
        np.testing.assert_array_equal(y[1], neural[1])
        self.assertTrue(((weight >= 0) & (weight <= 1)).all())

    def test_gate_is_odd_under_joint_reflection_and_batch_independent(self):
        rng = np.random.default_rng(16)
        neural, physical = rng.normal(size=(5, 30)), rng.normal(size=(5, 30))
        rms = np.linspace(0, .003, 5)
        y, _ = blend(neural, physical, rms, .5)
        z, _ = blend(-neural, -physical, rms, .5)
        np.testing.assert_array_equal(y, -z)
        for i in range(5):
            single, _ = blend(neural[i:i+1], physical[i:i+1], rms[i:i+1], .5)
            np.testing.assert_array_equal(y[i:i+1], single)

    def test_invalid_scale_and_residual_are_rejected(self):
        a = np.ones((1, 30))
        for residual, scale in [(np.array([0.]), 0), (np.array([-1.]), .5), (np.array([np.nan]), .5)]:
            with self.assertRaises(ValueError): blend(a, a, residual, scale)
