import copy
import unittest
import numpy as np
from cranepilot.physics import LengthCalibration
from cranepilot.data import QUANTIZATION_RAD
from scripts.combined_stress_simulator import simulate_stress


class CombinedStressTests(unittest.TestCase):
    def setUp(self):
        self.calibration = LengthCalibration()
        self.cfg = {'length_m': [.13, .91], 'amplitude_deg': [5.0],
                    'phase_rad': [.6], 'conditions': [
                        {'name': 'nominal', 'omega_ratio': 1., 'damping_per_s': 0.},
                        {'name': 'fast', 'omega_ratio': 1.15, 'damping_per_s': .2},
                        {'name': 'hidden_impulse', 'omega_ratio': 1., 'damping_per_s': 0.,
                         'future_accel_m_s2': .15}],
                    'measurement': ['exact', 'quantized', 'noisy_quantized', 'biased_quantized'],
                    'noise_seed': 7112026, 'noise_std_bins': .25, 'bias_bins': .5}

    def test_repeatable_paired_observations_never_change_clean_targets(self):
        meta, past, future = simulate_stress(self.cfg, self.calibration)
        meta2, past2, future2 = simulate_stress(self.cfg, self.calibration)
        self.assertTrue(meta.equals(meta2))
        np.testing.assert_array_equal(past, past2)
        np.testing.assert_array_equal(future, future2)
        self.assertEqual(past.shape, (24, 41))
        self.assertEqual(future.shape, (24, 30))
        n = 6
        for j in range(1, 4):
            np.testing.assert_array_equal(future[:n], future[j*n:(j+1)*n])
            np.testing.assert_array_equal(meta.trajectory_id[:n], meta.trajectory_id[j*n:(j+1)*n])
        self.assertGreater(np.max(abs(past[:n] - past[2*n:3*n])), 0)
        self.assertGreater(np.max(abs(past[:n] - past[3*n:4*n])), 0)
        np.testing.assert_allclose(past[n:] / QUANTIZATION_RAD,
                                   np.round(past[n:] / QUANTIZATION_RAD), atol=1e-12)

    def test_timestamps_and_newest_first_against_small_angle_solution(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(length_m=[.91], amplitude_deg=[1e-5], conditions=[cfg['conditions'][0]], measurement=['exact'])
        _, past, future = simulate_stress(cfg, self.calibration)
        omega = self.calibration.omega(np.array([.91]))[0]
        amplitude = np.deg2rad(1e-5)
        expected_past = amplitude * np.cos(.6 + omega * (.4 - np.arange(41)*.01))
        expected_future = amplitude * np.cos(.6 + omega * (.4 + np.arange(1, 31)*.05))
        np.testing.assert_allclose(past[0], expected_past, atol=1e-15, rtol=0)
        np.testing.assert_allclose(future[0], expected_future, atol=1e-15, rtol=0)

    def test_unobserved_future_disturbance_is_identical_before_onset(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(length_m=[.13])
        meta, past, future = simulate_stress(cfg, self.calibration)
        for mode in cfg['measurement']:
            nominal = np.flatnonzero((meta.measurement == mode) & (meta.condition == 'nominal'))[0]
            hidden = np.flatnonzero((meta.measurement == mode) & (meta.condition == 'hidden_impulse'))[0]
            np.testing.assert_array_equal(past[nominal], past[hidden])
            np.testing.assert_array_equal(future[nominal, :5], future[hidden, :5])
            self.assertGreater(np.max(abs(future[nominal, 5:] - future[hidden, 5:])), .001)
            self.assertNotEqual(meta.trajectory_id[nominal], meta.trajectory_id[hidden])
            self.assertEqual(meta.noise_pair_id[nominal], meta.noise_pair_id[hidden])
        self.assertGreater(np.max(abs(past[0] - past[1])), .001)

    def test_rk4_step_halving_including_impulse_boundaries(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(amplitude_deg=[12.], measurement=['exact'])
        _, coarse_past, coarse_future = simulate_stress(cfg, self.calibration, dt=.001)
        _, fine_past, fine_future = simulate_stress(cfg, self.calibration, dt=.0005)
        self.assertLess(np.max(abs(np.rad2deg(coarse_past - fine_past))), 1e-7)
        self.assertLess(np.max(abs(np.rad2deg(coarse_future - fine_future))), 1e-7)

    def test_invalid_step_and_unknown_measurement_rejected(self):
        with self.assertRaises(ValueError):
            simulate_stress(self.cfg, self.calibration, dt=.003)
        cfg = dict(self.cfg, measurement=['unknown'])
        with self.assertRaises(ValueError):
            simulate_stress(cfg, self.calibration)


if __name__ == '__main__':
    unittest.main()
