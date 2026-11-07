import unittest
import numpy as np

from cranepilot.data import QUANTIZATION_RAD
from scripts.combined_history_baseline import adaptive_free_forecast


def trajectory(time, omega, damping, a=.13, b=.07):
    wd = np.sqrt(omega**2 - damping**2 / 4)
    return np.exp(-damping * time / 2) * (a * np.cos(wd * time) + b * np.sin(wd * time))


class CombinedHistoryBaselineTests(unittest.TestCase):
    def test_recovers_analytic_damped_motion_and_exact_forecast_times(self):
        past = trajectory(-np.arange(41) * .01, 4.3, .2)[None]
        expected = trajectory(np.arange(1, 31) * .05, 4.3, .2)[None]
        prediction, chosen = adaptive_free_forecast(past, 4.3)
        np.testing.assert_allclose(prediction, expected, atol=1e-12, rtol=0)
        self.assertEqual(chosen['frequency_ratio'][0], 1.)
        self.assertEqual(chosen['damping_per_s'][0], .2)

    def test_corrects_grid_frequency_mismatch_using_history_only(self):
        nominal = 6.4
        past = trajectory(-np.arange(41) * .01, nominal * .85, .1)[None]
        expected = trajectory(np.arange(1, 31) * .05, nominal * .85, .1)[None]
        prediction, chosen = adaptive_free_forecast(past, nominal)
        self.assertEqual(chosen['frequency_ratio'][0], .85)
        np.testing.assert_allclose(prediction, expected, atol=1e-12, rtol=0)

    def test_quantized_predictions_are_finite_and_packet_independent(self):
        omega = np.array([3., 5., 8.])
        past = np.stack([trajectory(-np.arange(41) * .01, w, .1) for w in omega])
        past = np.round(past / QUANTIZATION_RAD) * QUANTIZATION_RAD
        prediction, chosen = adaptive_free_forecast(past, omega)
        self.assertEqual(prediction.shape, (3, 30))
        self.assertTrue(np.isfinite(prediction).all())
        for i in range(3):
            single, meta = adaptive_free_forecast(past[i:i+1], omega[i])
            np.testing.assert_allclose(single[0], prediction[i], atol=1e-14, rtol=0)
            for name in chosen:
                np.testing.assert_allclose(meta[name][0], chosen[name][i], atol=1e-14, rtol=0)

    def test_rejects_invalid_history_and_frequency_contract(self):
        good = np.zeros((2, 41))
        cases = [(np.zeros((2, 42)), 4.), (np.zeros((0, 41)), 4.),
                 (np.full((2, 41), np.nan), 4.), (good, 0.),
                 (good, np.inf), (good, [3., 4., 5.]),
                 (good, np.ones((2, 1)))]
        for past, frequency in cases:
            with self.subTest(shape=past.shape, omega=frequency):
                with self.assertRaises(ValueError):
                    adaptive_free_forecast(past, frequency)


if __name__ == '__main__':
    unittest.main()
