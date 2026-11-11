"""Only synthetic packet and paired-API contract checks."""
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.timestamp_offset import new_packet, score_batch, valid_offsets


class TimestampOffsetTests(unittest.TestCase):
    def test_boundary_offsets_are_omitted_not_clamped(self):
        offsets = [-2, -1, 0, 1, 2]
        self.assertEqual(valid_offsets(0, offsets), [0, 1, 2])
        self.assertEqual(valid_offsets(25, offsets), [-2, -1, 0])
        self.assertEqual(valid_offsets(2, offsets), offsets)

    def test_history_uses_true_age_and_targets_use_origin(self):
        t = np.arange(700.)
        s = SimpleNamespace(signals={'theta': t+1, 'L': t+2, 'x': t+3, 'xr': t+4})
        history, target = new_packet(s, 'theta', 'trolley', 20, np.array([425]), np.array([10, 150]))
        np.testing.assert_array_equal(history['theta'][0], t[405-np.arange(401)] + 1)
        np.testing.assert_array_equal(history['x'][0], t[425-np.arange(401)] + 3)
        np.testing.assert_array_equal(target, [[436, 576]])

    def test_same_inputs_and_declared_age_despite_mutating_predictor(self):
        h = np.array([10, 150])
        inp = {'theta': np.ones((1, 401))}
        seen = []
        def forecast(history, *, family, axis, angle_age_samples):
            seen.append((history['theta'][0, 0], family, axis, angle_age_samples))
            history['theta'][:] = 99
            return dict(angle_age_samples=angle_age_samples, horizons_s=h*.01, forecast_rad=np.zeros((1, 2)))
        api = SimpleNamespace(forecast=forecast)
        score_batch(inp, np.zeros((1, 2)), api, api, 'trolley', 'theta', 22, h)
        self.assertEqual(seen, [(1., 'trolley', 'theta', 22)] * 2)
        self.assertEqual(inp['theta'][0, 0], 1.)


if __name__ == '__main__':
    unittest.main()
