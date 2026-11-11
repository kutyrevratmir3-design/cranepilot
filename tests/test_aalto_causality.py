import unittest
import numpy as np
from scripts.prepare_aalto import causal_hold


class EventAlignmentTests(unittest.TestCase):
    def test_future_value_changes_cannot_modify_past_inputs(self):
        time = np.array([.1, .4, .8])
        grid = np.array([0, .25, .5, .75])
        old, _ = causal_hold(time, time, [1, 2, 3], grid)
        new, _ = causal_hold(time, time, [1, 2, 1e9], grid)
        np.testing.assert_equal(old, new)
        np.testing.assert_equal(old, [np.nan, 1, 2, 2])

    def test_server_availability_and_source_time_both_bound_usage(self):
        value, age = causal_hold([.1, .2, .9], [.1, .6, .7], [1, 2, 3], [.5, .65, .8, 1.])
        np.testing.assert_equal(value, [1, 2, 2, 3])
        np.testing.assert_allclose(age, [.4, .45, .6, .1])

    def test_unsorted_events_and_absent_history_preserve_gaps(self):
        value, _ = causal_hold([.4, .1, .4], [.4, .1, .4], [2, 1, 3], [0, .25, .5])
        np.testing.assert_equal(value, [np.nan, 1, 3])
        value, age = causal_hold([], [], [], [0, 1])
        self.assertTrue(np.isnan(value).all() and np.isnan(age).all())
