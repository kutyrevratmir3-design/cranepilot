import unittest
import numpy as np
from scipy.integrate import solve_ivp

from cranepilot.data import Scenario, make_windows
from cranepilot.physics import oscillator_forecast, pendulum_rhs, G
from cranepilot.models import sensitivity_basis


class ScientificContractTests(unittest.TestCase):
    def test_current_measurement_and_exact_horizon(self):
        time = np.arange(100) * .01
        theta = np.arange(100, dtype=float)
        s = Scenario(1, time, {"theta":theta, "L":np.ones(100)}, {"family":"trolley"})
        w = make_windows(s, "theta", [1, 10], history=3)
        np.testing.assert_array_equal(w["past"][0], [2, 1, 0])
        np.testing.assert_array_equal(w["y"][0], [3, 12])

    def test_forecast_features_cannot_change_when_future_is_modified(self):
        rng = np.random.default_rng(4)
        theta = rng.normal(size=100)
        scenario = Scenario(1, np.arange(100)*.01, {"theta":theta.copy(), "L":np.ones(100)}, {"family":"trolley"})
        before = make_windows(scenario,"theta",[10],history=21)
        row = np.flatnonzero(before["k"] == 40)[0]
        scenario.signals["theta"][41:] += 1e6
        after = make_windows(scenario,"theta",[10],history=21)
        np.testing.assert_array_equal(before["past"][row], after["past"][row])
        self.assertNotEqual(before["y"][row,0], after["y"][row,0])

    def test_analytic_oscillator_recovery_across_lengths_and_phases(self):
        rng = np.random.default_rng(5)
        omega = rng.uniform(2, 10, 30)
        phase = rng.uniform(-np.pi, np.pi, 30)
        amplitude = rng.uniform(.01, .4, 30)
        past = amplitude[:,None] * np.cos(omega[:,None]*(-np.arange(41)*.01) + phase[:,None])
        horizons = np.array([.1,.5,1.5])
        expected = amplitude[:,None] * np.cos(omega[:,None]*horizons + phase[:,None])
        actual, _ = oscillator_forecast(past, omega, horizons)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_damped_oscillator_recovery(self):
        omega, damping = np.array([4.0]), .2
        wd = np.sqrt(omega**2-damping**2/4)
        time = -np.arange(41)*.01
        past = (np.exp(-damping*time/2)*(.2*np.cos(wd*time)+.1*np.sin(wd*time)))[None]
        actual, _ = oscillator_forecast(past, omega, [.5,1.0], damping=damping)
        t = np.array([.5,1.0])
        expected = np.exp(-damping*t/2)*(.2*np.cos(wd*t)+.1*np.sin(wd*t))
        np.testing.assert_allclose(actual[0], expected, atol=1e-10)

    def test_undamped_nonlinear_energy_conservation(self):
        length = .7
        result = solve_ivp(lambda t,y: pendulum_rhs(y,0,length,damping=0), (0,20), [.4,0], rtol=1e-10, atol=1e-12)
        theta, rate = result.y
        energy = .5*length**2*rate**2 + G*length*(1-np.cos(theta))
        self.assertLess(np.ptp(energy)/energy[0], 2e-8)

    def test_zero_input_damping_dissipates_energy(self):
        length = .7
        result = solve_ivp(lambda t,y: pendulum_rhs(y,0,length,damping=.2), (0,10), [.4,0],
                           t_eval=np.linspace(0,10,1001), rtol=1e-10, atol=1e-12)
        theta, rate = result.y
        energy = .5*length**2*rate**2 + G*length*(1-np.cos(theta))
        self.assertLess(np.max(np.diff(energy)), 1e-9)

    def test_rank_deficient_sensitivity_has_no_invented_direction(self):
        s = np.zeros((2,7,2))
        s[0,:,0] = np.arange(7)
        s[0,:,1] = 2*np.arange(7)
        u = sensitivity_basis(s)
        self.assertEqual(np.linalg.matrix_rank(u[0]), 1)
        self.assertEqual(np.linalg.matrix_rank(u[1]), 0)


if __name__ == "__main__":
    unittest.main()

