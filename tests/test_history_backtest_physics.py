import unittest
import numpy as np
from scipy.linalg import expm
from scripts.history_backtest_physics import history_prediction, backtest_forced_forecast
from scripts.forced_adaptive_physics import forced_fit, damped_rollout


def exact_piecewise_history(omega=6.3, damping=.2):
    time = np.arange(41)*.01
    x = .03*np.sin(4*time) + .02*time*time
    slopes = np.diff(x)/.01
    matrix = np.array([[0., 1.], [-omega**2, -damping]])
    transition = expm(.01*matrix)
    state = np.array([.08, -.17])
    angles = [state[0]]
    for j in range(40):
        if j: state[1] -= omega**2/9.81*(slopes[j]-slopes[j-1])
        state = transition @ state
        angles.append(state[0])
    return np.asarray(angles)[None, ::-1], x[None, ::-1], slopes[-1]


class HistoryBacktestPhysicsTests(unittest.TestCase):
    def test_withheld_angles_cannot_change_prefix_fit_or_predictions(self):
        theta, x, _ = exact_piecewise_history()
        for prefix in (21, 31):
            predicted, meta = history_prediction(theta, x, 6.3, .2, prefix)
            poisoned = theta.copy(); poisoned[:, :41-prefix] += 100
            changed, other = history_prediction(poisoned, x, 6.3, .2, prefix)
            np.testing.assert_array_equal(predicted, changed)
            np.testing.assert_array_equal(meta['coefficients'], other['coefficients'])
            # Those samples do enter the parameter-selection validation score.
            _, fit = backtest_forced_forecast(theta, x, 6.3, np.zeros((1,30)), prefixes=(prefix,))
            _, bad = backtest_forced_forecast(poisoned, x, 6.3, np.zeros((1,30)), prefixes=(prefix,))
            self.assertGreater(bad['backtest_mse_rad2'][0], fit['backtest_mse_rad2'][0] + 1)

    def test_later_observed_positions_cannot_change_earlier_fit(self):
        theta, x, _ = exact_piecewise_history()
        prefix = 21
        base, meta = history_prediction(theta, x, 6.3, .2, prefix)
        changed_x = x.copy(); changed_x[:, :41-prefix] += np.linspace(.01, .1, 41-prefix)
        changed, other = history_prediction(theta, changed_x, 6.3, .2, prefix)
        np.testing.assert_array_equal(base[:, :prefix], changed[:, :prefix])
        np.testing.assert_array_equal(meta['coefficients'], other['coefficients'])
        self.assertGreater(float(np.max(abs(base[:, prefix:] - changed[:, prefix:]))), .01)

    def test_withheld_forced_trajectory_matches_independent_matrix_solution(self):
        theta, x, _ = exact_piecewise_history()
        for prefix in (21, 31):
            predicted, _ = history_prediction(theta, x, 6.3, .2, prefix)
            np.testing.assert_allclose(predicted, theta[:, ::-1], atol=2e-11, rtol=2e-10)
        # A sufficiently excited case identifies an exact-grid oscillator.
        _, meta = backtest_forced_forecast(theta, x, 6.3/1.15, np.zeros((1,30)), prefixes=(21,31))
        self.assertEqual(meta['frequency_ratio'][0], 1.15)
        self.assertEqual(meta['damping_per_s'][0], .2)

    def test_selected_state_is_refitted_on_all_samples(self):
        rng = np.random.default_rng(15)
        theta = rng.normal(0, .03, (3,41)); x = rng.normal(0,.001,(3,41))
        omega = np.array([3.,5.,8.]); a = rng.normal(0,.2,(3,30)); v = np.array([.01, .02, -.03])
        forecast, meta = backtest_forced_forecast(theta, x, omega, a, velocity=v, prefixes=(21,31))
        q, qdot, residual = forced_fit(theta, x, meta['omega_rad_s'], meta['damping_per_s'], v)
        np.testing.assert_array_equal(q, meta['initial_q'])
        np.testing.assert_array_equal(qdot, meta['initial_qdot'])
        np.testing.assert_array_equal(np.sqrt(np.mean(residual**2,axis=1)), meta['full_history_rms'])
        np.testing.assert_array_equal(damped_rollout(q,qdot,meta['omega_rad_s'],meta['damping_per_s'],a), forecast)

    def test_guarded_zero_excitation_falls_back_and_ties_remain_visible(self):
        theta = np.zeros((2,41)); x = theta.copy(); a = np.zeros((2,30))
        y, meta = backtest_forced_forecast(theta,x,[3.,7.],a,prefixes=(21,31),guard_fraction=.1,guard_absolute_bins2=1.)
        np.testing.assert_array_equal(y,0.)
        np.testing.assert_array_equal(meta['frequency_ratio'],1.)
        np.testing.assert_array_equal(meta['damping_per_s'],0.)
        np.testing.assert_array_equal(meta['score_gap_rad2'],0.)
        np.testing.assert_array_equal(meta['accepted_adaptation'],False)
        # Custom grids may omit nominal. A guard still has a valid nominal fallback.
        _, other = backtest_forced_forecast(theta,x,[3.,7.],a,ratios=(.85,),dampings=(.2,),guard_fraction=.1)
        np.testing.assert_array_equal(other['frequency_ratio'],1.)
        np.testing.assert_array_equal(other['chosen_before_guard_ratio'],.85)
        np.testing.assert_array_equal(other['guard_rejected_candidate'],True)

    def test_future_acceleration_cannot_change_selection_and_packets_are_independent(self):
        rng = np.random.default_rng(7)
        theta = rng.normal(0,.03,(3,41)); x = rng.normal(0,.001,(3,41))
        omega = np.array([3.,5.,8.]); a = rng.normal(0,.2,(3,30))
        kwargs = dict(prefixes=(21,31),guard_fraction=.1,guard_absolute_bins2=1.)
        y, meta = backtest_forced_forecast(theta,x,omega,a,**kwargs)
        changed, other = backtest_forced_forecast(theta,x,omega,-5*a,**kwargs)
        for key in meta: np.testing.assert_array_equal(meta[key],other[key])
        self.assertGreater(float(np.max(abs(y-changed))),.01)
        reflected, _ = backtest_forced_forecast(-theta,-x,omega,-a,**kwargs)
        np.testing.assert_allclose(reflected,-y,atol=2e-12,rtol=2e-11)
        for j in range(3):
            single, chosen = backtest_forced_forecast(theta[j:j+1],x[j:j+1],omega[j],a[j:j+1],**kwargs)
            np.testing.assert_allclose(single,y[j:j+1],atol=2e-12,rtol=2e-11)
            for key in meta: np.testing.assert_allclose(chosen[key],meta[key][j:j+1],atol=2e-12,rtol=2e-11)

    def test_multiple_splits_have_equal_weight_and_mixed_validity_is_supported(self):
        theta,x,_ = exact_piecewise_history()
        _, first = backtest_forced_forecast(theta,x,6.3,np.zeros((1,30)),ratios=(1.,),dampings=(0.,),prefixes=(21,))
        _, second = backtest_forced_forecast(theta,x,6.3,np.zeros((1,30)),ratios=(1.,),dampings=(0.,),prefixes=(31,))
        _, both = backtest_forced_forecast(theta,x,6.3,np.zeros((1,30)),ratios=(1.,),dampings=(0.,),prefixes=(21,31))
        np.testing.assert_allclose(both['backtest_mse_rad2'],.5*(first['backtest_mse_rad2']+second['backtest_mse_rad2']),atol=1e-20)
        y,meta=backtest_forced_forecast(np.zeros((2,41)),np.zeros((2,41)),[.1,5.],np.zeros((2,30)),ratios=(1.,),dampings=(0.,.4))
        np.testing.assert_array_equal(meta['candidate_count'],[1,2])
        self.assertTrue(np.isfinite(y).all())

    def test_invalid_contracts_fail(self):
        theta=np.zeros((1,41)); x=theta.copy(); a=np.zeros((1,30))
        for kwargs in [dict(prefixes=()),dict(prefixes=(41,)),dict(prefixes=(21,21)),dict(prefixes=(21.5,)),
                       dict(guard_fraction=-1.),dict(guard_absolute_bins2=np.nan),dict(ratios=(1.,1.)),
                       dict(dampings=(50.,)),dict(penalty=-1.)]:
            with self.assertRaises(ValueError):backtest_forced_forecast(theta,x,5.,a,**kwargs)
        with self.assertRaises(ValueError):backtest_forced_forecast(theta,x,5.,a[:,:29])
        with self.assertRaises(ValueError):history_prediction(theta,x,5.,0.,True)


if __name__ == '__main__': unittest.main()
