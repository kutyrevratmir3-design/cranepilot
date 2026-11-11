import unittest
import numpy as np
from scipy.linalg import expm
from cranepilot.forced_observer import forced_observer, analytic_rollout
from scripts.combined_history_baseline import adaptive_free_forecast
from scripts.forced_adaptive_physics import forced_fit, damped_rollout, adaptive_forced_forecast


class ForcedAdaptivePhysicsTests(unittest.TestCase):
    def test_zero_damping_reproduces_legacy_forced_observer_and_rollout(self):
        rng = np.random.default_rng(714)
        t = np.arange(41)*.01
        omega = np.array([3.4, 5.7, 8.2])
        theta = rng.normal(0,.03,(3,41))
        x = np.stack([.02*np.sin(2*t),.04*np.cos(3*t),.01*t*t])[:,::-1]
        legacy_q, legacy_p, legacy_res = forced_observer(theta,x,omega)
        q, rate, residual = forced_fit(theta,x,omega,0.)
        np.testing.assert_allclose(q,legacy_q,atol=2e-13,rtol=2e-12)
        np.testing.assert_allclose(rate,legacy_p*omega,atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(residual,legacy_res,atol=2e-13,rtol=2e-12)
        acceleration = rng.normal(0,.3,(3,30))
        expected = analytic_rollout(legacy_q,legacy_p,omega,acceleration)
        actual = damped_rollout(q,rate,omega,0.,acceleration)
        np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)

    def test_damped_forced_history_and_future_match_independent_matrix_exponential(self):
        omega, damping = 6.3, .27
        kappa = omega**2/9.81
        t = np.arange(41)*.01
        x = .03*np.sin(4*t)+.02*t*t
        slopes = np.diff(x)/.01
        matrix = np.array([[0.,1.],[-omega**2,-damping]])
        transition = expm(.01*matrix)
        state = np.array([.08,-.17]); theta = [state[0]]
        for j in range(40):
            if j: state[1] -= kappa*(slopes[j]-slopes[j-1])
            state = transition@state
            theta.append(state[0])
        q, rate, residual = forced_fit(np.array(theta)[None,::-1],x[None,::-1],omega,damping,velocity=slopes[-1])
        np.testing.assert_allclose([q[0],rate[0]],state,atol=3e-12,rtol=2e-11)
        self.assertLess(float(np.abs(residual).max()),3e-12)
        acceleration = np.linspace(-.15,.3,30)[None]
        expected=[]
        for a in acceleration[0]:
            augmented=np.array([[0.,1.,0.],[-omega**2,-damping,-kappa*a],[0.,0.,0.]])
            state=(expm(.05*augmented)@np.r_[state,1.])[:2]
            expected.append(state[0])
        actual=damped_rollout(q,rate,omega,damping,acceleration)
        np.testing.assert_allclose(actual[0],expected,atol=3e-12,rtol=2e-11)

    def test_autonomous_adaptation_reproduces_previous_causal_baseline(self):
        t=-np.arange(41)*.01
        nominal=np.array([4.2,6.7,8.3])
        actual=nominal*np.array([.85,1.,1.15]);damping=np.array([.1,.2,.4])
        wd=np.sqrt(actual**2-damping**2/4)
        theta=np.exp(-damping[:,None]*t/2)*(.1*np.cos(wd[:,None]*t)+.04*np.sin(wd[:,None]*t))
        expected, old=adaptive_free_forecast(theta,nominal)
        actual, choices=adaptive_forced_forecast(theta,np.zeros_like(theta),nominal,np.zeros((3,30)))
        np.testing.assert_allclose(actual,expected,atol=5e-12,rtol=5e-11)
        np.testing.assert_array_equal(choices['frequency_ratio'],old['frequency_ratio'])
        np.testing.assert_array_equal(choices['damping_per_s'],old['damping_per_s'])

    def test_future_input_cannot_select_parameters_and_reflection_is_equivariant(self):
        rng=np.random.default_rng(7);theta=rng.normal(0,.03,(4,41));x=np.cumsum(rng.normal(0,.0002,(4,41)),axis=1)
        omega=np.array([3.,4.,5.,6.]);a=rng.normal(0,.2,(4,30))
        y,meta=adaptive_forced_forecast(theta,x,omega,a)
        changed,other=adaptive_forced_forecast(theta,x,omega,-5*a)
        for name in meta: np.testing.assert_array_equal(meta[name],other[name])
        self.assertTrue((meta['score_gap_rad2']>=0).all())
        np.testing.assert_allclose(meta['penalized_score_rad2']+meta['score_gap_rad2'],meta['second_best_score_rad2'])
        np.testing.assert_array_equal(meta['candidate_count'],20)
        self.assertGreater(float(np.max(abs(y-changed))),1e-3)
        reflected,negmeta=adaptive_forced_forecast(-theta,-x,omega,-a)
        np.testing.assert_allclose(reflected,-y,atol=1e-12,rtol=1e-11)
        for j in range(4):
            single,_=adaptive_forced_forecast(theta[j:j+1],x[j:j+1],omega[j],a[j:j+1])
            np.testing.assert_allclose(single,y[j:j+1],atol=1e-12,rtol=1e-11)

    def test_absolute_trolley_offset_does_not_change_forecast(self):
        rng=np.random.default_rng(9);theta=rng.normal(0,.02,(2,41));x=rng.normal(0,.001,(2,41))
        a=rng.normal(0,.1,(2,30))
        y,_=adaptive_forced_forecast(theta,x,5.,a)
        shifted,_=adaptive_forced_forecast(theta,x+np.array([5.,-8.])[:,None],5.,a)
        np.testing.assert_allclose(y,shifted,atol=5e-12,rtol=5e-11)

    def test_mixed_underdamped_validity_is_handled_per_packet(self):
        t=-np.arange(41)*.01
        omega=np.array([.1,5.])
        theta=.08*np.cos(omega[:,None]*t)+.03*np.sin(omega[:,None]*t)
        x=np.zeros_like(theta); acceleration=np.zeros((2,30))
        prediction,meta=adaptive_forced_forecast(theta,x,omega,acceleration,
            ratios=(1.,),dampings=(0.,.4))
        # d=.4 is invalid for omega=.1, but valid for omega=5. The other
        # packet must retain both candidates without changing the first fit.
        np.testing.assert_array_equal(meta['candidate_count'],[1,2])
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(meta['score_gap_rad2'][0],0.)
        for j in range(2):
            single,choice=adaptive_forced_forecast(theta[j:j+1],x[j:j+1],omega[j],
                acceleration[j:j+1],ratios=(1.,),dampings=(0.,.4))
            np.testing.assert_allclose(prediction[j:j+1],single,atol=1e-12,rtol=1e-11)
            for name in meta:
                np.testing.assert_allclose(meta[name][j:j+1],choice[name],atol=1e-12,rtol=1e-11)

    def test_zero_excitation_exposes_tie_without_inventing_identifiability(self):
        theta=np.zeros((2,41)); x=theta.copy(); omega=np.array([3.,7.])
        prediction,meta=adaptive_forced_forecast(theta,x,omega,np.zeros((2,30)))
        np.testing.assert_array_equal(prediction,0.)
        # All nominal-frequency damping candidates fit zero history exactly.
        # The prior and fixed enumeration select nominal/zero damping, but the
        # zero best-versus-second-best gap records the remaining ambiguity.
        np.testing.assert_array_equal(meta['frequency_ratio'],1.)
        np.testing.assert_array_equal(meta['damping_per_s'],0.)
        np.testing.assert_array_equal(meta['score_gap_rad2'],0.)
        np.testing.assert_array_equal(meta['penalized_score_rad2'],0.)
        np.testing.assert_array_equal(meta['candidate_count'],20)
        np.testing.assert_array_equal(meta['ratio_boundary_hit'],False)
        future_changed,other=adaptive_forced_forecast(theta,x,omega,np.ones((2,30)))
        for name in meta: np.testing.assert_array_equal(meta[name],other[name])
        self.assertGreater(float(np.max(np.abs(future_changed))),.01)

    def test_invalid_contracts_fail_and_fixed_nominal_grid_is_supported(self):
        theta=np.zeros((1,41));x=theta.copy();a=np.zeros((1,30))
        y,meta=adaptive_forced_forecast(theta,x,5.,a,ratios=(1.,),dampings=(0.,))
        np.testing.assert_array_equal(y,0.)
        self.assertEqual(meta['frequency_ratio'][0],1.)
        self.assertEqual(meta['candidate_count'][0],1)
        self.assertEqual(meta['score_gap_rad2'][0],0.)
        for kwargs in [{'omega':0.},{'omega':np.nan},{'omega':5.,'penalty':-1.},
                       {'omega':5.,'ratios':()},{'omega':5.,'dampings':(-1.,)},
                       {'omega':5.,'dampings':(20.,)}]:
            with self.assertRaises(ValueError): adaptive_forced_forecast(theta,x,acceleration=a,**kwargs)
        with self.assertRaises(ValueError): forced_fit(theta,x,1.,3.)
        with self.assertRaises(ValueError): forced_fit(theta[:,:40],x[:,:40],5.,0.)
        with self.assertRaises(ValueError): damped_rollout(0.,0.,5.,0.,np.zeros((1,29)))


if __name__=='__main__': unittest.main()
