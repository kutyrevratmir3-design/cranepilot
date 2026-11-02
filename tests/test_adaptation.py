import unittest
import numpy as np
from cranepilot.adaptation import fit_prefix_oscillator, fit_validated_prefix


class AdaptationTests(unittest.TestCase):
    def test_prefix_cannot_see_later_samples(self):
        t=np.arange(2001)*.01
        theta=.2*np.sin(4*t)
        x=np.zeros(len(t))
        a=fit_prefix_oscillator(theta,x,4.1,5.)
        theta[501:]=1000
        x[501:]=-1000
        b=fit_prefix_oscillator(theta,x,4.1,5.)
        self.assertEqual(a,b)

    def test_recovers_frequency_of_independent_analytic_signal(self):
        t=np.arange(2001)*.01
        theta=.2*np.sin(4*t)
        fit=fit_prefix_oscillator(theta,np.zeros(len(t)),4.3,10.)
        self.assertLess(abs(fit["omega"]-4),.01)

    def test_does_not_adapt_without_excitation(self):
        fit=fit_prefix_oscillator(np.zeros(1000),np.zeros(1000),4.,5.)
        self.assertEqual(fit["status"],"insufficient_excitation")
        self.assertEqual(fit["omega"],4.)

    def test_validated_calibration_recovers_unseen_analytic_frequency(self):
        t=np.arange(4001)*.01
        fit=fit_validated_prefix(.15*np.sin(4.*t+.3),4.3,20.)
        self.assertTrue(fit["accepted"])
        self.assertLess(abs(fit["omega"]-4.),.005)

    def test_validated_calibration_never_uses_post_budget_measurements(self):
        t=np.arange(4001)*.01
        theta=.15*np.sin(4*t+.3)
        before=fit_validated_prefix(theta,4.3,20.)
        theta[2001:]=np.nan
        self.assertEqual(before,fit_validated_prefix(theta,4.3,20.))

    def test_validated_calibration_rejects_training_regime_that_fails_later(self):
        t=np.arange(2001)*.01
        theta=.15*np.sin(4.*t)
        theta[:1200]=.15*np.sin(4.6*t[:1200])
        fit=fit_validated_prefix(theta,4.,20.)
        self.assertFalse(fit["accepted"])
        self.assertEqual(fit["status"],"rejected_by_validation")
        self.assertEqual(fit["omega"],4.)

    def test_validated_calibration_abstains_with_too_few_validation_blocks(self):
        t=np.arange(1001)*.01
        fit=fit_validated_prefix(.15*np.sin(4*t),4.3,3.)
        self.assertFalse(fit["accepted"])
        self.assertEqual(fit["status"],"insufficient_prefix")


if __name__=="__main__":
    unittest.main()

