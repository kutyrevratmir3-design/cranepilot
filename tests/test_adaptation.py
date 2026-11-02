import unittest
import numpy as np
from cranepilot.adaptation import fit_prefix_oscillator


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


if __name__=="__main__":
    unittest.main()

