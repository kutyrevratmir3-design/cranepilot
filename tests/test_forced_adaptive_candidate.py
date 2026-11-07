import unittest
import numpy as np
from scripts.forced_adaptive_candidate import compose
from scripts.reliability_blend import blend


class HybridCompositionTests(unittest.TestCase):
    def test_nominal_identity_preserves_existing_controls(self):
        rng=np.random.default_rng(17)
        nominal=rng.normal(size=(4,30));soft=rng.normal(size=(4,30));rms=np.array([0,.0001,.001,.1])
        ys=compose(soft,nominal,nominal,rms)
        np.testing.assert_allclose(ys['residual_transfer'],soft,atol=1e-15)
        expected=blend(soft,nominal,rms,.5)[0]
        for name in ('residual_gate','soft_adaptive_gate'): np.testing.assert_allclose(ys[name],expected,atol=1e-15)

    def test_gate_and_residual_transfer_have_distinct_limits(self):
        soft=np.full((2,30),3.); nominal=np.ones((2,30)); adaptive=np.full((2,30),2.)
        ys=compose(soft,nominal,adaptive,np.array([0.,1.]))
        np.testing.assert_array_equal(ys['residual_gate'][0],adaptive[0])
        np.testing.assert_array_equal(ys['residual_gate'][1],ys['residual_transfer'][1])
        np.testing.assert_array_equal(ys['soft_adaptive_gate'][1],soft[1])
        mirrored=compose(-soft,-nominal,-adaptive,np.array([0.,1.]))
        for name in ys: np.testing.assert_array_equal(ys[name],-mirrored[name])
