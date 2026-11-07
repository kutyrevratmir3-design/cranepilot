import unittest
from unittest.mock import patch
import numpy as np
from cranepilot.reliability_pilot import FrozenReliabilityPilot

class ReliabilityPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pilot=FrozenReliabilityPilot()
    def packet(self):
        cal=self.pilot.cases['trolley_theta']['calibration'];omega=np.sqrt(9.81/(cal['a']*.4+cal['b']));t=-np.arange(401)*.01
        return dict(theta=(.02*np.cos(omega*t))[None,:],L=np.full((1,401),.4),x=np.zeros((1,401)),xr=np.zeros((1,401))),dict(xr=np.zeros((1,151)))
    def test_data_free_blend_matches_returned_parts(self):
        history,future=self.packet()
        with patch('numpy.load',side_effect=AssertionError('Inference needs no files')):r=self.pilot.forecast(history,future,family='trolley',axis='theta',known_plan=True)
        w=r['modal_weight'];self.assertTrue(((w>=0)&(w<=1)).all());np.testing.assert_allclose(r['forecast_rad'],w[:,None]*r['modal_forecast_rad']+(1-w[:,None])*r['reference_arx_rad'],atol=1e-14)
    def test_rejects_unknown_future_and_preserves_gate(self):
        history,future=self.packet()
        with self.assertRaises(ValueError):self.pilot.forecast(history,future,family='trolley',axis='theta',known_plan=False)
        history['L'][0,-1]+=.003;r=self.pilot.forecast(history,future,family='trolley',axis='theta',known_plan=True);self.assertFalse(r['eligible'][0]);self.assertTrue(np.isnan(r['forecast_rad']).all());self.assertTrue(np.isnan(r['modal_weight']).all())

if __name__=='__main__':unittest.main()
