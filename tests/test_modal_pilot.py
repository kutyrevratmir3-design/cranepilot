import unittest
from unittest.mock import patch
import numpy as np
from cranepilot.modal_pilot import FrozenModalPilot

class ModalPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pilot=FrozenModalPilot()
    def packet(self):
        cal=self.pilot.cases['trolley_theta']['calibration'];w=np.sqrt(9.81/(cal['a']*.4+cal['b']));t=-np.arange(401)*.01
        return dict(L=np.full((1,401),.4),theta=(.012+.04*np.cos(w*t)+.02*np.sin(w*t))[None,:],x=np.zeros((1,401)),xr=np.zeros((1,401))),dict(xr=np.zeros((1,151))),w
    def test_data_free_exact_oscillator(self):
        history,future,w=self.packet()
        # Models are loaded already. Forecast must not read any dataset/run/NPZ.
        with patch('numpy.load',side_effect=AssertionError('No inference data-file access')):r=self.pilot.forecast(history,future,family='trolley',axis='theta',known_plan=True)
        truth=.012+.04*np.cos(w*r['horizons_s'])+.02*np.sin(w*r['horizons_s']);np.testing.assert_allclose(r['forecast_rad'][0],truth,atol=3e-5);self.assertTrue(r['eligible'][0])
    def test_eligibility_and_known_plan_are_enforced(self):
        history,future,_=self.packet()
        with self.assertRaises(ValueError):self.pilot.forecast(history,future,family='trolley',axis='theta',known_plan=False)
        history['L'][0,-1]+=.003;r=self.pilot.forecast(history,future,family='trolley',axis='theta',known_plan=True);self.assertFalse(r['eligible'][0]);self.assertTrue(np.isnan(r['forecast_rad']).all())
    def test_shape_current_reference_and_unknown_fields(self):
        history,future,_=self.packet();future['xr'][:]=1
        r=self.pilot.forecast(history,future,family='trolley',axis='theta',known_plan=True);self.assertFalse(r['eligible'][0]);history['target']=np.ones((1,401))
        with self.assertRaises(ValueError):self.pilot.forecast(history,future,family='trolley',axis='theta',known_plan=True)

if __name__=='__main__':unittest.main()
