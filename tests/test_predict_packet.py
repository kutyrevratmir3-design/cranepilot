import unittest
import numpy as np
from scripts.predict_cranepilot import build_packet
from cranepilot.physics import LengthCalibration
from cranepilot.data import Scenario
from cranepilot.rollout import command_windows


class HistoryOnlyPredictionTests(unittest.TestCase):
    def test_history_only_builder_matches_benchmark_information(self):
        t=np.arange(401)*.01;theta=.1*np.sin(4*t);x=.02*np.sin(2*t);u=1+.2*np.sin(t)
        s=Scenario(1,t,{'theta':theta,'x':x,'xr':u,'L':np.full(len(t),.5)},{'family':'trolley'})
        calibration=LengthCalibration();k=np.array([100,150])
        old=command_windows(s,calibration,k);past=k[:,None]-np.arange(41)
        new=build_packet(theta[past],x[past],u[past],.5,u[k[:,None]+np.arange(30)*5],calibration)
        for key in new:np.testing.assert_allclose(new[key],old[key],atol=1e-12,rtol=1e-12)
        self.assertNotIn('target_q',new);self.assertNotIn('target_x',new)

    def test_inconsistent_origin_command_is_rejected(self):
        with self.assertRaises(ValueError):
            build_packet(np.zeros((1,41)),np.zeros((1,41)),np.ones((1,41)),.5,np.zeros((1,30)),LengthCalibration())
