import copy
import unittest
import numpy as np

from cranepilot.data import Scenario
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows


class RolloutInformationTests(unittest.TestCase):
    def test_future_measured_positions_and_angles_are_targets_only(self):
        t=np.arange(401)*.01
        scenario=Scenario(1,t,{'theta':.1*np.sin(4*t),'x':.1*t,
                              'xr':np.ones(len(t)),'L':np.full(len(t),.5)}, {'family':'trolley'})
        before=command_windows(scenario,LengthCalibration(),[100])
        scenario.signals['theta'][101:]+=100
        scenario.signals['x'][101:]-=100
        after=command_windows(scenario,LengthCalibration(),[100])
        for key in ('omega','initial_x','initial_velocity','initial_q','initial_p',
                    'actuator_context','angular_context','future_u','harmonic'):
            np.testing.assert_array_equal(before[key],after[key])
        self.assertFalse(np.array_equal(before['target_q'],after['target_q']))
        self.assertFalse(np.array_equal(before['target_x'],after['target_x']))

    def test_future_command_is_available_only_as_prescribed_input(self):
        t=np.arange(401)*.01
        scenario=Scenario(1,t,{'theta':.1*np.sin(4*t),'x':.1*t,
                              'xr':np.ones(len(t)),'L':np.full(len(t),.5)}, {'family':'trolley'})
        before=command_windows(scenario,LengthCalibration(),[100])
        scenario.signals['xr'][105:]=2.
        after=command_windows(scenario,LengthCalibration(),[100])
        np.testing.assert_array_equal(before['angular_context'],after['angular_context'])
        self.assertFalse(np.array_equal(before['future_u'],after['future_u']))

    def test_hoisting_cannot_enter_fixed_length_protocol(self):
        t=np.arange(401)*.01
        scenario=Scenario(1,t,{'theta':.1*np.sin(4*t),'x':.1*t,
                              'xr':np.ones(len(t)),'L':.2+.1*t}, {'family':'trolley'})
        with self.assertRaises(ValueError):
            command_windows(scenario,LengthCalibration(),[100])
