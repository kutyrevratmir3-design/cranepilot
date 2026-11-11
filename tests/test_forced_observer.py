import unittest
import numpy as np
from scipy.integrate import solve_ivp
from cranepilot.forced_observer import forced_observer
from cranepilot.forced_observer import refine_windows,phase_kernel_features
from cranepilot.data import Scenario
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import command_windows,tensors


class ForcedObserverTests(unittest.TestCase):
    def test_recovers_analytic_state_under_uniform_translation(self):
        t=np.arange(41)*.01;omega=np.array([3.,5.,7.])
        theta=.1*np.cos(omega[:,None]*t)+.03*np.sin(omega[:,None]*t)
        x=np.broadcast_to(.2*t+5.,theta.shape)
        q,p,_=forced_observer(theta[:,::-1],x[:,::-1],omega,velocity=np.full(3,.2))
        np.testing.assert_allclose(q,theta[:,-1],atol=1e-10)
        expected=-.1*np.sin(omega*t[-1])+.03*np.cos(omega*t[-1])
        np.testing.assert_allclose(p,expected,atol=1e-10)

    def test_independent_forced_ode_and_position_offset(self):
        t=np.arange(401)*.001;w=5.;v=.08*3*np.cos(3*t[-1])
        x=.08*np.sin(3*t)
        sol=solve_ivp(lambda t,z:[z[1],-w*w*z[0]+w*w/9.81*.08*9*np.sin(3*t)],
                      (0,t[-1]),[.04,.02],t_eval=t,rtol=1e-11,atol=1e-13)
        q,p,_=forced_observer(sol.y[:1,::-1],x[None,::-1],w,dt=.001,velocity=[v])
        self.assertAlmostEqual(q[0],sol.y[0,-1],places=6)
        self.assertAlmostEqual(p[0],sol.y[1,-1]/w,places=6)
        q2,p2,_=forced_observer(sol.y[:1,::-1],x[None,::-1]+100,w,dt=.001,velocity=[v])
        np.testing.assert_allclose([q,p],[q2,p2],atol=1e-11)

    def test_future_measurements_cannot_change_refined_or_kernel_inputs(self):
        t=np.arange(401)*.01
        scenario=Scenario(1,t,{'theta':.1*np.sin(4*t),'x':.02*np.sin(2*t),
            'xr':np.ones(len(t)),'L':np.full(len(t),.5)},{'family':'trolley'})
        before=command_windows(scenario,LengthCalibration(),[100])
        refined=refine_windows(before)
        self.assertEqual(tuple(tensors(refined)['forced_residual'].shape),(1,41))
        inputs=phase_kernel_features(before,refined,np.zeros((1,30)))
        scenario.signals['theta'][101:]+=50
        scenario.signals['x'][101:]-=50
        after=command_windows(scenario,LengthCalibration(),[100]);updated=refine_windows(after)
        np.testing.assert_array_equal(inputs,phase_kernel_features(after,updated,np.zeros((1,30))))
        for key in ['initial_q','initial_p','angular_context','forced_residual']:
            np.testing.assert_array_equal(refined[key],updated[key])
