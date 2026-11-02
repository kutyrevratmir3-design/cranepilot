import unittest
import numpy as np
from scipy.integrate import solve_ivp
from cranepilot.control import PlantParameters, plant_rhs, rk4_step, LinearMPC


class ControlTests(unittest.TestCase):
    def test_integrator_matches_independent_adaptive_solver(self):
        p=PlantParameters(.7,actuator_tau=.12)
        initial=np.array([0.,.1,.2,-.3,.05])
        actual=initial.copy()
        for _ in range(20):
            actual=rk4_step(actual,.3,p,.05,substeps=10)
        expected=solve_ivp(lambda t,x:plant_rhs(x,.3,p),[0,1],initial,rtol=1e-11,atol=1e-12).y[:,-1]
        np.testing.assert_allclose(actual,expected,atol=2e-8)

    def test_stationary_equilibrium(self):
        p=PlantParameters(.7)
        np.testing.assert_array_equal(plant_rhs(np.zeros(5),0,p),np.zeros(5))
        mpc=LinearMPC(.7)
        self.assertAlmostEqual(mpc(np.zeros(5),0),0,places=10)

    def test_command_bound_with_large_initial_error(self):
        mpc=LinearMPC(.7,max_acceleration=.8)
        command=mpc(np.zeros(5),100)
        self.assertLessEqual(abs(command),.8+1e-12)


if __name__=="__main__":
    unittest.main()

