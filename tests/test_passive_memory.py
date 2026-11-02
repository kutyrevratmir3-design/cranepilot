import unittest

import numpy as np
from scipy.integrate import solve_ivp
import torch

from cranepilot.passive_memory import PassiveMemoryCell


class PassiveMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.cell = PassiveMemoryCell().double()

    def test_discrete_storage_identity_and_unforced_contraction(self):
        state = torch.randn(100, 6, dtype=torch.float64)
        damping = torch.rand(100, dtype=torch.float64)
        coupling = torch.randn(100, 4, dtype=torch.float64)
        rates = .01 + torch.rand(100, 4, dtype=torch.float64) * 8
        omega = 2 + torch.rand(100, dtype=torch.float64) * 10
        for acceleration in (torch.zeros(100, dtype=torch.float64),
                             torch.randn(100, dtype=torch.float64)):
            after, report = self.cell.step(state, omega, acceleration, .1,
                                           (damping, coupling, rates))
            self.assertLess(float(report['balance_residual'].abs().max()), 1e-12)
            if torch.count_nonzero(acceleration) == 0:
                self.assertTrue(torch.all(after.square().sum(-1) <= state.square().sum(-1) + 1e-12))

    def test_exact_physical_rotation_without_memory_or_damping(self):
        cell = PassiveMemoryCell(memory_dim=0).double()
        state = torch.tensor([[.1, -.3]], dtype=torch.float64)
        omega = torch.tensor([5.3], dtype=torch.float64)
        coefficients = (torch.zeros(1, dtype=torch.float64),
                        torch.zeros(1, 0, dtype=torch.float64),
                        torch.zeros(1, 0, dtype=torch.float64))
        after, _ = cell.step(state, omega, torch.zeros_like(omega), .17, coefficients)
        expected = [.1*np.cos(5.3*.17)-.3*np.sin(5.3*.17),
                    -.1*np.sin(5.3*.17)-.3*np.cos(5.3*.17)]
        np.testing.assert_allclose(after.numpy()[0], expected, atol=1e-14)

    def test_reference_solver_and_step_convergence(self):
        state = torch.tensor([[.15,-.08,.02,-.01,.04,-.03]], dtype=torch.float64)
        coefficients = (torch.tensor([.08], dtype=torch.float64),
                        torch.tensor([[.4,-.25,.1,.3]], dtype=torch.float64),
                        torch.tensor([[.1,.5,1,2]], dtype=torch.float64))
        matrix = self.cell.closure_matrix(*coefficients).numpy()[0]
        matrix[0,1] += 1
        matrix[1,0] -= 1
        force = np.array([0.,.12,0.,0.,0.,0.])
        reference = solve_ivp(lambda t,z: matrix@z + force, [0,1], state.numpy()[0],
                              rtol=1e-12, atol=1e-14, method='DOP853').y[:,-1]
        errors = []
        for dt in (.1,.05,.025):
            out = state.clone()
            for _ in range(round(1/dt)):
                out, _ = self.cell.step(out, torch.ones(1, dtype=torch.float64),
                                       torch.tensor([-9.81*.12], dtype=torch.float64),dt,coefficients)
            errors.append(float(np.linalg.norm(out.numpy()[0]-reference)))
        self.assertGreater(np.log2(errors[0]/errors[1]), 1.95)
        self.assertGreater(np.log2(errors[1]/errors[2]), 1.95)

    def test_autograd_agrees_with_independent_central_difference(self):
        state = torch.randn(2, 6, dtype=torch.float64) * .1
        context = torch.randn(2, 8, dtype=torch.float64)
        omega = torch.tensor([3.,6.], dtype=torch.float64)
        acceleration = torch.tensor([.3,-.1], dtype=torch.float64)
        parameter = self.cell.coefficients[-1].bias
        def loss():
            result, _ = self.cell.step(state, omega, acceleration, .05,
                                      self.cell.parameters_from_context(context))
            return result[:,0].square().sum()
        loss().backward()
        gradient = float(parameter.grad[1])
        with torch.no_grad():
            original = float(parameter[1]); eps = 1e-5
            parameter[1] = original+eps; plus = float(loss())
            parameter[1] = original-eps; minus = float(loss())
            parameter[1] = original
        self.assertAlmostEqual(gradient, (plus-minus)/(2*eps), delta=1e-8)

    def test_batches_do_not_exchange_states(self):
        state = torch.randn(4, 6, dtype=torch.float64)
        context = torch.randn(4, 8, dtype=torch.float64)
        omega = torch.tensor([3.,4.,5.,6.], dtype=torch.float64)
        u = torch.randn(4, dtype=torch.float64)
        a, _ = self.cell.step(state, omega, u, .05,self.cell.parameters_from_context(context))
        b, _ = self.cell.step(state[2:3],omega[2:3],u[2:3],.05,
                             self.cell.parameters_from_context(context[2:3]))
        torch.testing.assert_close(a[2:3], b)


if __name__ == '__main__':
    unittest.main()
