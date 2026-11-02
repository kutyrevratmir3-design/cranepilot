import unittest
import torch
from cranepilot.innovation import InnovationForecaster


class InnovationTests(unittest.TestCase):
    def batch(self):
        return {'angular_context':torch.randn(3,125)*.1,
                'initial_q':torch.tensor([.1,-.05,.02]),
                'initial_p':torch.tensor([.02,.01,-.03]),
                'omega':torch.tensor([3.,4.,5.]),'future_u':torch.ones(3,30)}

    def test_targets_never_change_forecast(self):
        torch.manual_seed(13)
        m=InnovationForecaster(True).eval();b=self.batch();a=torch.zeros(3,30)
        y,_=m(b,a)
        b.update(target_q=torch.randn(3,30)*100,target_x=torch.randn(3,30)*100)
        z,_=m(b,a)
        torch.testing.assert_close(y,z,rtol=0,atol=0)

    def test_learned_port_enters_storage_balance_and_gradients(self):
        torch.manual_seed(1)
        m=InnovationForecaster(True)
        with torch.no_grad():m.force_output.bias.fill_(.7)
        y,d=m(self.batch(),torch.zeros(3,30))
        self.assertGreater(float(d['innovation_force'].detach().abs().max()),.01)
        self.assertLess(float(d['balance'].detach().abs().max()),1e-7)
        y.square().mean().backward()
        grad=m.force_output.bias.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().max()),1e-8)

    def test_observer_frequency_bound_under_extreme_adjustments(self):
        m=InnovationForecaster(False)
        with torch.no_grad():m.observer.bias.fill_(100)
        _,d=m(self.batch(),torch.zeros(3,30))
        self.assertTrue(((d['frequency_ratio']>=.85-1e-6)&(d['frequency_ratio']<=1.15+1e-6)).all())
        self.assertEqual(float(d['innovation_force'].detach().abs().max()),0)

    def test_measurement_anchor_respects_nominal_encoder_bin(self):
        from cranepilot.data import QUANTIZATION_RAD
        m=InnovationForecaster(True,measurement_anchor=True)
        with torch.no_grad():m.observer.bias.fill_(100.)
        b=self.batch();b['observed_q']=b['initial_q']+.01
        with torch.no_grad():_,d=m(b,torch.zeros(3,30))
        self.assertLessEqual(float((d['initial_angle']-b['observed_q']).abs().max()),.5*QUANTIZATION_RAD+1e-8)
