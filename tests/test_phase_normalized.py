import unittest
import numpy as np
import torch
from cranepilot.phase_normalized import PhaseNormalizedForecaster,phase_context,GRUResidualForecaster


class PhaseSimilarityTests(unittest.TestCase):
    def test_dimensional_rescaling_preserves_observation_features(self):
        rng=np.random.default_rng(1)
        raw={'omega':np.array([4.,5.]),'angular_context':rng.normal(size=(2,125)),
             'actuator_context':rng.normal(size=(2,82)),
             'initial_q':np.array([.03,.05]),'initial_p':np.array([.01,-.02])}
        refined={'initial_q':np.array([.02,.04]),'initial_p':np.array([.02,-.01])}
        scaled=dict(raw);scaled['omega']=raw['omega']/2
        scaled['actuator_context']=raw['actuator_context'].copy()
        scaled['actuator_context'][:,:41]*=4
        np.testing.assert_allclose(phase_context(raw,refined,dt=.01),phase_context(scaled,refined,dt=.02),rtol=1e-13,atol=1e-13)

    def test_driven_angular_model_preserves_phase_similarity(self):
        torch.manual_seed(9);model=PhaseNormalizedForecaster(True).double()
        with torch.no_grad():model.force.bias.fill_(.4);model.force.weight.fill_(.1)
        b={'phase_context':torch.randn(4,54,dtype=torch.double)*.2,
           'initial_q':torch.randn(4,dtype=torch.double)*.1,
           'initial_p':torch.randn(4,dtype=torch.double)*.1,
           'omega':torch.tensor([3.,4.,5.,7.],dtype=torch.double)}
        a=torch.randn(4,30,dtype=torch.double)*.2
        with torch.no_grad():
            y,d=model(b,a,dt=.05);scaled=dict(b);scaled['omega']=b['omega']/2
            z,_=model(scaled,a,dt=.1)
        torch.testing.assert_close(y,z,rtol=1e-12,atol=1e-12)
        self.assertGreater(float(d['innovation_force'].abs().max()),.001)
        self.assertLess(float(d['balance'].abs().max()),1e-12)

    def test_insufficient_phase_history_is_rejected(self):
        raw={'omega':np.array([1.]),'angular_context':np.zeros((1,125)),
             'actuator_context':np.zeros((1,82))}
        with self.assertRaises(ValueError):phase_context(raw,{})

    def test_direct_gru_never_reads_future_measured_targets(self):
        torch.manual_seed(12);model=GRUResidualForecaster().eval()
        b={'reference_context':torch.randn(3,128),'omega':torch.ones(3)*4,
           'future_u':torch.ones(3,30),'physical_base':torch.randn(3,30)*.1}
        a=torch.zeros(3,30)
        with torch.no_grad():
            y,_=model(b,a);b['target_q']=torch.ones(3,30)*999;b['target_x']=torch.ones(3,30)*999
            z,_=model(b,a)
        torch.testing.assert_close(y,z,atol=0,rtol=0)
