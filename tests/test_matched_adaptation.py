import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from scripts.external_crane_data import packets
from scripts.matched_adaptation import features,fit_linear,linear_predict,save_linear,load_linear,angular_batch,make_angular,AdaptedAngular
from scripts.run_matched_adaptation import get_config

class MatchedAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.cfg=get_config()
    def packet(self):
        t=np.arange(90)*.05;s=dict(sid=1,time=t,theta=.03*np.sin(3.1*t),radial=np.zeros(len(t)),boom=.1*np.sin(.7*t),velocity=.07*np.cos(.7*t),acceleration=-.049*np.sin(.7*t))
        return packets(s,self.cfg,0.,2)
    def test_linear_and_narx_command_prefix_and_reload(self):
        p=self.packet();q=copy.deepcopy(p);q['future_velocity'][:,10:]=999;q['future_acceleration'][:,10:]=-999
        for basis,base in [('linear','none'),('linear','geometric'),('linear','fitted'),('structured','fitted')]:
            m=fit_linear(p,self.cfg,1e-6,basis,base);a=linear_predict(m,p,self.cfg);b=linear_predict(m,q,self.cfg);np.testing.assert_array_equal(a[:,:10],b[:,:10])
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'m.npz';save_linear(path,m);np.testing.assert_array_equal(a,linear_predict(load_linear(path),p,self.cfg))
    def test_neural_adapter_same_history_and_known_reference_alignment(self):
        p=self.packet();b=angular_batch(p,self.cfg);q=copy.deepcopy(p);q['target'][:]=999;q['actual_horizon_s'][:]=-999
        for key,value in b.items():torch.testing.assert_close(value,angular_batch(q,self.cfg)[key],rtol=0,atol=0)
        torch.testing.assert_close(b['future_u'][:,0],torch.as_tensor(2*p['past_velocity'][:,0]/.2,dtype=torch.float32),rtol=0,atol=0)
        self.assertEqual(b['angular_context'].shape[1],125);self.assertTrue(torch.all(b['omega']>0))
    def test_equilibrium_exact_and_gradient_exists(self):
        b=angular_batch(self.packet(),self.cfg);m=make_angular(0,True)
        z={key:(value if key=='omega' else torch.zeros_like(value)) for key,value in b.items()}
        y=m(z,z['acceleration'])[0];self.assertLess(y.detach().abs().max().item(),1e-7)
        loss=m(b,b['acceleration'])[0].square().mean();loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().max()>0 for p in m.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_arx_parameters_do_not_read_target_at_prediction(self):
        p=self.packet();m=fit_linear(p,self.cfg,1e-4,'structured','fitted');a=linear_predict(m,p,self.cfg);p['target'][:]=-999
        np.testing.assert_array_equal(a,linear_predict(m,p,self.cfg))
    def test_strong_reference_and_retrospective_contract(self):
        self.assertEqual(self.cfg['selection_reference'],'strong_arx');self.assertLess(min(self.cfg['ridge_grid']),1e-4)
        self.assertIsNone(self.cfg['fresh_confirmation_source']);self.assertIn('already seen',self.cfg['test_status'])
        self.assertFalse(set(self.cfg['train_ids'])&set(self.cfg['test_ids']))

if __name__=='__main__':unittest.main()
