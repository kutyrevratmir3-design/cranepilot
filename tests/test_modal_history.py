import copy,json,unittest
from pathlib import Path
import numpy as np
from cranepilot.data import Scenario
from cranepilot.physics import LengthCalibration
from scripts.modal_history import modal,local_ar,packets,load_allowed,arx_statistics,fit_arx,predict_arx

class ModalHistoryTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('configs/modal_history_20261107.json').read_text());self.w=5.73;self.d=.04;self.offset=.012;self.time=-np.arange(81)*.05
        y=self.offset+np.exp(-self.d*self.time/2)*(.04*np.cos(self.w*self.time)+.02*np.sin(self.w*self.time));self.p=dict(theta=np.tile(y,(2,1)),omega=np.full(2,self.w),length=np.full((2,81),.3),x=np.zeros((2,81)),xr=np.zeros((2,81)),scenario=np.array([1,2]),k=np.array([500,510]))
    def test_exact_fixed_oscillator_with_offset(self):
        p=copy.deepcopy(self.p);p['theta']=self.offset+.04*np.cos(self.w*self.time)[None,:]+.02*np.sin(self.w*self.time)[None,:];p['theta']=np.repeat(p['theta'],2,axis=0)
        y,_=modal(p,self.cfg,'fixed');t=np.array(self.cfg['horizons_raw'])*.01;truth=self.offset+.04*np.cos(self.w*t)+.02*np.sin(self.w*t)
        np.testing.assert_allclose(y,np.tile(truth,(2,1)),atol=1e-12)
    def test_identifies_frequency_and_damping_from_past(self):
        p=copy.deepcopy(self.p);p['omega']*=1.1;y,d=modal(p,self.cfg,'one_mode');t=np.array(self.cfg['horizons_raw'])*.01;truth=self.offset+np.exp(-self.d*t/2)*(.04*np.cos(self.w*t)+.02*np.sin(self.w*t))
        np.testing.assert_allclose(y,np.tile(truth,(2,1)),atol=3e-5);self.assertLess(np.max(np.abs(d['omega']-self.w)),.002)
    def test_zero_and_low_amplitude_fallback(self):
        p=copy.deepcopy(self.p);p['theta'][:]=.002
        for variant in self.cfg['modal_variants']:
            y,d=modal(p,self.cfg,variant);np.testing.assert_allclose(y,.002,atol=1e-14);self.assertTrue((d['modes']==0).all())
    def test_secondary_mode_requires_past_fit_improvement(self):
        p=copy.deepcopy(self.p);p['omega'][:]=8.;p['theta']=np.tile(.03*np.cos(8*self.time)+.015*np.sin(16*self.time),(2,1))
        one,_=modal(p,self.cfg,'one_mode');two,d=modal(p,self.cfg,'two_mode');t=np.array(self.cfg['horizons_raw'])*.01;truth=.03*np.cos(8*t)+.015*np.sin(16*t)
        self.assertTrue((d['modes']==2).all());self.assertLess(np.abs(two-truth).mean(),np.abs(one-truth).mean()/2)
    def test_local_references_finite_and_targets_ignored(self):
        p=copy.deepcopy(self.p);p['target']=np.full((2,7),999.)
        for order in [2,4,8]:
            a=local_ar(p,self.cfg,order=order);p['target']*=-1;np.testing.assert_array_equal(a,local_ar(p,self.cfg,order=order));self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.isfinite(local_ar(p,self.cfg,tls=True)).all())
    def test_arx_and_modal_ignore_future_targets(self):
        p=copy.deepcopy(self.p);p['target']=modal(p,self.cfg,'fixed')[0]
        m=fit_arx(arx_statistics(p,self.cfg,'lpv_harmonic_residual'),1e-4);a=predict_arx(m,p,self.cfg);b=modal(p,self.cfg,'one_mode')[0];p['target'][:]=999
        np.testing.assert_array_equal(a,predict_arx(m,p,self.cfg));np.testing.assert_array_equal(b,modal(p,self.cfg,'one_mode')[0])
    def test_sealed_loader_rejects_before_io(self):
        with self.assertRaises(ValueError):load_allowed(self.cfg,[9],'train')
        with self.assertRaises(ValueError):load_allowed(self.cfg,[9],'confirmation')
    def test_gate_never_uses_future_measured_motion(self):
        cfg=copy.deepcopy(self.cfg);cfg['origin_start_s']=4;n=701;signals=dict(L=np.full(n,.4),theta=np.sin(np.arange(n)*.01),x=np.zeros(n),xr=np.zeros(n));s=Scenario(1,np.arange(n)*.01,signals,{})
        p,_=packets(s,'theta',cfg,LengthCalibration(),False);first=p['k'][0];signals['x'][first+1:]=1.;signals['L'][first+1:]=2.;signals['theta'][first+1:]=999.
        q,_=packets(s,'theta',cfg,LengthCalibration(),False)
        self.assertEqual(q['k'][0],first)
        for key in p:np.testing.assert_array_equal(p[key][0],q[key][0])

if __name__=='__main__':unittest.main()
