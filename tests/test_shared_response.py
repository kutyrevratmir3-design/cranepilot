import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.external_crane_data import packets
from scripts.shared_response import design,sufficient_statistics,fit,predict,save,load
from scripts.run_matched_adaptation import get_config

class SharedResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads(Path('configs/shared_response_20261107.json').read_text());cls.cfg['oscillator_parameters']=get_config()['oscillator_parameters']
    def packet(self):
        t=np.arange(100)*.05;s=dict(sid=1,time=t,theta=.03*np.sin(3.1*t),radial=np.zeros(len(t)),boom=.1*np.sin(.7*t),velocity=.07*np.cos(.7*t),acceleration=-.049*np.sin(.7*t))
        return packets(s,self.cfg,0.,2)
    def test_causal_convolution_alignment(self):
        p=self.packet();p['future_velocity'][:]=0;p['future_acceleration'][:]=0;p['future_velocity'][:,0]=.2
        d=design(p,self.cfg,4,'harmonic');tail=d[:,-60:-30];self.assertTrue((tail[:,3]==1).all());self.assertEqual(np.count_nonzero(tail),len(tail))
        q=copy.deepcopy(p);q['future_velocity'][:,4:]=999;q['future_acceleration'][:,4:]=-999
        np.testing.assert_array_equal(d,design(q,self.cfg,4,'harmonic'))
    def test_shared_fit_reload_and_target_invariance(self):
        p=self.packet()
        for variant in self.cfg['variants']:
            m=fit(sufficient_statistics(p,self.cfg,variant),1e-6);a=predict(m,p,self.cfg);q=copy.deepcopy(p);q['target'][:]=999
            np.testing.assert_array_equal(a,predict(m,q,self.cfg));q['future_velocity'][:,10:]=999;q['future_acceleration'][:,10:]=-999
            np.testing.assert_array_equal(a[:,:10],predict(m,q,self.cfg)[:,:10])
            with tempfile.TemporaryDirectory() as td:
                file=Path(td)/'model.npz';save(file,m);np.testing.assert_array_equal(a,predict(load(file),p,self.cfg))
    def test_shared_harmonic_represents_known_feature_map(self):
        p=self.packet();cfg=self.cfg;rng=np.random.default_rng(53);width=design(p,cfg,1,'harmonic').shape[1];coefs=rng.normal(0,.00001,width)
        from scripts.external_baselines import oscillator
        p['target']=oscillator(p,cfg,cfg['oscillator_parameters'])+np.stack([design(p,cfg,h+1,'harmonic')@coefs for h in range(30)],1)
        m=fit(sufficient_statistics(p,cfg,'harmonic'),1e-8);y=predict(m,p,cfg)
        self.assertLess(np.abs(y[:,cfg['horizon_index']]-p['target'][:,cfg['horizon_index']]).max(),2e-6)

if __name__=='__main__':unittest.main()
