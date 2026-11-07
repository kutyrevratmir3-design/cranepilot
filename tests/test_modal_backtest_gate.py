import copy,json,unittest
from pathlib import Path
import numpy as np
from cranepilot.physics import LengthCalibration
from scripts.modal_history import modal
from scripts.modal_reference_challenge import statistics,fit
from scripts.modal_backtest_gate import sliced,backtest,weight,compose

class ModalBacktestGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('configs/modal_history_20261107.json').read_text());self.gate=json.loads(Path('configs/modal_backtest_gate_20261107.json').read_text());self.cal=LengthCalibration();n=5;time=-np.arange(81)*.05;omega=self.cal.omega(.4);q=.03*np.cos(omega*time)+.01*np.sin(omega*time)
        self.p=dict(theta=np.tile(q,(n,1)),length=np.full((n,81),.4),omega=np.full(n,omega),k=np.arange(n),scenario=np.arange(n));self.p['target']=modal(self.p,self.cfg,'fixed')[0];prefix=sliced(self.p,0,61,self.cal);prefix['target']=self.p['target'];self.model=fit(statistics(prefix,self.cfg,'difference','direct'),.01)
    def test_exact_prefix_and_observed_indices(self):
        prefix=sliced(self.p,20,None,self.cal);np.testing.assert_array_equal(prefix['theta'],self.p['theta'][:,20:]);self.assertEqual(prefix['theta'].shape[1],61);self.assertNotIn('target',prefix)
        e=backtest(self.p,self.cfg,self.gate,self.cal,self.model);np.testing.assert_array_equal(e['observed'],self.p['theta'][:,[18,16,13,10,5,0]]);self.assertLess(e['modal_backtest_mae'].max(),1e-5)
    def test_future_targets_are_ignored(self):
        a=backtest(self.p,self.cfg,self.gate,self.cal,self.model);p=copy.deepcopy(self.p);p['target'][:]=999.;b=backtest(p,self.cfg,self.gate,self.cal,self.model)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    def test_no_backtest_head_beyond_current_time(self):
        gate=copy.deepcopy(self.gate);gate['backtest_horizons_raw']=[150]
        with self.assertRaises(ValueError):backtest(self.p,self.cfg,gate,self.cal,self.model)
    def test_weights_bounded_and_exact_endpoints(self):
        e=dict(modal_backtest_mae=np.array([0.,2.,1.]),arx_backtest_mae=np.array([2.,0.,1.]));ym=np.arange(21).reshape(3,7);yr=-ym
        for variant in self.gate['variants']:
            w=weight(e,variant);self.assertTrue(((w>=0)&(w<=1)).all());y,_=compose(ym,yr,e,variant);np.testing.assert_allclose(y,w[:,None]*ym+(1-w[:,None])*yr)
        np.testing.assert_array_equal(compose(ym,yr,e,'reference')[0],yr);np.testing.assert_array_equal(compose(ym,yr,e,'modal')[0],ym)

if __name__=='__main__':unittest.main()
