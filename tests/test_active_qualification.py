import copy,json,unittest
from pathlib import Path
import numpy as np
from cranepilot.data import Scenario
from cranepilot.physics import LengthCalibration
from cranepilot.reliability_pilot import FrozenReliabilityPilot
from scripts.active_qualification import qualification_packets,frozen_predictions

class ActiveQualificationTests(unittest.TestCase):
    def setUp(self):
        self.task=json.loads(Path('artifacts/modal_bundle_20261107/task.json').read_text());self.cfg=json.loads(Path('configs/active_qualification_20261107.json').read_text());self.cal=LengthCalibration();n=701;t=np.arange(n)*.01;self.s=Scenario(1,t,dict(theta=.03*np.cos(5*t),L=np.full(n,.4),x=np.zeros(n),xr=np.zeros(n)),{})
    def test_history_and_target_indexing(self):
        p,target,labels=qualification_packets(self.s,'theta',self.task,self.cfg,self.cal);k=p['k'][0];self.assertEqual(k,400);np.testing.assert_array_equal(p['theta'][0],self.s.signals['theta'][k-np.arange(81)*5]);np.testing.assert_array_equal(target[0],self.s.signals['theta'][k+np.array(self.task['horizons_raw'])]);self.assertNotIn('target',p)
    def test_regimes_use_past_only(self):
        p,_,labels=qualification_packets(self.s,'theta',self.task,self.cfg,self.cal);k=p['k'][0];s=copy.deepcopy(self.s)
        for key in ['theta','L','x']:s.signals[key][k+1:]+=1
        q,_,other=qualification_packets(s,'theta',self.task,self.cfg,self.cal)
        for key in p:np.testing.assert_array_equal(p[key][0],q[key][0])
        for key in ['regime','drive_active','length_active']:self.assertEqual(labels[key][0],other[key][0])
    def test_four_regimes_and_no_exclusions(self):
        count=None
        for drive,length,expected in [(False,False,0),(True,False,1),(False,True,2),(True,True,3)]:
            s=copy.deepcopy(self.s)
            if drive:s.signals['x']=np.linspace(0,.1,len(s.time))
            if length:s.signals['L']+=np.linspace(0,.1,len(s.time))
            p,_,labels=qualification_packets(s,'theta',self.task,self.cfg,self.cal);count=len(p['k']) if count is None else count;self.assertEqual(len(p['k']),count);self.assertTrue((labels['regime']==expected).all())
    def test_frozen_kernel_rejects_targets_and_preserves_api_gate(self):
        pilot=FrozenReliabilityPilot();cal=LengthCalibration(**pilot.cases['trolley_theta']['calibration']);p,_,_=qualification_packets(self.s,'theta',self.task,self.cfg,cal);p={k:v[:2].copy() for k,v in p.items()};a,_=frozen_predictions(p,pilot,'trolley_theta');p['target']=np.ones((2,7))*999
        with self.assertRaises(ValueError):frozen_predictions(p,pilot,'trolley_theta')
        self.assertTrue(all(np.isfinite(y).all() for y in a.values()));history={key:self.s.signals[key][400-np.arange(401)][None,:] for key in ['theta','L','x','xr']};history['x'][0,-1]=.1;future=dict(xr=np.zeros((1,151)));r=pilot.forecast(history,future,family='trolley',axis='theta',known_plan=True);self.assertFalse(r['eligible'][0]);self.assertTrue(np.isnan(r['forecast_rad']).all())

if __name__=='__main__':unittest.main()
