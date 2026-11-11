import copy,json,unittest
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from cranepilot.physics import LengthCalibration
from scripts.drive_conditioned import transition,transfer_history_future,prepare,predict,balanced_weights,future_positions,io_features

class DriveConditionedTests(unittest.TestCase):
    def setUp(self):
        self.task=json.loads(Path('artifacts/modal_bundle_20261107/task.json').read_text());self.cal=LengthCalibration();n=2;self.p=dict(theta=np.zeros((n,81)),length=np.full((n,81),.4),omega=np.full(n,self.cal.omega(.4)),x=np.zeros((n,81)),xr=np.zeros((n,81)),k=np.arange(n),scenario=np.arange(n))
    def test_foh_transition_matches_independent_matrix_exponential(self):
        w=np.array([3.,7.]);d=np.array([0.,.12]);Ad,B0,B1,lam=transition(w,d,.05)
        for i in range(2):
            E=np.zeros((4,4));E[:2,:2]=[[0,1],[-lam[i],-d[i]]];E[1,2]=1;E[2,3]=1;M=expm(E*.05);np.testing.assert_allclose(Ad[i],M[:2,:2],atol=1e-14);np.testing.assert_allclose(B0[i],M[:2,2],atol=1e-14);np.testing.assert_allclose(B1[i],M[:2,3]/.05,atol=1e-14)
    def test_constant_support_level_is_invariant(self):
        h=np.tile(np.linspace(0,.1,81)[None,:,None],(2,1,1));f=np.full((2,31,1),.1);a,b=transfer_history_future(h,f,self.p['omega'],np.zeros(2),np.full(2,.4),np.array([0]),.05);c,d=transfer_history_future(h+9,f+9,self.p['omega'],np.zeros(2),np.full(2,.4),np.array([0]),.05);np.testing.assert_allclose(a,c,atol=1e-12);np.testing.assert_allclose(b,d,atol=1e-12)
    def test_zero_equilibrium_and_hypothetical_support_sensitivity(self):
        hold=prepare(self.p,self.task);model=dict(channels=np.array(hold['channels']),scale=np.ones(1),gamma=np.ones(1));zero,_=predict(hold,model,'fixed');np.testing.assert_allclose(zero,0.,atol=1e-14);time=np.arange(31)*.05;given=dict(x=np.tile((.02*(time/1.5)**2),(2,1)));changed=prepare(self.p,self.task,given=given);y,_=predict(changed,model,'fixed');self.assertGreater(np.abs(y).max(),1e-4)
    def test_targets_rejected_and_given_path_starts_at_current_state(self):
        p=copy.deepcopy(self.p);p['target']=np.ones((2,7))
        with self.assertRaises(ValueError):prepare(p,self.task)
        with self.assertRaises(ValueError):prepare(self.p,self.task,given=dict(x=np.ones((2,31))))
    def test_record_regime_weights_balance_present_groups(self):
        sid=np.array([1,1,1,1,2,2]);drive=np.array([0,0,0,1,0,1],dtype=bool);w=balanced_weights(sid,drive);np.testing.assert_allclose(w.sum(),len(w));np.testing.assert_allclose(w[sid==1].sum(),w[sid==2].sum());np.testing.assert_allclose(w[(sid==1)&drive].sum(),w[(sid==1)&~drive].sum())
    def test_linear_continuation_is_bounded_and_preserves_unseen_current(self):
        p=copy.deepcopy(self.p);p['x'][:]=np.linspace(.6,.2,81);_,f=future_positions(p,.05,'linear',np.array([[0,.5]]));np.testing.assert_array_equal(f['x'][:,0],p['x'][:,0]);self.assertLessEqual(f['x'].max(),.6);self.assertGreaterEqual(f['x'].min(),0)
    def test_io_features_do_not_encode_coordinate_origins(self):
        p=copy.deepcopy(self.p);p['theta'][:]=np.sin(np.arange(81)*.2);p['x'][:]=np.linspace(.4,.2,81);a=io_features(p,'io_lpv');p['theta']+=.3;p['x']+=2;p['xr']+=7;np.testing.assert_allclose(a,io_features(p,'io_lpv'),atol=1e-12)
    def test_record_packing_keeps_targets_and_regimes_separate(self):
        from cranepilot.data import Scenario
        from scripts.run_drive_conditioned import pack
        time=np.arange(1001)*.01;signals={'theta':np.sin(time),'L':np.full(len(time),.4),'x':time*.01,'xr':np.zeros(len(time))};records={sid:Scenario(sid,time,signals,{}) for sid in [1,2]};qc=dict(origin_start_s=4,origin_stride_raw=25);p,y,labels=pack(records,'theta',self.task,qc,self.cal);self.assertNotIn('target',p);self.assertEqual(len(p['k']),len(y));self.assertEqual(len(y),len(labels['regime']));self.assertEqual(set(p['scenario']),{1,2});self.assertTrue(labels['drive_active'].all())

if __name__=='__main__':unittest.main()
