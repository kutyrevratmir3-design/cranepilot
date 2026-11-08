import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from unittest.mock import patch
from cranepilot.physics import LengthCalibration
from cranepilot.regime_pilot import FrozenRegimePilot
from scripts import latency_observer as lo

class LatencyObserverTests(unittest.TestCase):
    def setUp(self):
        self.task=json.loads(Path('artifacts/modal_bundle_20261107/task.json').read_text());self.cal=LengthCalibration();self.omega=np.full(2,self.cal.omega(.4));self.raw=dict(theta=np.zeros((2,401)),L=np.full((2,401),.4),x=np.zeros((2,401)),xr=np.zeros((2,401)));self.pilot=SimpleNamespace(pilot=SimpleNamespace(cfg=self.task,cases={'trolley_theta':dict(calibration=dict(a=self.cal.a,b=self.cal.b))}),models={'trolley_theta':dict(prior=dict(channels=np.array(['x_support']),scale=np.ones(1),gamma=np.ones(1)))})
    def test_zero_age_is_exact_identity_for_every_adapter(self):
        self.raw['theta'][:]=np.sin(np.arange(401)*.03)
        for method in ['hold','linear_0.10','quadratic_0.25','modal_3s','forced_3s_adapt']:
            aligned=lo.complete_head(self.raw,0,'theta',method,self.pilot,'trolley_theta')
            for key in self.raw:np.testing.assert_array_equal(aligned[key],self.raw[key])
    def test_observed_tail_is_reindexed_without_extrapolation(self):
        self.raw['theta'][:]=np.arange(401)
        for age in [2,5,15]:
            aligned=lo.complete_head(self.raw,age,'theta','hold',self.pilot,'trolley_theta');np.testing.assert_array_equal(aligned['theta'][:,age:],self.raw['theta'][:,:401-age]);np.testing.assert_array_equal(aligned['L'],self.raw['L'])
    def test_linear_completion_recovers_ramp_and_preserves_input(self):
        age=7;t=-np.arange(401)*.01-age*.01;self.raw['theta'][:]=.1+.03*t;before=copy.deepcopy(self.raw);aligned=lo.complete_head(self.raw,age,'theta','linear_0.10',self.pilot,'trolley_theta');np.testing.assert_allclose(aligned['theta'],np.tile(.1-.03*np.arange(401)*.01,(2,1)),atol=1e-13)
        for key in before:np.testing.assert_array_equal(self.raw[key],before[key])
    def test_dated_modal_advances_the_sensor_clock(self):
        ages=np.array([5,10]);t=-np.arange(81)*.05-ages[:,None]*.01;theta=.1*np.cos(self.omega[:,None]*t+.3);y=lo.dated_modal(theta,self.omega,ages,self.task);future=np.array(self.task['horizons_raw'])*.01;truth=.1*np.cos(self.omega[:,None]*future+.3);self.assertLess(np.abs(y-truth).max(),.001)
    def test_forced_head_uses_known_support_gap_without_angle_truth(self):
        age=10;self.raw['x'][:,:age+1]=np.linspace(.01,0,age+1);a=lo.complete_head(self.raw,age,'theta','forced_3s_fixed',self.pilot,'trolley_theta');self.assertGreater(np.abs(a['theta'][:,:age]).max(),1e-4);np.testing.assert_array_equal(a['theta'][:,age:],0)
    def test_mixed_step_force_basis_matches_independent_foh_integration(self):
        w=np.array([4.]);d=np.array([.1]);h=np.linspace(0,.05,11)[None,:,None];gap=np.linspace(.05,.06,6)[None,:,None];F,Y=lo.force_basis(h,gap,w,d,np.array([.4]),np.array([0]));state=np.zeros(4);state[2]=0;values=[]
        for series,dt in [(h[0,:,0],.05),(gap[0,:,0],.01)]:
            for k in range(1,len(series)):
                A=np.zeros((4,4));A[:2,:2]=[[0,1],[-(w[0]**2+d[0]**2/4),-d[0]]];A[1,2]=1;A[2,3]=1;state[2]=series[k-1];state[3]=(series[k]-series[k-1])/dt;state=expm(A*dt)@state
                if dt==.01:values.append(((w[0]**2+d[0]**2/4)*state[0]+d[0]*state[1]-series[k])/.4)
        np.testing.assert_allclose(Y[0,:,0],values,atol=1e-12)
    def test_unknown_unsupported_age_and_future_fields_are_rejected(self):
        for age in [None,-1,16,.5,True]:
            with self.assertRaises(ValueError):lo.validate(self.raw,age,'theta')
        with self.assertRaises(ValueError):lo.validate(dict(self.raw,target=np.zeros((2,7))),5,'theta')
    def test_completed_head_is_excluded_from_past_error_truth(self):
        pilot=FrozenRegimePilot();t=-np.arange(401)*.01-.1;self.raw['theta'][:]=.05*np.cos(self.omega[:,None]*t);errors=[];original=lo.simplex_fit
        def capture(E,prior,strength):errors.append(E.copy());return original(E,prior,strength)
        with patch('scripts.latency_observer.simplex_fit',capture):
            for method in ['hold','linear_0.10']:
                _,diag=lo.predict(self.raw,10,'theta',method,pilot,'trolley_theta');np.testing.assert_array_equal(diag['valid_heads'],[True,True,True,True,True,False])
        self.assertEqual(errors[0].shape[1],5);np.testing.assert_array_equal(errors[0],errors[1])
    def test_sensor_packets_use_real_pre_current_measurements(self):
        from cranepilot.data import Scenario
        from scripts.run_latency_observer import raw_history
        t=np.arange(1000)*.01;s=Scenario(1,t,dict(theta=np.arange(1000.),L=np.full(1000,.4),x=np.zeros(1000),xr=np.zeros(1000)),{});cfg=dict(noise_seed=20261108);a=raw_history(s,np.array([425]),'theta','trolley',15,0,cfg,0);self.assertEqual(a['theta'][0,0],410);self.assertEqual(a['theta'][0,-1],10);s.signals['theta'][426:]=1e9;b=raw_history(s,np.array([425]),'theta','trolley',15,0,cfg,0);np.testing.assert_array_equal(a['theta'],b['theta'])

if __name__=='__main__':unittest.main()
