import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from scripts.actuator_identification import identify,forced_positions,fit_with_parameters
from scripts.adaptive_actuator import AdaptiveModel,prepare_history,solve
from scripts.angular_control import expand_plan
from scripts.run_adaptive_actuator import cases,experiment

class AdaptiveActuatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.cfg=json.loads(Path('configs/adaptive_actuator_20261107.json').read_text())
    def observed(self,n=2,tau=.16,gain=1.2):
        rng=np.random.default_rng(31);c=np.repeat(rng.uniform(-.7,.7,(n,20)),2,axis=1)
        t=np.arange(41)*.05;xf,vf,af=forced_positions(c,tau)
        x=.13+.03*t[None]+.12*tau*(t[None]-tau*(1-np.exp(-t[None]/tau)))+gain*xf
        v=.03+.12*tau*(1-np.exp(-2/tau))+gain*vf;a=.12*np.exp(-2/tau)+gain*af
        return x,c,v,a
    def batch(self,n=2):
        model=AdaptiveModel('nominal');t=-np.arange(41)*.01
        return prepare_history(np.tile(.03*np.cos(5*t),(n,1)),np.tile(.14+.02*t+.01*t*t,(n,1)),np.linspace(.3,.8,n),model.calibration)

    def test_profiled_lag_gain_recovery_without_derivatives(self):
        x,c,_,_=self.observed();f=identify(x,c,self.cfg['identification'])
        self.assertTrue(f['valid'].all());np.testing.assert_allclose(f['tau'],.16,atol=1e-12)
        np.testing.assert_allclose(f['gain'],1.2,atol=.002)
        translated=identify(x+5,c,self.cfg['identification'])
        np.testing.assert_allclose(translated['gain'],f['gain'],atol=1e-10)

    def test_endpoint_state_recovery_with_unknown_initial_state(self):
        x,c,v,a=self.observed();vh,ah,error=fit_with_parameters(x,c,.16,1.2)
        np.testing.assert_allclose(vh,v,atol=1e-12);np.testing.assert_allclose(ah,a,atol=1e-12)
        self.assertLess(error.max(),1e-12)

    def test_no_excitation_and_short_history_keep_prior(self):
        x=np.tile(np.linspace(.1,.3,41),(2,1));c=np.zeros((2,40))
        self.assertFalse(identify(x,c,self.cfg['identification'])['valid'].any())
        self.assertFalse(identify(x[:,:10],c[:,:9],self.cfg['identification'])['valid'].any())
        m=AdaptiveModel('adaptive_observer');f=m.update(x,c,0)
        np.testing.assert_array_equal(f['tau'],.1);np.testing.assert_array_equal(f['gain'],1)
        self.assertFalse(f['observed_valid'].any())
        with self.assertRaises(ValueError):identify(x,c[:,:-1],self.cfg['identification'])

    def test_update_schedule_smoothing_and_estimate_bounds(self):
        x,c,_,_=self.observed();m=AdaptiveModel('adaptive_params');m.update(x,c,0)
        np.testing.assert_allclose(m.tau,.115,atol=1e-12)
        first=m.tau.copy();m.update(x,c,1);np.testing.assert_array_equal(m.tau,first)
        for tick in range(4,100,4):m.update(x,c,tick)
        np.testing.assert_allclose(m.tau,.16,atol=.0001);np.testing.assert_allclose(m.gain,1.2,atol=.002)
        self.assertTrue(((m.gain>=.5)&(m.gain<=1.6)).all())

    def test_factorial_banks_and_observer_only_changes_trolley_state(self):
        b,i=self.batch();x,c,_,_=self.observed();expected={'nominal':1,'force_only':2,'actuator_only':3,'force_actuator':6,'joint_previous':18,'adaptive_force':2}
        for name,count in expected.items():
            m=AdaptiveModel(name);m.update(x,c,0);ctx=m.context(b,i,torch.zeros(2,30))
            self.assertEqual(len(m.scenarios),count)
        m=AdaptiveModel('adaptive_observer')
        for tick in range(0,100,4):m.update(x,c,tick)
        self.assertTrue(m.observed_valid.all());used=m.used_initial(i)
        torch.testing.assert_close(used[:,0],i[:,0],rtol=0,atol=0)
        self.assertFalse(torch.equal(used[:,1:],i[:,1:]))
        ctx=m.context(b,i,torch.zeros(2,30));torch.testing.assert_close(ctx['q'][:,0],b['initial_q'],rtol=0,atol=0)

    def test_affine_frozen_identification_and_independent_cases(self):
        x,c,_,_=self.observed();b,i=self.batch();m=AdaptiveModel('adaptive_force_observer');m.update(x,c,0)
        tau=m.tau.copy();gain=m.gain.copy();committed=torch.zeros(2);warm=torch.zeros(2,6);target=torch.full((2,),.3)
        plan,_,_,d=solve(m,b,i,committed,warm,target,self.cfg['solver'])
        self.assertTrue(torch.all(d['cost']<=d['warm_cost']+.001));self.assertLessEqual(plan.abs().max().item(),.800001)
        one=AdaptiveModel('adaptive_force_observer');one.update(x[:1].copy(),c[:1].copy(),0)
        p=solve(one,{k:v[:1] for k,v in b.items()},i[:1],committed[:1],warm[:1],target[:1],self.cfg['solver'])[0]
        torch.testing.assert_close(plan[:1],p,rtol=1e-4,atol=2e-5)
        np.testing.assert_array_equal(m.tau,tau);np.testing.assert_array_equal(m.gain,gain)
        ctx=m.context(b,i,expand_plan(warm,committed));f0=m.evaluate(ctx,torch.zeros(2,30),target)[1]['features']
        a=m.evaluate(ctx,torch.full((2,30),.4),target)[1]['features'];h=m.evaluate(ctx,torch.full((2,30),.2),target)[1]['features']
        torch.testing.assert_close(h,(a+f0)/2,rtol=2e-5,atol=1e-6)

    def test_short_rollout_history_alignment_and_new_cases(self):
        cfg=copy.deepcopy(self.cfg);cfg['duration_s']=.1;f=cases(cfg,'development').iloc[:1]
        with tempfile.TemporaryDirectory() as td:r=experiment(f,cfg,AdaptiveModel('adaptive_observer'),Path(td))
        self.assertEqual(r['raw_states'].shape,(1,51,5));self.assertEqual(r['id_tau'].shape,(1,2))
        np.testing.assert_array_equal(r['commands'][:,1],r['plans'][:,0,1])
        f=cases(self.cfg,'evaluate');self.assertEqual(len(f),96);self.assertEqual(set(f.length),{.25,.62,1.07})
        self.assertFalse(set(f.length)&{.29,.64,1.11});self.assertEqual(set(f.angle_deg),{3.,5.})

if __name__=='__main__':unittest.main()
