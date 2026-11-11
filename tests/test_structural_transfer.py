import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from scripts.structural_transfer import TransferModel,parameters,observable,plant_step,delayed_command,target_at,prepare_history,solve
from scripts.adaptive_actuator import AdaptiveModel
from scripts.angular_control import plant_step as old_step
from scripts.actuator_identification import forced_positions
from scripts.run_structural_transfer import cases,experiment,metric_rows

class StructuralTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.cfg=json.loads(Path('configs/structural_transfer_20261107.json').read_text())
    def test_frozen_observer_identical_to_historical_rule(self):
        rng=np.random.default_rng(13);command=rng.uniform(-.7,.7,(2,40));x=forced_positions(command,.16)[0]*1.2+.1
        a=TransferModel('frozen_adaptive_force_observer',self.cfg);b=AdaptiveModel('adaptive_force_observer',self.cfg)
        for tick in range(20):
            ra=a.update(x,command,tick);rb=b.update(x,command,tick)
            for key in rb:np.testing.assert_array_equal(ra[key],rb[key])
        t=-np.arange(41)*.01;batch,initial=prepare_history(np.tile(.03*np.cos(5*t),(2,1)),np.tile(.1+.02*t,(2,1)),np.array([.28,.98]),a.calibration)
        args=(batch,initial,torch.zeros(2),torch.zeros(2,6),torch.full((2,),.3),self.cfg['solver'])
        torch.testing.assert_close(solve(a,*args)[0],solve(b,*args)[0],rtol=0,atol=0)
    def test_matched_observer_gate_without_parameter_excitation(self):
        t=np.arange(41)*.05;x=np.tile(.1+.02*t,(2,1));command=np.zeros((2,40))
        fixed=TransferModel('fixed_observer',self.cfg);adaptive=TransferModel('adaptive_observer',self.cfg);frozen=TransferModel('frozen_adaptive_force_observer',self.cfg)
        a=fixed.update(x,command,0);b=adaptive.update(x,command,0);c=frozen.update(x,command,0)
        self.assertTrue(a['observed_valid'].all());self.assertTrue(b['observed_valid'].all());self.assertFalse(c['observed_valid'].any())
        self.assertFalse(a['accepted'].any());np.testing.assert_array_equal(a['tau'],.1);np.testing.assert_array_equal(a['gain'],1.)
        np.testing.assert_array_equal(a['observed'],b['observed']);np.testing.assert_allclose(a['observed'][:,0],.02,atol=1e-13)
        self.assertFalse(fixed.update(x[:,:10],command[:,:9],1)['observed_valid'].any())
    def test_delay_is_causal_and_uses_issued_queue(self):
        issued=np.array([[1.,2.,999.],[3.,4.,888.]])
        np.testing.assert_array_equal(delayed_command(issued,0,np.array([0,1])),[1,0])
        np.testing.assert_array_equal(delayed_command(issued,1,np.array([0,1])),[2,3])
        changed=issued.copy();changed[:,2]=-7
        np.testing.assert_array_equal(delayed_command(issued,1,np.array([0,1])),delayed_command(changed,1,np.array([0,1])))
    def test_matched_plant_equals_previous_plant(self):
        frame=cases(self.cfg,'evaluate').iloc[:4];m=TransferModel('nominal',self.cfg);p=parameters(frame,m.calibration)
        s=np.tile([.1,.02,.04,-.1,.12,0.],(4,1));c=np.linspace(-.5,.5,4)
        actual=plant_step(s,c,p,np.zeros(4),2,1.,self.cfg)
        expected=old_step(s[:,:5],c,p,np.zeros(4),2)
        np.testing.assert_array_equal(actual[:,:5],expected);np.testing.assert_array_equal(actual[:,5],0.)
    def test_friction_and_cascade_and_switch_are_distinct(self):
        frame=cases(self.cfg,'evaluate').iloc[[0,32,48,64,80]].reset_index(drop=True);m=TransferModel('nominal',self.cfg);p=parameters(frame,m.calibration)
        s=np.tile([.1,.08,.04,0.,.12,0.],(5,1));o=observable(s,p,self.cfg)
        self.assertEqual(o[0,4],.12);self.assertLess(o[2,4],.12);self.assertLess(o[4,4],.12)
        early=plant_step(s,np.full(5,.4),p,np.zeros(5),2,3.49,self.cfg);late=plant_step(s,np.full(5,.4),p,np.zeros(5),2,3.5,self.cfg)
        np.testing.assert_array_equal(early[:3],late[:3]);self.assertFalse(np.array_equal(early[3:],late[3:]));self.assertLess(early[1,4],early[0,4])
    def test_new_grid_pairing_and_target_reversal(self):
        f=cases(self.cfg,'evaluate');self.assertEqual(len(f),96);self.assertEqual(f.context.nunique(),24)
        self.assertEqual(set(f.family),set(self.cfg['families']));self.assertEqual(set(f.length),{.28,.98})
        self.assertTrue((f.groupby('context').size()==4).all());self.assertEqual(target_at(3.95,self.cfg),.3);self.assertEqual(target_at(4.,self.cfg),.15)
        self.assertEqual(len(cases(self.cfg,'development')),6)
    def test_short_run_pipeline_hidden_state_and_metric_target(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(duration_s=.1,reverse_time_s=.05)
        f=cases(cfg,'evaluate').iloc[80:81].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as td:r=experiment(f,cfg,TransferModel('fixed_force_observer',cfg),Path(td))
        self.assertEqual(r['raw_states'].shape,(1,51,6));self.assertEqual(r['states'].shape,(1,3,5))
        np.testing.assert_array_equal(r['commands'][:,1],r['plans'][:,0,1]);np.testing.assert_array_equal(r['targets'],[.3,.15,.15])
        np.testing.assert_array_equal(r['issued_raw'][:,:40],0.)
        row=metric_rows(f,r,cfg,'fixed_force_observer')[0]
        expected=np.trapezoid((r['states'][0,:,0]-np.array([.3,.15,.15]))**2,dx=.05)
        self.assertAlmostEqual(row['tracking_ise_m2_s'],expected,14)
    def test_halfstep_convergence_under_combined_mismatch(self):
        f=cases(self.cfg,'evaluate').iloc[80:84];m=TransferModel('nominal',self.cfg);p=parameters(f,m.calibration)
        a=np.tile([.1,.1,.04,0.,0.,0.],(4,1));b=a.copy()
        for k in range(100):
            a=plant_step(a,np.full(4,.4),p,np.zeros(4),2,k*.01,self.cfg)
            b=plant_step(b,np.full(4,.4),p,np.zeros(4),4,k*.01,self.cfg)
        self.assertLess(np.abs(a-b).max(),1e-6)

if __name__=='__main__':unittest.main()
