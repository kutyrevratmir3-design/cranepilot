import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm
from cranepilot.physics import LengthCalibration
from cranepilot.forced_observer import analytic_rollout
from scripts.angular_control import (AngularModel,prepare_history,nominal_rollout,physical_angle,
    objective,solve,expand_plan,plant_step)
from scripts.run_angular_control import cases,parameters,exogenous,experiment
import json

class AngularControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg=json.loads(Path('configs/angular_control_20261107.json').read_text())
        cls.model=AngularModel('candidate')

    def batch(self,n=3):
        t=-np.arange(41)*.01
        theta=np.array([.025*np.cos((4+i)*t)+.01*np.sin((4+i)*t) for i in range(n)])
        x=np.tile(.12+.02*t+.01*t*t,(n,1))
        return prepare_history(theta,x,np.linspace(.3,.9,n),self.model.calibration)

    def test_exact_actuator_integrals(self):
        initial=torch.tensor([[.1,.03,.12]],dtype=torch.double);c=torch.full((1,30),.3,dtype=torch.double)
        x,v,a=nominal_rollout(c,initial)
        A=np.array([[0,1,0,0],[0,0,1,0],[0,0,-10,3],[0,0,0,0]],float)
        reference=expm(A*.05)@np.r_[initial.numpy()[0],1]
        self.assertAlmostEqual(x[0,0].item(),reference[0],13)
        self.assertAlmostEqual(v[0,0].item(),reference[1],13)
        self.assertAlmostEqual(a[0,0].item(),(reference[1]-.03)/.05,13)

    def test_physical_rollout_and_feature_cost_identity(self):
        b,i=self.batch();acc=torch.linspace(-.2,.2,90).reshape(3,30)
        actual=physical_angle(b,acc).numpy()
        ref=analytic_rollout(b['initial_q'].numpy(),b['initial_p'].numpy(),b['omega'].numpy(),acc.numpy())
        np.testing.assert_allclose(actual,ref,atol=2e-8,rtol=1e-5)
        model=AngularModel('physical');c=torch.full((3,30),.1)
        cost,pred=objective(model,b,i,c,torch.full((3,),.3),self.cfg['solver'])
        torch.testing.assert_close(cost,pred['features'].square().sum(1),rtol=3e-6,atol=1e-5)

    def test_candidate_command_jacobian_matches_autograd(self):
        m=copy.deepcopy(self.model)
        for model in m.models:model.double()
        b,i=self.batch(1);b={k:v.double() for k,v in b.items()};i=i.double()
        z=torch.full((1,6),.04,dtype=torch.double,requires_grad=True)
        def angular_loss(z):
            _,_,a=nominal_rollout(expand_plan(z,torch.zeros(1,dtype=torch.double)),i)
            return m(b,a).square().sum()
        loss=angular_loss(z);g=torch.autograd.grad(loss,z)[0]
        for k in [0,3,5]:
            plus=z.detach().clone();minus=plus.clone();plus[0,k]+=1e-5;minus[0,k]-=1e-5
            fd=(angular_loss(plus)-angular_loss(minus))/2e-5
            torch.testing.assert_close(fd,g[0,k],rtol=2e-5,atol=1e-7)
        self.assertTrue(all(p.grad is None for model in m.models for p in model.parameters()))

    def test_solver_bounds_committed_step_batch_independence_and_improvement(self):
        b,i=self.batch(3);m=AngularModel('physical');committed=torch.tensor([.1,-.1,0.])
        warm=torch.zeros(3,6);target=torch.full((3,),.3)
        plan,next_warm,_,d=solve(m,b,i,committed,warm,target,self.cfg['solver'])
        torch.testing.assert_close(plan[:,0],committed,rtol=0,atol=0)
        self.assertLessEqual(plan.abs().max().item(),.8)
        self.assertLessEqual((plan[:,1]-warm[:,0]).abs().max().item(),.200001)
        self.assertTrue(torch.all(d['cost']<=d['warm_cost']+1e-4))
        for j in range(3):
            one=solve(m,{k:v[j:j+1] for k,v in b.items()},i[j:j+1],committed[j:j+1],warm[j:j+1],target[j:j+1],self.cfg['solver'])[0]
            torch.testing.assert_close(one,plan[j:j+1],rtol=1e-4,atol=1e-5)
        self.assertEqual(next_warm.shape,(3,6))

    def test_plant_rk_convergence_and_hidden_gain_changes_truth(self):
        f=cases(self.cfg,'evaluate').iloc[:2].copy();p=parameters(f,self.model.calibration)
        s=np.array([[.1,.03,.05,0.,0.],[.1,.03,.05,0.,0.]])
        c=np.full(2,.4);dist=np.zeros(2)
        a=plant_step(s,c,p,dist,2);b=plant_step(s,c,p,dist,4)
        np.testing.assert_allclose(a,b,atol=2e-7,rtol=1e-6)
        changed={k:v.copy() for k,v in p.items()};changed['gain']*=1.4
        self.assertGreater(np.abs(plant_step(s,c,changed,dist)[:,4]-a[:,4]).max(),.001)

    def test_complete_case_grid_and_paired_inputs(self):
        f=cases(self.cfg,'evaluate');self.assertEqual(len(f),96)
        self.assertEqual(len(f.drop_duplicates()),96)
        for _,group in f.groupby('length'):
            self.assertEqual(group.angle_deg.value_counts().to_dict(),{2.:16,6.:16})
        n,d=exogenous(f,self.cfg);n2,d2=exogenous(f,self.cfg)
        np.testing.assert_array_equal(n,n2);np.testing.assert_array_equal(d,d2)
        self.assertEqual(n.shape,(96,641,2))

    def test_short_closed_loop_uses_pipeline_and_real_warmup(self):
        cfg=copy.deepcopy(self.cfg);cfg['duration_s']=.1;f=cases(cfg,'development').iloc[:1]
        with tempfile.TemporaryDirectory() as td:
            r=experiment(f,cfg,AngularModel('physical'),Path(td))
        self.assertEqual(r['raw_states'].shape,(1,51,5))
        self.assertGreater(np.ptp(r['raw_states'][0,:41,2]),1e-4)
        np.testing.assert_array_equal(r['commands'][:,0],0)
        np.testing.assert_array_equal(r['commands'][:,1],r['plans'][:,0,1])

    def test_future_targets_unused(self):
        b,i=self.batch();a=torch.zeros(3,30)
        with torch.no_grad():
            original=self.model(b,a);poison=dict(b,target_q=torch.full((3,30),float('nan')),target_x=torch.full((3,30),999.))
            torch.testing.assert_close(original,self.model(poison,a),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
