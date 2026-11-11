import copy,json,unittest
from pathlib import Path
import numpy as np
import torch
from scripts.scenario_control import ScenarioModel,prepare_history,solve
from scripts.angular_control import expand_plan,nominal_rollout,physical_angle,objective,AngularModel
from scripts.run_scenario_control import cases

class ScenarioControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.cfg=json.loads(Path('configs/scenario_control_20261107.json').read_text())
    def batch(self,n=2):
        model=ScenarioModel('nominal');t=-np.arange(41)*.01
        return prepare_history(np.tile(.03*np.cos(5*t),(n,1)),np.tile(.14+.02*t+.01*t*t,(n,1)),np.linspace(.3,.8,n),model.calibration)

    def test_nominal_exactly_matches_prior_physics_objective(self):
        b,i=self.batch();m=ScenarioModel('nominal');commands=torch.full((2,30),.13);target=torch.full((2,),.3)
        ctx=m.context(b,i,commands);value,p=m.evaluate(ctx,commands,target)
        ref,other=objective(AngularModel('physical'),b,i,commands,target,self.cfg['solver'])
        torch.testing.assert_close(value,ref,rtol=2e-6,atol=2e-5)
        torch.testing.assert_close(p['theta'],other['theta'],rtol=1e-5,atol=3e-8)

    def test_scenario_mean_equals_separate_costs_and_preserves_rate(self):
        b,i=self.batch();m=ScenarioModel('joint');c=torch.full((2,30),.1);target=torch.full((2,),.3)
        ctx=m.context(b,i,c);value,_=m.evaluate(ctx,c,target);parts=[]
        self.assertEqual(len(m.scenarios),18)
        rate=ctx['p']*ctx['omega'];expected=(b['initial_p']*b['omega'])[:,None].expand_as(rate)
        torch.testing.assert_close(rate,expected,rtol=1e-6,atol=1e-8)
        for j,scenario in enumerate(m.scenarios):
            one=copy.copy(m);one.scenarios=[scenario]
            local={k:(v[:,j:j+1] if k in ['q','p','omega','force'] else v) for k,v in ctx.items()}
            parts.append(one.evaluate(local,c,target)[0])
        torch.testing.assert_close(value,torch.stack(parts).mean(0),rtol=2e-6,atol=2e-5)

    def test_spread_adds_only_declared_variance(self):
        b,i=self.batch();mean=ScenarioModel('joint');spread=ScenarioModel('spread');c=torch.zeros(2,30);target=torch.full((2,),.3)
        ctx=mean.context(b,i,c);v,p=mean.evaluate(ctx,c,target);w,q=spread.evaluate(ctx,c,target)
        self.assertTrue(torch.all(w>=v-1e-5));torch.testing.assert_close(q['theta'],p['theta'],rtol=0,atol=0)
        self.assertGreater(q['features'].shape[1],p['features'].shape[1])

    def test_hybrid_residual_bounded_frozen_and_physical_jacobian(self):
        b,i=self.batch();m=ScenarioModel('hybrid');m.neural=lambda b,a:physical_angle(b,a)+.1
        c=torch.zeros(2,30);ctx=m.context(b,i,c)
        torch.testing.assert_close(ctx['offset'],torch.full((2,30),.015),rtol=0,atol=0)
        nominal=ScenarioModel('nominal');nctx=nominal.context(b,i,c);target=torch.full((2,),.3)
        a=m.evaluate(ctx,c,target)[1]['theta'];base=nominal.evaluate(nctx,c,target)[1]['theta']
        trial=c.clone();trial[:,5:10]=.2
        difference=m.evaluate(ctx,trial,target)[1]['theta']-a
        ndifference=nominal.evaluate(nctx,trial,target)[1]['theta']-base
        torch.testing.assert_close(difference,ndifference,rtol=1e-5,atol=1e-8)
        torch.testing.assert_close(ctx['offset'],torch.full((2,30),.015),rtol=0,atol=0)

    def test_features_affine_for_all_six_methods(self):
        b,i=self.batch();target=torch.full((2,),.3);committed=torch.zeros(2);z=torch.zeros(2,6)
        for name in self.cfg['methods']:
            m=ScenarioModel(name);ctx=m.context(b,i,expand_plan(z,committed))
            f0=m.evaluate(ctx,expand_plan(z,committed),target)[1]['features']
            z1=z.clone();z1[:,2]=1.;f1=m.evaluate(ctx,expand_plan(z1,committed),target)[1]['features']
            z2=z.clone();z2[:,2]=.17;actual=m.evaluate(ctx,expand_plan(z2,committed),target)[1]['features']
            torch.testing.assert_close(actual,f0+.17*(f1-f0),rtol=2e-5,atol=1e-6)

    def test_solver_independence_bounds_pipeline_and_no_target_access(self):
        b,i=self.batch();target=torch.full((2,),.3);committed=torch.tensor([.1,-.05]);warm=torch.zeros(2,6)
        for name in ['nominal','joint_hybrid']:
            m=ScenarioModel(name);plan,_,_,d=solve(m,b,i,committed,warm,target,self.cfg['solver'])
            self.assertLessEqual(plan.abs().max().item(),.800001)
            torch.testing.assert_close(plan[:,0],committed,rtol=0,atol=0)
            self.assertTrue(torch.all(d['cost']<=d['warm_cost']+.001))
            one=solve(m,{k:v[:1] for k,v in b.items()},i[:1],committed[:1],warm[:1],target[:1],self.cfg['solver'])[0]
            torch.testing.assert_close(plan[:1],one,rtol=2e-4,atol=2e-5)
            poisoned=dict(b,target_q=torch.full((2,30),float('nan')),target_x=torch.full((2,30),999.))
            other=solve(m,poisoned,i,committed,warm,target,self.cfg['solver'])[0]
            torch.testing.assert_close(plan,other,rtol=0,atol=0)

    def test_new_case_grid_disjoint_and_balanced(self):
        f=cases(self.cfg,'evaluate');self.assertEqual(len(f),96)
        self.assertEqual(set(f.length),{.29,.64,1.11});self.assertEqual(set(f.angle_deg),{2.5,5.5})
        self.assertFalse(set(f.length)&{.27,.57,1.03})
        for _,part in f.groupby('length'):self.assertEqual(part.angle_deg.value_counts().to_dict(),{2.5:16,5.5:16})

if __name__=='__main__':unittest.main()
