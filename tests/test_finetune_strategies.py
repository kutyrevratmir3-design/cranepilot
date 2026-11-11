import copy
import unittest
import numpy as np
import torch
from torch import nn
from cranepilot.broad_strategies import make_model
from scripts.finetune_strategies import (STRATEGIES, configure_trainable, anchor_penalty,
    phase_space_terms, GroupDRO, update_ema, sam_step, frozen_teacher_predictions, fit_strategy)


class TinyForecast(nn.Module):
    """Cheap differentiable surrogate used only to exercise training mechanics."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(2,2); self.initial=nn.Linear(2,2); self.observer=nn.Linear(2,2)
        self.hidden_initial=nn.Linear(2,2); self.innovation=nn.Linear(2,2); self.force_output=nn.Linear(2,1)
        self.cell=nn.Linear(2,2)

    def forward(self,batch,acceleration):
        h=torch.tanh(self.encoder(batch['angular_context']))
        memory=self.initial(h); observed=self.observer(h)
        latent=self.innovation(torch.tanh(self.hidden_initial(h)))+self.cell(h)
        force=self.force_output(latent)
        slope=torch.arange(1,31,dtype=h.dtype,device=h.device)[None]/30
        prediction=.02*(memory[:,0,None]+observed[:,0,None]+force*slope)+.01*acceleration
        return prediction,{'initial_memory':memory,'innovation_force':force.expand(-1,30)}


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(193)

    def test_partial_freezing_updates_only_declared_original_modules(self):
        for strategy,prefixes in [('observer_only',('initial.','observer.')),
                                  ('innovation_only',('hidden_initial.','innovation.','force_output.'))]:
            model=make_model({'symmetry_mix':.5}).double()
            names=configure_trainable(model,strategy)
            self.assertEqual(set(names),{n for n,_ in model.named_parameters() if n.startswith(prefixes)})
            initial={n:p.detach().clone() for n,p in model.named_parameters()}
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
            # A nonzero derivative even at the zero-initialized output heads.
            sum((p+1).square().sum() for p in model.parameters() if p.requires_grad).backward();opt.step()
            for n,p in model.named_parameters():
                if n in names:self.assertFalse(torch.equal(p,initial[n]),n)
                else:torch.testing.assert_close(p,initial[n],rtol=0,atol=0)
            configure_trainable(model,'continued')
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_parameter_anchor_is_exact_zero_then_sum_of_displacements(self):
        model=nn.Linear(2,1).double();anchor={n:p.detach().clone() for n,p in model.named_parameters()}
        self.assertEqual(float(anchor_penalty(model,anchor).detach()),0.)
        with torch.no_grad():model.weight.add_(.25)
        self.assertAlmostEqual(float(anchor_penalty(model,anchor).detach()),2*.25**2,places=14)
        anchor_penalty(model,anchor).backward()
        torch.testing.assert_close(model.weight.grad,torch.full_like(model.weight,.5))

    def test_teacher_is_detached_and_does_not_track_student_updates(self):
        model=TinyForecast().double();batch=self.batch(6)
        teacher=frozen_teacher_predictions(model,batch,chunk=2)
        retained=teacher.clone()
        self.assertFalse(teacher.requires_grad)
        with torch.no_grad():model.observer.bias.add_(.5)
        torch.testing.assert_close(teacher,retained,rtol=0,atol=0)
        self.assertFalse(torch.equal(teacher,frozen_teacher_predictions(model,batch)))

    def test_phase_surrogate_zero_exact_and_sensitive_to_rate(self):
        target=torch.randn(3,30,dtype=torch.double);omega=torch.tensor([3.,4.,5.],dtype=torch.double)
        a,b=phase_space_terms(target,target,omega)
        self.assertEqual(float(a),0.);self.assertEqual(float(b),0.)
        constant=target+.01
        rate,_=phase_space_terms(constant,target,omega)
        self.assertLess(float(rate),1e-25)
        ramp=target+torch.linspace(0,.01,30,dtype=torch.double)
        rate,_=phase_space_terms(ramp,target,omega)
        self.assertGreater(float(rate),0.)

    def test_group_risk_uses_means_and_only_observed_group_updates(self):
        dro=GroupDRO(3,eta=.1,dtype=torch.double)
        losses=torch.tensor([1.,1.,3.],dtype=torch.double,requires_grad=True)
        ids=torch.tensor([0,0,1],dtype=torch.long)
        result=dro.loss(losses,ids)
        expected=torch.softmax(torch.tensor([.1,.3],dtype=torch.double),dim=0)@torch.tensor([1.,3.],dtype=torch.double)
        torch.testing.assert_close(result,expected)
        self.assertGreater(float(dro.weights[1]),float(dro.weights[0]))
        # Groups1 and2 are absent: their log-odds cannot change.
        previous=float(dro.log_weights[1]-dro.log_weights[2])
        dro.loss(torch.tensor([5.],dtype=torch.double),torch.tensor([0]))
        self.assertAlmostEqual(float(dro.log_weights[1]-dro.log_weights[2]),previous,places=14)
        result.backward();self.assertTrue(torch.isfinite(losses.grad).all())
        duplicated=GroupDRO(3,eta=0,dtype=torch.double)
        x=duplicated.loss(torch.tensor([1.,3.],dtype=torch.double),torch.tensor([0,1]))
        y=duplicated.loss(torch.tensor([1.,1.,1.,3.],dtype=torch.double),torch.tensor([0,0,0,1]))
        torch.testing.assert_close(x,y)

    def test_sam_restores_base_before_optimizer_and_excludes_missing_gradients(self):
        model=nn.Linear(1,1,bias=True).double()
        with torch.no_grad():model.weight.fill_(1.);model.bias.fill_(2.)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        calls=[]
        def closure():
            calls.append(float(model.weight.detach()))
            loss=model.weight.square().sum()
            return loss,{}
        sam_step(model,optimizer,closure,rho=.2,clip_norm=100.)
        self.assertEqual(len(calls),2);self.assertAlmostEqual(calls[1],1.2,places=10)
        self.assertAlmostEqual(float(model.weight.detach()),1.-.1*2*1.2,places=10)
        self.assertEqual(float(model.bias.detach()),2.)

    def test_sam_restores_exactly_on_perturbed_closure_failure(self):
        model=nn.Linear(2,1).double();initial=copy.deepcopy(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.1);calls=0
        def closure():
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('deliberate test failure')
            return sum(p.square().sum() for p in model.parameters()),{}
        with self.assertRaisesRegex(RuntimeError,'deliberate'):sam_step(model,optimizer,closure)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,initial[key],rtol=0,atol=0)
        self.assertFalse(optimizer.state)

    def test_ema_averages_floats_and_copies_integer_buffers(self):
        old={'w':torch.tensor([1.,2.]),'counter':torch.tensor(2,dtype=torch.long)}
        current={'w':torch.tensor([5.,6.],requires_grad=True),'counter':torch.tensor(7,dtype=torch.long)}
        update_ema(old,current,.75)
        torch.testing.assert_close(old['w'],torch.tensor([2.,3.]))
        self.assertEqual(int(old['counter']),7);self.assertFalse(old['w'].requires_grad)

    def batch(self,n):
        return {'angular_context':torch.randn(n,2,dtype=torch.double),
                'omega':torch.linspace(3,7,n,dtype=torch.double),
                'acceleration':torch.randn(n,30,dtype=torch.double)*.1}

    def test_every_strategy_returns_reloadable_checkpoint_and_actual_counts(self):
        config={'epochs':1,'patience':1,'batch_size':4,'learning_rate':.01,'weight_decay':.0001,
                'lbfgs_epochs':1,'lbfgs_patience':1}
        base=TinyForecast().double();train=self.batch(8);validation=self.batch(4)
        with torch.no_grad():
            target=base(train,train['acceleration'])[0]+.01
            valtarget=base(validation,validation['acceleration'])[0]+.01
        active=torch.tensor([False,True]*4);valactive=torch.tensor([False,True]*2)
        scenario=np.array([1,1,1,1,2,2,2,2])
        for strategy in STRATEGIES:
            model=copy.deepcopy(base)
            selected,logs,meta=fit_strategy(model,train,target,active,scenario,validation,valtarget,valactive,config,strategy,7)
            self.assertEqual(logs[0]['epoch'],-1);self.assertEqual(len(logs),2)
            self.assertTrue(meta['epoch_minus_one_eligible']);self.assertFalse(model.training)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,selected[key],rtol=0,atol=0)
            self.assertEqual(meta['optimizer_steps'],1 if strategy=='lbfgs_polish' else 2)
            if strategy=='sam':self.assertEqual(meta['backward_passes'],4)
            elif strategy=='lbfgs_polish':self.assertGreaterEqual(meta['backward_passes'],1)
            else:self.assertEqual(meta['backward_passes'],2)
            self.assertTrue(np.isfinite(meta['selection_score']))
            for value in selected.values():self.assertTrue(torch.isfinite(value).all())
            for name,p in model.named_parameters():
                if not p.requires_grad:torch.testing.assert_close(p,dict(base.named_parameters())[name],rtol=0,atol=0)

    def test_validation_regression_restores_epoch_minus_one_after_actual_updates(self):
        class ScalarForecast(nn.Module):
            def __init__(self):
                super().__init__();self.observer=nn.Linear(1,1,bias=False).double()
                with torch.no_grad():self.observer.weight.fill_(1.)
            def forward(self,batch,acceleration):
                y=self.observer(batch['angular_context']).expand(-1,30)
                return y,{'initial_memory':torch.zeros_like(y),'innovation_force':torch.zeros_like(y)}
        model=ScalarForecast();initial=copy.deepcopy(model.state_dict())
        batch={'angular_context':torch.ones(4,1,dtype=torch.double),'omega':torch.ones(4,dtype=torch.double)*4,
               'acceleration':torch.zeros(4,30,dtype=torch.double)}
        train=torch.full((4,30),2.,dtype=torch.double);validation=torch.zeros_like(train)
        active=torch.tensor([False,True,False,True]);scenario=np.array([1,1,2,2])
        cfg={'epochs':4,'patience':2,'batch_size':4,'learning_rate':.1,'weight_decay':0.}
        selected,logs,meta=fit_strategy(model,batch,train,active,scenario,batch,validation,active,cfg,'continued',0)
        self.assertEqual(meta['best_epoch'],-1);self.assertEqual(meta['epochs_executed'],2)
        self.assertEqual(meta['optimizer_steps'],2)
        self.assertTrue(all(row['selection_score']>1 for row in logs[1:]))
        for key,value in selected.items():torch.testing.assert_close(value,initial[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
