import copy
import unittest

import numpy as np
import torch
from torch import nn

from cranepilot.broad_strategies import make_model
from scripts.stable_latent import make_stable_latent,fit_stable,stability_audit


def causal_batch(n=8,dtype=torch.float64):
    return dict(angular_context=.1*torch.randn(n,125,dtype=dtype),
                initial_q=.03*torch.randn(n,dtype=dtype),
                initial_p=.02*torch.randn(n,dtype=dtype),
                omega=torch.linspace(3.,9.,n,dtype=dtype),
                future_u=.1*torch.randn(n,30,dtype=dtype),
                acceleration=.2*torch.randn(n,30,dtype=dtype),
                observer_residual_rms=torch.full((n,),.002,dtype=dtype))


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__();self.gain=nn.Parameter(torch.tensor(.1))

    def forward(self,batch,acceleration):
        pred=batch['initial_q'][:,None]+self.gain*acceleration
        return pred,{'initial_memory':pred[:,:1],'innovation_force':pred}


class StableLatentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(771)

    def test_arbitrary_learned_generator_has_analytic_cayley_contraction_identity(self):
        for dim in [6,12]:
            model=make_stable_latent({'mode':'standalone','latent_dim':dim}).double()
            with torch.no_grad():
                model.skew_raw.normal_(0,7)
                model.damping_raw.uniform_(-10,3)
            omega=torch.tensor([.01,.3,3.,10.,70.],dtype=torch.float64)
            transition,_=model.discrete_operators(omega)
            generator=model.generator();damping=torch.nn.functional.softplus(model.damping_raw)+1e-3
            identity=torch.eye(dim,dtype=torch.float64)[None]
            left=identity-.5*.05*omega[:,None,None]*generator
            inverse=torch.linalg.inv(left)
            observed=identity-transition.transpose(1,2)@transition
            expected=inverse.transpose(1,2)@(2*.05*omega[:,None,None]*torch.diag(damping))@inverse
            torch.testing.assert_close(observed,expected,atol=2e-14,rtol=2e-10)
            self.assertTrue((torch.linalg.eigvalsh(observed)>0).all())
            self.assertTrue((torch.linalg.svdvals(transition)[:,0]<1).all())
            # This is stronger than testing only the eigenvalue spectral radius.
            states=torch.randn(len(omega),dim,1,dtype=torch.float64)
            previous=torch.linalg.vector_norm(states,dim=1)
            for _ in range(100):
                states=transition@states
                current=torch.linalg.vector_norm(states,dim=1)
                self.assertTrue((current<=previous+1e-14).all());previous=current

    def test_initial_residual_matches_real_frozen_base_bit_for_bit(self):
        base=make_model({'symmetry_mix':.5}).double().eval()
        model=make_stable_latent('residual6',base)
        model.double();batch=causal_batch()
        with torch.no_grad():
            expected=base(batch,batch['acceleration'])[0]
            prediction,diag=model(batch,batch['acceleration'])
        self.assertTrue(torch.equal(expected,prediction))
        self.assertEqual(torch.count_nonzero(diag['latent_output']),0)
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        model.train();self.assertFalse(model.base.training)

    def test_dynamics_are_global_and_omega_scaled(self):
        model=make_stable_latent('standalone6').double()
        low=model.discrete_operators(torch.tensor([4.],dtype=torch.float64),.05)
        equivalent=model.discrete_operators(torch.tensor([8.],dtype=torch.float64),.025)
        for a,b in zip(low,equivalent):self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):model.discrete_operators(torch.tensor([0.]))
        with self.assertRaises(ValueError):model.discrete_operators(torch.tensor([float('nan')]))
        with self.assertRaises(ValueError):make_stable_latent('residual6')
        with self.assertRaises(ValueError):make_stable_latent('standalone6',TinyBase())

    def test_late_inputs_cannot_change_prediction_prefix_and_targets_are_ignored(self):
        for choice in ['standalone12','residual6']:
            model=make_stable_latent(choice,TinyBase() if choice.startswith('residual') else None).double()
            with torch.no_grad():model.readout_delta.normal_(0,.1);model.input_raw.normal_()
            original=causal_batch();changed={k:v.clone() for k,v in original.items()}
            changed['future_u'][:,17:]+=11;changed['acceleration'][:,17:]-=13
            changed['target_q']=torch.full((8,30),float('nan'),dtype=torch.float64)
            first=model(original,original['acceleration'])[0]
            second=model(changed,changed['acceleration'])[0]
            self.assertTrue(torch.equal(first[:,:17],second[:,:17]))
            self.assertFalse(torch.equal(first[:,17:],second[:,17:]))

    def test_learned_input_discretization_matches_direct_implicit_midpoint(self):
        model=make_stable_latent('standalone6').double()
        with torch.no_grad():model.input_raw.normal_();model.skew_raw.normal_()
        batch=causal_batch();states,initial=model.latent_rollout(batch,batch['acceleration'])
        generator=model.generator();forcing=model.input_anchor+.1*model.input_raw
        z=initial
        for k in range(30):
            u=torch.stack([batch['acceleration'][:,k]/9.81,batch['future_u'][:,k]/2],dim=1)
            delta=states[:,k]-z
            rhs=batch['omega'][:,None]*.05*(.5*(states[:,k]+z)@generator.T+u@forcing.T)
            torch.testing.assert_close(delta,rhs,atol=2e-15,rtol=1e-10)
            z=states[:,k]

    def test_training_four_variants_reload_and_freeze_base(self):
        batch=causal_batch(dtype=torch.float32);active=np.arange(8)%2==0;scenario=np.arange(8)%3
        config=dict(epochs=3,patience=2,batch_size=4,learning_rate=.001,weight_decay=.0001)
        for choice in ['standalone6','standalone12','residual6','residual6_phase']:
            model=make_stable_latent(choice,TinyBase() if choice.startswith('residual') else None)
            initial_base=copy.deepcopy(model.base.state_dict()) if model.base is not None else None
            with torch.no_grad():target=model(batch,batch['acceleration'])[0]+.002
            state,log,meta=fit_stable(model,batch,target,active,scenario,batch,target.numpy(),active,config,choice,6)
            replica=make_stable_latent(choice,TinyBase() if choice.startswith('residual') else None)
            replica.load_state_dict(state)
            self.assertTrue(torch.equal(model(batch,batch['acceleration'])[0],replica(batch,batch['acceleration'])[0]))
            self.assertEqual(meta['optimizer_steps'],meta['backward_passes'])
            self.assertEqual(len(log),meta['epochs_executed']+1)
            self.assertTrue(np.isfinite(meta['selection_score']))
            self.assertLess(meta['stability_train_omega']['max_transition_singular_value'],1)
            if initial_base is not None:
                for key,value in initial_base.items():self.assertTrue(torch.equal(value,model.base.state_dict()[key]))
                self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_epoch_minus_one_rollback_is_exact_when_validation_starts_perfect(self):
        model=make_stable_latent('residual6',TinyBase()).double();batch=causal_batch()
        before=copy.deepcopy(model.state_dict())
        with torch.no_grad():target=model(batch,batch['acceleration'])[0]
        active=np.arange(8)%2==0
        config=dict(epochs=2,patience=2,batch_size=4,learning_rate=.001,weight_decay=.0001)
        state,_,meta=fit_stable(model,batch,target+.01,active,np.ones(8),batch,target,active,config,'residual6',7)
        self.assertEqual(meta['selected_epoch'],-1)
        for key,value in before.items():self.assertTrue(torch.equal(value,state[key]))

    def test_input_gradients_survive_frozen_parameters(self):
        model=make_stable_latent('residual6',TinyBase()).double();batch=causal_batch()
        acceleration=batch['acceleration'].clone().requires_grad_()
        model(batch,acceleration)[0].square().mean().backward()
        self.assertIsNotNone(acceleration.grad)
        self.assertGreater(float(acceleration.grad.abs().sum()),0)
        self.assertIsNone(model.base.gain.grad)

    def test_extra_latent_dimensions_have_a_gradient_path_from_first_step(self):
        for choice in ['standalone12','residual6']:
            model=make_stable_latent(choice,TinyBase() if choice.startswith('residual') else None).double()
            batch=causal_batch();prediction,_=model(batch,batch['acceleration'])
            target=prediction.detach()+torch.linspace(.01,.03,30,dtype=torch.float64)[None]
            (prediction-target).square().mean().backward()
            self.assertGreater(float(model.readout_delta.grad[2:].abs().sum()),1e-10)


if __name__=='__main__':unittest.main()
