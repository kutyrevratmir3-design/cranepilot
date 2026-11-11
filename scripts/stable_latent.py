"""Global forced latent dynamics with a contractive homogeneous transition.

This is an input/output identification model inspired by stable lifted linear
predictors, not an exact Koopman embedding.  For G=J-D, J=-J.T and D>0, let
M=I-hG/2 and A=M^{-1}(I+hG/2).  Then

    I-A.T A = M^{-T} (2 h D) M^{-1} > 0, h=omega*dt>0.

Thus the *unforced latent recurrence* contracts in Euclidean norm for every
finite parameter vector and fixed positive omega.  This does not certify a
plant, an actor, a controller, or the frozen base in the additive variant.
Known command and acceleration forcing is unrestricted, and can add energy.

The skew-minus-damping idea is related to Sakib/Pan, arXiv:2408.06607v6,
eqs.13-15, but that paper uses an exponential and a similarity transform.  Our
Cayley discretization, physical anchors and small history encoder differ.
No Koopman-invariance or reconstruction loss is imposed, so the assumptions
of Fan et al., arXiv:2401.08153v2, are not established here.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.finetune_strategies import _criterion, _validation, phase_space_terms


def _choice(choice):
    choice = {'strategy':choice} if isinstance(choice,str) else dict(choice)
    strategy = choice.get('strategy','')
    defaults = {
        'standalone6':('standalone',6,False),
        'standalone12':('standalone',12,False),
        'residual6':('residual',6,False),
        'residual6_phase':('residual',6,True),
    }
    mode,dim,phase = defaults.get(strategy,('standalone',6,False))
    return dict(mode=choice.get('mode',mode),latent_dim=int(choice.get('latent_dim',dim)),
                phase_space=bool(choice.get('phase_space',phase)))


class StableLatentForecaster(nn.Module):
    """One TRAIN-learned global generator; length enters through omega only.

    ``angular_context`` is the existing 125-vector from the 41 past samples.
    No future command enters the initial-state encoder.  Output k consumes only
    prescribed inputs 0..k.  The residual mode adds this model's scalar readout
    to a frozen base; its readout is zero-initialized for exact base fallback.
    """
    def __init__(self, choice, base=None):
        super().__init__()
        self.choice = _choice(choice)
        self.mode = self.choice['mode']
        dim = self.choice['latent_dim']
        if self.mode not in ('standalone','residual') or dim not in (6,12):
            raise ValueError('Stable latent mode must be standalone/residual with dimension 6 or 12')
        if (self.mode == 'residual') != (base is not None):
            raise ValueError('A frozen base is required exactly for residual mode')
        self.latent_dim = dim
        self.base = base
        if base is not None:
            base.eval()
            for parameter in base.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        self.encoder = nn.Sequential(nn.Linear(125,32),nn.Tanh(),nn.Linear(32,dim))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        # Nonzero hidden observables keep the extra dimensions learnable.  If
        # encoder, input and readout were all zero on an isolated latent block,
        # every gradient into that block would remain zero indefinitely.
        # The initial q/p corrections and all extra readout weights remain zero,
        # preserving the near-physical standalone / exact-base residual output.
        with torch.no_grad():self.encoder[-1].weight[2:].normal_(0,.01)
        self.skew_raw = nn.Parameter(torch.zeros(dim,dim))
        self.damping_raw = nn.Parameter(torch.full((dim,),math.log(math.expm1(.01))))
        self.input_raw = nn.Parameter(torch.zeros(dim,2))
        self.readout_delta = nn.Parameter(torch.zeros(dim))
        skew = torch.zeros(dim,dim)
        for pair,frequency in enumerate((1.,.5,1.5,2.,.25,3.)[:dim//2]):
            skew[2*pair,2*pair+1] = frequency
            skew[2*pair+1,2*pair] = -frequency
        input_anchor = torch.zeros(dim,2);input_anchor[1,0] = -1.
        readout_anchor = torch.zeros(dim)
        if self.mode == 'standalone':readout_anchor[0] = 1.
        initial_scale = torch.full((dim,),.05/math.sqrt(dim-2))
        initial_scale[0] = .02;initial_scale[1] = .2
        self.register_buffer('skew_anchor',skew)
        self.register_buffer('input_anchor',input_anchor)
        self.register_buffer('readout_anchor',readout_anchor)
        self.register_buffer('initial_scale',initial_scale)

    def train(self, mode=True):
        super().train(mode)
        if self.base is not None:self.base.eval()
        return self

    def generator(self):
        damping = F.softplus(self.damping_raw)+1e-3
        skew = self.skew_anchor+.1*(self.skew_raw-self.skew_raw.T)
        return skew-torch.diag(damping)

    def discrete_operators(self, omega, dt=.05):
        """Return N×d×d A and N×d×2 B for constant forcing on each step."""
        if omega.ndim != 1 or not len(omega) or not torch.isfinite(omega).all() or torch.any(omega <= 0):
            raise ValueError('Finite positive one-dimensional omega required')
        if not math.isfinite(dt) or dt <= 0:raise ValueError('Positive finite timestep required')
        phase = omega[:,None,None]*dt
        generator = self.generator()[None]
        eye = torch.eye(self.latent_dim,dtype=omega.dtype,device=omega.device)[None]
        left = eye-.5*phase*generator
        # A single solve shares the factorization between transition and input.
        forcing = self.input_anchor+.1*self.input_raw
        solved = torch.linalg.solve(left,torch.cat([eye+.5*phase*generator,phase*forcing[None]],dim=2))
        return solved[:,:,:self.latent_dim],solved[:,:,self.latent_dim:]

    def initial_state(self, batch):
        context = batch['angular_context']
        n = len(context)
        if context.shape != (n,125):raise ValueError('Expected the fixed causal N by 125 context')
        initial = torch.stack([batch['initial_q'],batch['initial_p']],dim=1)
        initial = F.pad(initial,(0,self.latent_dim-2))
        return initial+self.initial_scale*torch.tanh(self.encoder(context))

    def latent_rollout(self, batch, acceleration):
        if acceleration.ndim != 2 or acceleration.shape != batch['future_u'].shape or acceleration.shape[1] != 30:
            raise ValueError('Matching N by 30 acceleration and future commands required')
        if len(acceleration) != len(batch['omega']):raise ValueError('Mismatched batch rows')
        state = self.initial_state(batch)
        initial = state
        transition,forcing = self.discrete_operators(batch['omega'])
        controls = torch.stack([acceleration/9.81,batch['future_u']/2.],dim=2)
        inputs = torch.einsum('ndi,nti->ntd',forcing,controls)
        states = []
        for k in range(30):
            state = (transition@state.unsqueeze(2)).squeeze(2)+inputs[:,k]
            states.append(state)
        return torch.stack(states,dim=1),initial

    def forward(self, batch, acceleration):
        states,initial = self.latent_rollout(batch,acceleration)
        readout = self.readout_anchor+self.readout_delta
        learned = states@readout
        if self.base is not None:
            # Frozen parameters still allow input derivatives for later studies;
            # the present campaign does not optimize commands or actor weights.
            prediction,base_diag = self.base(batch,acceleration)
            prediction = prediction+learned
        else:
            prediction = learned
            base_diag = {}
        return prediction,dict(initial_memory=initial[:,2:],
                               innovation_force=torch.zeros_like(prediction),
                               latent_state=states,latent_output=learned,
                               **{'base_'+k:v for k,v in base_diag.items()})


def make_stable_latent(choice, base=None):
    return StableLatentForecaster(choice,base)


@torch.no_grad()
def stability_audit(model, omega):
    """Numerical audit of the learned latent matrices at supplied frequencies."""
    omega = torch.unique(torch.as_tensor(omega,dtype=next(model.parameters()).dtype,
                                        device=next(model.parameters()).device))
    transition,_ = model.discrete_operators(omega)
    generator = model.generator()
    return dict(omega_values=omega.detach().cpu().tolist(),
                max_transition_singular_value=float(torch.linalg.svdvals(transition)[:,0].max()),
                max_transition_spectral_radius=float(torch.linalg.eigvals(transition).abs().max()),
                max_generator_symmetric_eigenvalue=float(torch.linalg.eigvalsh(.5*(generator+generator.T)).max()),
                latent_homogeneous_only=True,plant_or_controller_certificate=False)


def fit_stable(model,trainbatch,target,active,scenario,validationbatch,valtarget,
               valactive,config,choice,seed):
    """Fit on TRAIN, select by validation, and return state, epoch log, metadata.

    Epoch -1 is eligible for both modes.  Residual epoch -1 equals the frozen
    base exactly; standalone epoch -1 is a damped physical initialization.
    There are no inherited memory/forcing penalties.  The optional phase-space
    objective adds the same .25 increment and .25 terminal surrogates as the
    preceding fine-tuning stage, without a closed-loop interpretation.
    """
    parsed = _choice(choice)
    if parsed != model.choice:raise ValueError('Training and model choices differ')
    torch.manual_seed(seed);rng = np.random.default_rng(seed)
    parameter = next(model.parameters());dtype,device = parameter.dtype,parameter.device
    target = torch.as_tensor(target,dtype=dtype,device=device).detach()
    valtarget = torch.as_tensor(valtarget,dtype=dtype,device=device).detach()
    active = torch.as_tensor(active,dtype=torch.bool,device=device)
    valactive = torch.as_tensor(valactive,dtype=torch.bool,device=device)
    trainbatch = {k:torch.as_tensor(v,dtype=dtype,device=device).detach() for k,v in trainbatch.items()}
    validationbatch = {k:torch.as_tensor(v,dtype=dtype,device=device).detach() for k,v in validationbatch.items()}
    n = len(target)
    if not n or target.shape != (n,30) or active.shape != (n,):raise ValueError('Nonempty N by 30 TRAIN targets required')
    if valtarget.shape != (len(valactive),30):raise ValueError('Matching validation targets and masks required')
    if any(len(v) != n for v in trainbatch.values()) or any(len(v) != len(valtarget) for v in validationbatch.values()):
        raise ValueError('Batch rows do not match supervision')
    if np.asarray(scenario).shape != (n,):raise ValueError('One TRAIN scenario per row required')
    if not torch.isfinite(target).all() or not torch.isfinite(valtarget).all():raise ValueError('Finite supervision required')
    epochs,patience,batch_size = [int(config[k]) for k in ['epochs','patience','batch_size']]
    beta = float(config.get('smooth_l1_beta',.1))
    if epochs < 0 or patience < 1 or batch_size < 1 or not math.isfinite(beta) or beta <= 0:
        raise ValueError('Invalid training schedule')
    trainable = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    parameters = [p for _,p in trainable]
    optimizer = torch.optim.AdamW(parameters,lr=float(config['learning_rate']),weight_decay=float(config['weight_decay']))
    started = time.perf_counter()
    starting = _validation(model,validationbatch,valtarget,valactive)
    best_values = starting.copy();best_score = _criterion(starting,starting)
    best_epoch = -1;best_state = copy.deepcopy(model.state_dict())
    logs = [dict(epoch=-1,training_loss=None,selection_score=best_score,best_score=best_score,
                 checkpoint_improved=True,elapsed_s=0.,**{'validation_'+k:v for k,v in starting.items()})]
    optimizer_steps = 0;executed = 0;stale = 0
    for epoch in range(epochs):
        model.train();losses=[];sizes=[];data_losses=[];phase_losses=[]
        for ids_np in np.array_split(rng.permutation(n),int(np.ceil(n/batch_size))):
            ids = torch.as_tensor(ids_np,dtype=torch.long,device=device)
            batch = {k:v[ids] for k,v in trainbatch.items()}
            prediction,_ = model(batch,batch['acceleration'])
            error = (prediction-target[ids])/.1
            data_loss = F.smooth_l1_loss(error,torch.zeros_like(error),beta=beta)
            phase_loss = data_loss.new_zeros(())
            if parsed['phase_space']:
                rate,terminal = phase_space_terms(prediction,target[ids],batch['omega'],beta)
                phase_loss = .25*rate+.25*terminal
            loss = data_loss+phase_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite stable latent training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Nonfinite stable latent gradient')
            torch.nn.utils.clip_grad_norm_(parameters,float(config.get('gradient_clip',2.)));optimizer.step()
            optimizer_steps += 1;sizes.append(len(ids_np))
            losses.append(float(loss.detach()));data_losses.append(float(data_loss.detach()));phase_losses.append(float(phase_loss.detach()))
        values = _validation(model,validationbatch,valtarget,valactive)
        score = _criterion(values,starting)
        if not math.isfinite(score):raise FloatingPointError('Nonfinite stable latent validation criterion')
        improved = score < best_score-1e-8
        if improved:
            best_score,best_epoch,stale = score,epoch,0
            best_values = values.copy();best_state = copy.deepcopy(model.state_dict())
        else:stale += 1
        logs.append(dict(epoch=epoch,training_loss=float(np.average(losses,weights=sizes)),
                         training_data_loss=float(np.average(data_losses,weights=sizes)),
                         training_phase_penalty=float(np.average(phase_losses,weights=sizes)),
                         selection_score=score,best_score=best_score,checkpoint_improved=improved,
                         elapsed_s=time.perf_counter()-started,**{'validation_'+k:v for k,v in values.items()}))
        executed += 1
        if epoch%10 == 0:
            print(f'  stable_{parsed["mode"]}{parsed["latent_dim"]} seed={seed} epoch={epoch} selection={score:.6f} best={best_score:.6f}',flush=True)
        if stale >= patience:break
    model.load_state_dict(best_state);model.eval()
    metadata = dict(strategy='stable_latent',seed=int(seed),choice=parsed,best_epoch=best_epoch,
                    selected_epoch=best_epoch,epochs_executed=executed,optimizer_steps=optimizer_steps,
                    backward_passes=optimizer_steps,optimizer_closure_evaluations=optimizer_steps,
                    fit_seconds=time.perf_counter()-started,selection_score=best_score,validation=best_values,
                    starting_validation=starting,trainable_parameter_names=[name for name,_ in trainable],
                    trainable_parameter_count=sum(p.numel() for _,p in trainable),
                    total_parameter_count=sum(p.numel() for p in model.parameters()),
                    frozen_parameter_count=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                    epoch_minus_one_eligible=True,validation_gradients=False,optimizer='AdamW',
                    regularizers={},phase_space_weights={'increment':.25,'terminal':.25} if parsed['phase_space'] else {},
                    stability_train_omega=stability_audit(model,trainbatch['omega']))
    return {k:v.detach().clone() for k,v in best_state.items()},logs,metadata
