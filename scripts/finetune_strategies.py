"""Matched fine-tuning of retained soft-symmetry models on TRAIN only.

The caller supplies prepared causal batches and a distinct validation record.
All strategies retain the starting checkpoint as epoch -1. Validation only
chooses a checkpoint; it never contributes a gradient or an optimizer state.
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
from torch.nn import functional as F
from cranepilot.rollout import HORIZON_INDEX


STRATEGIES = ('continued', 'observer_only', 'innovation_only', 'l2sp',
              'function_anchor', 'sam', 'ema', 'phase_space', 'group_dro', 'lbfgs_polish')


def configure_trainable(model, strategy):
    """Reset requires_grad explicitly and return the exact trainable names."""
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown fine-tuning strategy: {strategy}')
    prefixes = {'observer_only': ('initial.', 'observer.'),
                'innovation_only': ('hidden_initial.', 'innovation.', 'force_output.')}
    selected = []
    for name, parameter in model.named_parameters():
        enabled = strategy not in prefixes or name.startswith(prefixes[strategy])
        parameter.requires_grad_(enabled)
        parameter.grad = None
        if enabled: selected.append(name)
    if not selected: raise ValueError('Strategy selected no trainable parameters')
    return selected


def anchor_penalty(model, anchor):
    """L2-SP sum, without division by parameter count or parameter magnitude."""
    terms = [(p-anchor[name]).square().sum() for name,p in model.named_parameters() if p.requires_grad]
    if not terms: raise ValueError('No trainable parameters for the anchor')
    return torch.stack(terms).sum()


def per_window_loss(prediction, target, beta=.1):
    return F.smooth_l1_loss((prediction-target)/.1, torch.zeros_like(prediction),
                            beta=beta, reduction='none').mean(1)


def phase_space_terms(prediction, target, omega, beta=.1):
    """Control-inspired angle-rate and terminal error surrogates, not MPC cost."""
    if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape[1] < 2:
        raise ValueError('Matching multi-step predictions and targets required')
    if omega.shape != (len(prediction),) or torch.any(omega <= 0):
        raise ValueError('Positive per-window frequencies required')
    error = prediction-target
    rate = torch.diff(error, dim=1)/(omega[:,None]*.05*.1)
    terminal = error[:,-1]/.1
    return (F.smooth_l1_loss(rate, torch.zeros_like(rate), beta=beta),
            F.smooth_l1_loss(terminal, torch.zeros_like(terminal), beta=beta))


class GroupDRO:
    """Persistent group weights; only observed minibatch groups update log-q.

    Each present group's mean gets its conditional q weight. This deliberately
    avoids treating absent groups as zero risk or weighting groups by their
    number of windows. It is a minibatch group-DRO adaptation.
    """
    def __init__(self, n_groups, eta=.05, *, dtype=torch.float32, device=None):
        if int(n_groups) != n_groups or n_groups < 1 or not np.isfinite(eta) or eta < 0:
            raise ValueError('Positive group count and finite nonnegative eta required')
        self.log_weights = torch.zeros(int(n_groups), dtype=dtype, device=device)
        self.eta = float(eta)

    @property
    def weights(self):
        return torch.softmax(self.log_weights, dim=0)

    def loss(self, window_losses, groups, update=True):
        if window_losses.ndim != 1 or groups.shape != window_losses.shape or not len(groups):
            raise ValueError('Nonempty matching loss/group vectors required')
        if groups.dtype != torch.long or groups.min() < 0 or groups.max() >= len(self.log_weights):
            raise ValueError('Integer group IDs outside the fixed training groups')
        present = torch.unique(groups, sorted=True)
        means = torch.stack([window_losses[groups == group].mean() for group in present])
        if not torch.isfinite(means).all(): raise FloatingPointError('Nonfinite group risk')
        if update:
            with torch.no_grad():
                self.log_weights[present] += self.eta*means.detach()
                # A common shift leaves all pairwise log-odds, including absent
                # groups, intact while bounding the stored exponent arguments.
                self.log_weights -= self.log_weights.max().clone()
        conditional_weights = torch.softmax(self.log_weights[present], dim=0)
        return (conditional_weights.detach()*means).sum()


def update_ema(ema_state, current_state, decay=.95):
    if not np.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError('EMA decay must be in [0,1)')
    if ema_state.keys() != current_state.keys(): raise ValueError('EMA state keys differ')
    with torch.no_grad():
        for name,value in current_state.items():
            if torch.is_floating_point(value) or torch.is_complex(value):
                ema_state[name].mul_(decay).add_(value.detach(), alpha=1-decay)
            else:
                ema_state[name].copy_(value)


def sam_step(model, optimizer, closure, rho=.01, clip_norm=2.):
    """Two backwards, exact restoration, then one AdamW step at base weights.

    ``closure`` returns (loss, detached scalar component dictionary). Frozen and
    missing-gradient parameters are excluded. Restoration also happens when
    the perturbed closure or backward raises, before the exception propagates.
    """
    if not np.isfinite(rho) or rho < 0: raise ValueError('Finite nonnegative SAM radius required')
    optimizer.zero_grad(set_to_none=True)
    loss, components = closure()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite first SAM loss')
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters: raise ValueError('SAM found no parameter gradients')
    if not all(torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('Nonfinite first SAM gradient')
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters]))
    originals = [(p, p.detach().clone()) for p in parameters]
    try:
        with torch.no_grad():
            for p,_ in originals: p.add_(p.grad, alpha=float(rho/(norm+1e-12)))
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss,_ = closure()
        if not torch.isfinite(perturbed_loss): raise FloatingPointError('Nonfinite perturbed SAM loss')
        perturbed_loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite perturbed SAM gradient')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip_norm)
    finally:
        with torch.no_grad():
            for p,original in originals: p.copy_(original)
    optimizer.step()
    return float(loss.detach()), {**components, 'sam_perturbed_loss':float(perturbed_loss.detach())}


@torch.no_grad()
def frozen_teacher_predictions(model, batch, chunk=512):
    """Cache detached starting predictions; no teacher parameters are retained."""
    previous_mode = model.training
    model.eval()
    predictions = []
    try:
        for start in range(0,len(batch['omega']),chunk):
            local = {key:value[start:start+chunk] for key,value in batch.items()}
            predictions.append(model(local,local['acceleration'])[0].detach().clone())
    finally:
        model.train(previous_mode)
    return torch.cat(predictions)


@torch.no_grad()
def _validation(model, batch, target, active):
    prediction = frozen_teacher_predictions(model,batch)
    error = torch.rad2deg(prediction[:,HORIZON_INDEX]-target[:,HORIZON_INDEX]).abs()
    if not active.any() or active.all(): raise ValueError('Both validation regimes required')
    return {'all':float(error.mean()), 'active':float(error[active].mean()),
            'quiet':float(error[~active].mean())}


def _criterion(values, starting):
    return .5*(values['all']/max(starting['all'],1e-12)+values['active']/max(starting['active'],1e-12))


def fit_strategy(model, trainbatch, target, active, scenario, validationbatch,
                 valtarget, valactive, config, choice, seed):
    """Return selected_state_dict, epoch_log, metadata; leave model selected/eval.

    ``config`` contains epochs, patience, batch_size, learning_rate,
    weight_decay and optionally smooth_l1_beta. ``choice`` is a strategy name
    or a dictionary with key ``strategy``. Strategy constants are intentionally
    fixed here: L2-SP .01, functional anchor .5, SAM rho .01, EMA .95,
    phase/rate .25 and terminal .25, group-DRO eta .05.
    LBFGS uses full TRAIN batches without gradient clipping or AdamW decay;
    lbfgs_epochs/lbfgs_patience optionally override its default30/10 schedule.
    """
    strategy = choice if isinstance(choice,str) else choice['strategy']
    trainable_names = configure_trainable(model,strategy)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    reference_parameter = next(model.parameters())
    dtype,device = reference_parameter.dtype,reference_parameter.device
    def target_tensor(value): return torch.as_tensor(value,dtype=dtype,device=device).detach()
    target,valtarget = target_tensor(target),target_tensor(valtarget)
    active = torch.as_tensor(active,dtype=torch.bool,device=device)
    valactive = torch.as_tensor(valactive,dtype=torch.bool,device=device)
    trainbatch = {k:torch.as_tensor(v,device=device).detach() for k,v in trainbatch.items()}
    validationbatch = {k:torch.as_tensor(v,device=device).detach() for k,v in validationbatch.items()}
    n = len(target)
    if n < 1 or target.shape != (n,30) or active.shape != (n,): raise ValueError('Nonempty N by30 training target required')
    if valtarget.shape != (len(valactive),30): raise ValueError('Invalid validation target shape')
    if any(len(v) != n for v in trainbatch.values()) or any(len(v) != len(valtarget) for v in validationbatch.values()):
        raise ValueError('Batch rows do not match targets')
    if not torch.isfinite(target).all() or not torch.isfinite(valtarget).all(): raise ValueError('Finite targets required')
    sids = np.asarray(scenario)
    if sids.shape != (n,): raise ValueError('One TRAIN scenario ID per window required')
    labels,inverse = np.unique(np.column_stack([sids,active.cpu().numpy().astype(int)]),axis=0,return_inverse=True)
    groups = torch.as_tensor(inverse,dtype=torch.long,device=device)
    dro = GroupDRO(len(labels),dtype=dtype,device=device)
    beta = float(config.get('smooth_l1_beta',.1))
    epochs,patience,batch_size = [int(config[key]) for key in ['epochs','patience','batch_size']]
    if strategy == 'lbfgs_polish':
        epochs = int(config.get('lbfgs_epochs',30)); patience = int(config.get('lbfgs_patience',10)); batch_size = n
    if epochs < 0 or patience < 1 or batch_size < 1 or not np.isfinite(beta) or beta <= 0:
        raise ValueError('Invalid fixed training schedule')
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = (torch.optim.LBFGS(parameters,lr=.3,max_iter=1,max_eval=4,history_size=10,line_search_fn='strong_wolfe')
                 if strategy == 'lbfgs_polish' else
                 torch.optim.AdamW(parameters,lr=float(config['learning_rate']),weight_decay=float(config['weight_decay'])))
    anchor = {name:p.detach().clone() for name,p in model.named_parameters()}
    teacher = frozen_teacher_predictions(model,trainbatch) if strategy == 'function_anchor' else None
    ema = copy.deepcopy(model.state_dict()) if strategy == 'ema' else None
    averaged_model = copy.deepcopy(model) if strategy == 'ema' else None
    start = time.perf_counter()
    starting_values = _validation(model,validationbatch,valtarget,valactive)
    best_score = _criterion(starting_values,starting_values)
    best_values = starting_values.copy(); best_epoch = -1; best_state = copy.deepcopy(model.state_dict())
    logs = [{'epoch':-1,'training_loss':None,'selection_score':best_score,'best_score':best_score,
             'checkpoint_improved':True,'elapsed_s':0.,**{f'validation_{k}':v for k,v in starting_values.items()}}]
    stale = 0; optimizer_steps = 0; backward_passes = 0; executed = 0
    for epoch in range(epochs):
        model.train(); batch_logs = []; batch_sizes = []
        order = np.arange(n) if strategy == 'lbfgs_polish' else rng.permutation(n)
        for ids_np in np.array_split(order,int(np.ceil(n/batch_size))):
            ids = torch.as_tensor(ids_np,dtype=torch.long,device=device)
            batch = {key:value[ids] for key,value in trainbatch.items()}
            def closure():
                prediction,diag = model(batch,batch['acceleration'])
                window_losses = per_window_loss(prediction,target[ids],beta)
                data_loss = dro.loss(window_losses,groups[ids]) if strategy == 'group_dro' else window_losses.mean()
                memory = .01*diag['initial_memory'].square().mean()
                force = .01*diag['innovation_force'].square().mean()
                loss = data_loss+memory+force
                components = {'data_loss':float(data_loss.detach()),'memory_regularizer':float(memory.detach()),
                              'force_regularizer':float(force.detach())}
                if strategy == 'l2sp':
                    penalty = .01*anchor_penalty(model,anchor); loss = loss+penalty
                    components['parameter_anchor']=float(penalty.detach())
                elif strategy == 'function_anchor':
                    penalty = .5*((prediction-teacher[ids])/.1).square().mean(); loss = loss+penalty
                    components['function_anchor']=float(penalty.detach())
                elif strategy == 'phase_space':
                    rate,terminal = phase_space_terms(prediction,target[ids],batch['omega'],beta)
                    loss = loss+.25*rate+.25*terminal
                    components.update(rate_penalty=float((.25*rate).detach()),terminal_penalty=float((.25*terminal).detach()))
                return loss,components
            if strategy == 'lbfgs_polish':
                def lbfgs_closure():
                    nonlocal backward_passes
                    optimizer.zero_grad(set_to_none=True)
                    loss,_ = closure()
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite LBFGS loss')
                    loss.backward(); backward_passes += 1
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                        raise FloatingPointError('Nonfinite LBFGS gradient')
                    return loss
                optimizer.step(lbfgs_closure)
                # Log the accepted parameter state, not a line-search trial.
                with torch.no_grad(): accepted,components = closure()
                if not torch.isfinite(accepted): raise FloatingPointError('Nonfinite accepted LBFGS loss')
                loss_value = float(accepted)
            elif strategy == 'sam':
                loss_value,components = sam_step(model,optimizer,closure)
                backward_passes += 2
            else:
                optimizer.zero_grad(set_to_none=True); loss,components = closure()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fine-tuning loss')
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('Nonfinite fine-tuning gradient')
                torch.nn.utils.clip_grad_norm_(parameters,2.); optimizer.step()
                loss_value = float(loss.detach()); backward_passes += 1
            optimizer_steps += 1
            if ema is not None: update_ema(ema,model.state_dict())
            batch_logs.append({'loss':loss_value,**components});batch_sizes.append(len(ids_np))
        if averaged_model is not None:
            averaged_model.load_state_dict(ema); candidate = averaged_model
        else: candidate = model
        values = _validation(candidate,validationbatch,valtarget,valactive)
        score = _criterion(values,starting_values)
        if not np.isfinite(score): raise FloatingPointError('Nonfinite validation criterion')
        improved = score < best_score-1e-8
        if improved:
            best_score,best_epoch,stale = score,epoch,0
            best_values = values.copy(); best_state = copy.deepcopy(candidate.state_dict())
        else: stale += 1
        fields = {key:float(np.average([item[key] for item in batch_logs],weights=batch_sizes)) for key in batch_logs[0]}
        logs.append({'epoch':epoch,'training_loss':fields.pop('loss'),'selection_score':score,
                     'best_score':best_score,'checkpoint_improved':improved,'elapsed_s':time.perf_counter()-start,
                     **{f'training_{key}':value for key,value in fields.items()},
                     **{f'validation_{key}':value for key,value in values.items()}})
        executed += 1
        if epoch%10 == 0:
            print(f'  {strategy} seed={seed} epoch={epoch} selection={score:.6f} best={best_score:.6f}',flush=True)
        if stale >= patience: break
    model.load_state_dict(best_state);model.eval()
    selected_state = {key:value.detach().clone() for key,value in best_state.items()}
    metadata = {'strategy':strategy,'seed':int(seed),'best_epoch':best_epoch,'selected_epoch':best_epoch,
                'epochs_executed':executed,'optimizer_steps':optimizer_steps,'backward_passes':backward_passes,
                'optimizer_closure_evaluations':backward_passes,
                'fit_seconds':time.perf_counter()-start,'selection_score':best_score,
                'validation':best_values,'starting_validation':starting_values,
                'trainable_parameter_names':trainable_names,
                'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'total_parameter_count':sum(p.numel() for p in model.parameters()),
                'epoch_minus_one_eligible':True,'validation_gradients':False,
                'regularizers':{'initial_memory':.01,'innovation_force':.01},
                'selection_candidate':'ema_only' if strategy == 'ema' else 'raw_weights',
                'optimizer':'LBFGS' if strategy == 'lbfgs_polish' else 'AdamW',
                'lbfgs':{'lr':.3,'max_iter':1,'max_eval':4,'history_size':10,'line_search_fn':'strong_wolfe',
                         'max_eval_is_not_a_strict_closure_cap':True,'gradient_clipping':False} if strategy == 'lbfgs_polish' else None,
                'group_labels':[{'scenario':int(sid),'active':bool(activity)} for sid,activity in labels],
                'group_weights_final':dro.weights.detach().cpu().tolist() if strategy == 'group_dro' else None,
                'group_dro_batch_rule':'conditional weights over present group means; update observed groups only' if strategy == 'group_dro' else None}
    return selected_state,logs,metadata
