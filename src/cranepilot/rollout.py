"""Causal command-conditioned windows, actuator and passive angular rollouts.

Future commands are prescribed references in their recorded native scale.
Future measured x/theta are targets only. This fixed-length pilot uses 50 ms
integration and 410 ms observation packets (41 observations at 100 Hz).
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
from torch import nn

from .data import DT, QUANTIZATION_RAD
from .physics import oscillator_forecast
from .passive_memory import PassiveMemoryCell


STEP = .05
RAW_STEP = 5
N_STEPS = 30
HORIZON_INDEX = [1, 3, 6, 9, 14, 19, 29]
HORIZONS_S = [.1, .2, .35, .5, .75, 1., 1.5]


def command_windows(scenario, calibration, origins):
    if not scenario.fixed_length or 'x' not in scenario.signals:
        raise ValueError('Primary rollout protocol requires fixed-length trolley records')
    k = np.asarray(origins, dtype=int)
    if len(k) == 0 or k.min() < 40 or k.max()+N_STEPS*RAW_STEP >= len(scenario.time):
        raise ValueError('Origins do not have complete past/target windows')
    past_index = k[:, None]-np.arange(41)
    state_index = k[:, None]+np.arange(1, N_STEPS+1)*RAW_STEP
    input_index = k[:, None]+np.arange(N_STEPS)*RAW_STEP
    theta, x, u = (scenario.signals[n] for n in ('theta', 'x', 'xr'))
    past_theta, past_x, past_u = theta[past_index], x[past_index], u[past_index]
    omega = calibration.omega(scenario.signals['L'][k])
    harmonic, coefficients = oscillator_forecast(past_theta, omega, HORIZONS_S)
    t = -np.arange(21)*DT
    derivative = (t-t.mean())/np.sum((t-t.mean())**2)
    initial_velocity = past_x[:, :21]@derivative
    # Translation-invariant position history, with current x supplied separately.
    actuator_context = np.column_stack([(past_x-past_x[:, :1])/.1, past_u/2.])
    angular_context = np.column_stack([past_theta/.1, (past_x-past_x[:, :1])/.1,
                                       past_u/2., coefficients/.1])
    return {'k':k, 'omega':omega, 'initial_x':past_x[:, 0],
            'initial_velocity':initial_velocity, 'initial_q':past_theta[:, 0],
            'initial_p':coefficients[:, 1], 'actuator_context':actuator_context,
            'angular_context':angular_context, 'future_u':u[input_index],
            'target_x':x[state_index], 'target_q':theta[state_index],
            'harmonic':harmonic, 'scenario':np.full(len(k),scenario.sid)}


def stack_windows(parts):
    return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}


def tensors(windows):
    return {key:torch.as_tensor(value,dtype=torch.float32)
            for key,value in windows.items() if key not in ('k','scenario')}


class NeuralActuator(nn.Module):
    """Output-error actuator model, not an identified force/voltage model."""
    def __init__(self, context_dim=82, hidden=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(context_dim,32),nn.Tanh(),nn.Linear(32,hidden),nn.Tanh())
        self.recurrent = nn.GRUCell(3,hidden)
        self.acceleration = nn.Linear(hidden,1)
        nn.init.zeros_(self.acceleration.weight)
        nn.init.zeros_(self.acceleration.bias)

    def forward(self, batch):
        hidden = self.encoder(batch['actuator_context'])
        x, v = batch['initial_x'], batch['initial_velocity']
        positions, accelerations = [], []
        for i in range(batch['future_u'].shape[1]):
            inputs = torch.stack([x/.5,v/.2,batch['future_u'][:,i]/2.],-1)
            hidden = self.recurrent(inputs,hidden)
            acceleration = 2.*torch.tanh(self.acceleration(hidden)[:,0])
            x = x+STEP*v+.5*STEP**2*acceleration
            v = v+STEP*acceleration
            positions.append(x)
            accelerations.append(acceleration)
        return torch.stack(positions,1),torch.stack(accelerations,1)


class PassiveForecaster(nn.Module):
    """Same causal observer for zero-memory and four-memory ablations."""
    def __init__(self, context_dim=125, memory_dim=4):
        super().__init__()
        self.memory_dim = memory_dim
        self.encoder = nn.Sequential(nn.Linear(context_dim,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.initial = nn.Linear(32,1+memory_dim)
        nn.init.zeros_(self.initial.weight)
        nn.init.zeros_(self.initial.bias)
        self.cell = PassiveMemoryCell(memory_dim=memory_dim,context_dim=32)

    def forward(self, batch, acceleration):
        context = self.encoder(batch['angular_context'])
        correction = torch.tanh(self.initial(context))
        q = batch['initial_q']
        p = batch['initial_p']+.2*correction[:,0]
        if self.memory_dim:
            budget = .5*torch.sqrt(q.square()+p.square()+QUANTIZATION_RAD**2)
            memory = budget[:,None]*correction[:,1:]/np.sqrt(self.memory_dim)
        else:
            memory = q.new_zeros(len(q),0)
        state = torch.cat([q[:,None],p[:,None],memory],-1)
        prediction,balance = self.cell.rollout(state,batch['omega'],acceleration,STEP,context)
        return prediction, {'balance':balance,'initial_memory':memory}


def train_module(module, train, validation, loss_function, config, seed=0):
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(module.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    n = len(train['initial_q'])
    best, stale, best_state = float('inf'),0,None
    log=[]
    start=time.perf_counter()
    for epoch in range(config['epochs']):
        module.train(); training_loss=[]
        for ids in np.array_split(rng.permutation(n),int(np.ceil(n/config['batch_size']))):
            batch={key:value[ids] for key,value in train.items()}
            loss=loss_function(module,batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; retain this failed run')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(),2.)
            optimizer.step()
            training_loss.append(float(loss.detach()))
        module.eval()
        with torch.no_grad():
            value=float(loss_function(module,validation))
        log.append({'epoch':epoch,'loss':float(np.mean(training_loss)),'validation_loss':value,
                    'elapsed_s':time.perf_counter()-start})
        if value < best-1e-8:
            best,stale,best_state=value,0,copy.deepcopy(module.state_dict())
        else:
            stale+=1
        if epoch%10==0:
            print(f'epoch={epoch} train={np.mean(training_loss):.6g} val={value:.6g}',flush=True)
        if stale>=config['patience']:
            break
    if best_state is None:
        raise RuntimeError('No finite validation checkpoint')
    module.load_state_dict(best_state)
    module.eval()
    return log
