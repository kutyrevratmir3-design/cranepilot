"""Phase-normalized causal observation and phase-scaled recurrent forcing.

Angular model only: the separately learned native-command actuator is not
claimed scale equivariant. Bounded latent forcing is an additional energy port.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .passive_memory import PassiveMemoryCell
from .data import QUANTIZATION_RAD


def phase_context(raw,refined,dt=.01,span=1.,points=25):
    omega=np.asarray(raw['omega'])
    angle=raw['angular_context'][:,:41]*.1
    position=raw['actuator_context'][:,:41]*.1
    if np.any(omega*(angle.shape[1]-1)*dt<span-1e-12):
        raise ValueError('Past packet does not cover the required phase interval')
    grid=np.linspace(0,span,points)
    qa=[];xp=[]
    for i,w in enumerate(omega):
        phase=w*np.arange(angle.shape[1])*dt
        qa.append(np.interp(grid,phase,angle[i]))
        xp.append(np.interp(grid,phase,position[i])*w*w/9.81)
    return np.column_stack([np.asarray(qa)/.1,np.asarray(xp)/.1,
        refined['initial_q']/.1,refined['initial_p']/.1,
        (raw['initial_q']-refined['initial_q'])/.1,
        (raw['initial_p']-refined['initial_p'])/.1])


class PhaseNormalizedForecaster(nn.Module):
    def __init__(self,driven=False):
        super().__init__();self.driven=driven
        self.encoder=nn.Sequential(nn.Linear(54,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.initial=nn.Linear(32,7)
        nn.init.zeros_(self.initial.weight);nn.init.zeros_(self.initial.bias)
        self.cell=PassiveMemoryCell(memory_dim=4,context_dim=32)
        if driven:
            self.latent_initial=nn.Linear(32,12)
            self.latent_target=nn.Linear(15,12)
            self.latent_rates=nn.Parameter(torch.zeros(12))
            self.force=nn.Linear(12,1)
            nn.init.zeros_(self.force.weight);nn.init.zeros_(self.force.bias)

    def forward(self,batch,acceleration,dt=.05):
        context=self.encoder(batch['phase_context'])
        adjustment=torch.tanh(self.initial(context))
        q=batch['initial_q']+.02*adjustment[:,0]
        omega=batch['omega']*(1+.15*adjustment[:,1])
        p=batch['initial_p']*batch['omega']/omega+.2*adjustment[:,2]
        budget=.5*torch.sqrt(q.square()+p.square()+QUANTIZATION_RAD**2)
        memory=budget[:,None]*adjustment[:,3:]/2
        state=torch.cat([q[:,None],p[:,None],memory],-1)
        coefficients=self.cell.parameters_from_context(context)
        if self.driven:
            hidden=torch.tanh(self.latent_initial(context))
            decay=torch.exp(-omega[:,None]*dt*(.1+F.softplus(self.latent_rates)))
        ys=[];balances=[];forces=[]
        for k in range(acceleration.shape[1]):
            forcing=torch.zeros_like(q)
            if self.driven:
                features=torch.cat([hidden,state[:,:2]/.1,acceleration[:,k:k+1]/9.81],-1)
                target=torch.tanh(self.latent_target(features))
                hidden=decay*hidden+(1-decay)*target
                forcing=.1*torch.tanh(self.force(hidden)[:,0])
            state,d=self.cell.step(state,omega,acceleration[:,k]-9.81*forcing,dt,coefficients)
            ys.append(state[:,0]);balances.append(d['balance_residual']);forces.append(forcing)
        return torch.stack(ys,1),{'balance':torch.stack(balances,1),
            'initial_memory':memory,'innovation_force':torch.stack(forces,1),
            'frequency_ratio':omega/batch['omega']}


class GRUResidualForecaster(nn.Module):
    """Conventional direct recurrent residual reference; no storage guarantee."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(128,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.recurrent=nn.GRUCell(6,32)
        self.output=nn.Linear(32,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,batch,acceleration,dt=.05):
        hidden=self.encoder(batch['reference_context']);ys=[]
        for k in range(acceleration.shape[1]):
            phase=batch['omega']*dt*(k+1)
            features=torch.stack([batch['future_u'][:,k]/2,acceleration[:,k]/9.81,
                batch['physical_base'][:,k]/.1,torch.sin(phase),torch.cos(phase),batch['omega']/5],-1)
            hidden=self.recurrent(features,hidden)
            ys.append(batch['physical_base'][:,k]+.1*self.output(hidden)[:,0])
        return torch.stack(ys,1),{}
