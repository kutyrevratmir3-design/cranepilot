"""Inference-only algebraic compilation of the fixed-coefficient angular flow."""
import numpy as np
import torch
from cranepilot.data import QUANTIZATION_RAD


@torch.no_grad()
def fast_prediction(model,batch,acceleration,dt=.05):
    if model.training:raise ValueError('Inference only: call model.eval() first')
    context=model.encoder(batch['angular_context'])
    correction=torch.tanh(model.initial(context));adjustment=torch.tanh(model.observer(context))
    q=(batch.get('observed_q',batch['initial_q'])+.5*QUANTIZATION_RAD*adjustment[:,0]
       if model.measurement_anchor else batch['initial_q']+.02*adjustment[:,0])
    omega=batch['omega']*(1+.15*adjustment[:,1])
    p=batch['initial_p']*batch['omega']/omega+.2*correction[:,0]
    budget=.5*torch.sqrt(q.square()+p.square()+QUANTIZATION_RAD**2)
    memory=budget[:,None]*correction[:,1:]/np.sqrt(4)
    state=torch.cat([q[:,None],p[:,None],memory],-1)
    damping,coupling,rates=model.cell.parameters_from_context(context)
    C=model.cell.closure_matrix(damping,coupling,rates)
    n,dim=state.shape;eye=torch.eye(dim,dtype=state.dtype,device=state.device).expand(n,-1,-1)
    phase=omega*dt;half=.25*phase[:,None,None]*C
    D=torch.linalg.solve(eye-half,eye+half)
    cosine=torch.cos(phase);sine=torch.sin(phase)
    R=eye.clone();R[:,0,0]=cosine;R[:,1,1]=cosine;R[:,0,1]=sine;R[:,1,0]=-sine
    transition=D@R@D
    port=state.new_zeros(n,dim);port[:,0]=2*torch.sin(.5*phase)**2;port[:,1]=sine
    port=(D@port[...,None])[...,0]
    if model.driven:hidden=torch.tanh(model.hidden_initial(context))
    ys=[]
    for k in range(acceleration.shape[1]):
        force=torch.zeros_like(q)
        if model.driven:
            inputs=torch.stack([state[:,0]/.1,state[:,1]/.1,batch['future_u'][:,k]/2,acceleration[:,k],omega/5],-1)
            hidden=model.innovation(inputs,hidden)
            force=.1*torch.tanh(model.force_output(hidden)[:,0])
        state=(transition@state[...,None])[...,0]+port*(-acceleration[:,k]/9.81+force)[:,None]
        ys.append(state[:,0])
    return torch.stack(ys,1)
