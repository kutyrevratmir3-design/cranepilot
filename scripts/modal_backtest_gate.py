"""Choose model reliability from forecasts of already observed recent history."""
import numpy as np
from cranepilot.data import QUANTIZATION_RAD
from scripts.modal_history import modal
from scripts.modal_reference_challenge import predict

def sliced(p,start,stop,calibration):
    out={}
    for key,value in p.items():
        if key=='target':continue
        if getattr(value,'ndim',0)==2 and value.shape[1]==81:out[key]=value[:,start:stop]
        else:out[key]=value.copy()
    out['omega']=calibration.omega(out['length'][:,0])
    return out

def backtest(p,cfg,gate_cfg,calibration,short_model):
    shift=gate_cfg['backtest_steps'];prefix=sliced(p,shift,None,calibration)
    if prefix['theta'].shape[1]!=gate_cfg['short_history_samples']:raise ValueError('Unexpected causal prefix length')
    indices=np.array([cfg['horizons_raw'].index(h) for h in gate_cfg['backtest_horizons_raw']]);observed_indices=shift-np.array(gate_cfg['backtest_horizons_raw'])//cfg['history_spacing_raw']
    if (observed_indices<0).any():raise ValueError('Backtest would require future measurement')
    truth=p['theta'][:,observed_indices];ym=modal(prefix,cfg,'two_mode')[0][:,indices];yr=predict(short_model,prefix,cfg)[:,indices]
    return dict(modal_backtest_mae=np.abs(ym-truth).mean(1),arx_backtest_mae=np.abs(yr-truth).mean(1),modal_backtest=ym,arx_backtest=yr,observed=truth)

def weight(evidence,variant):
    em=evidence['modal_backtest_mae'];er=evidence['arx_backtest_mae']
    if not np.isfinite(em).all() or not np.isfinite(er).all() or (em<0).any() or (er<0).any():raise ValueError('Finite nonnegative causal errors required')
    if variant=='reference':return np.zeros_like(em)
    if variant=='modal':return np.ones_like(em)
    if variant.startswith('hard_'):return (em<=float(variant.split('_')[1])*er).astype(float)
    if variant=='soft':return er**2/(em**2+er**2+QUANTIZATION_RAD**2/12)
    raise ValueError('Unknown gate')

def compose(ym,yr,evidence,variant):
    w=weight(evidence,variant)
    if ym.shape!=yr.shape or ym.ndim!=2 or len(ym)!=len(w):raise ValueError('Aligned forecasts and errors required')
    return w[:,None]*ym+(1-w[:,None])*yr,w
