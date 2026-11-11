"""Frozen residual-sway predictor; no training records or ancestral run tree required."""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.modal_history import modal
from scripts.modal_reference_challenge import load as load_reference,predict as predict_reference

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE=ROOT/'artifacts/modal_bundle_20261107'

class FrozenModalPilot:
    """Conditional forecast from4s of raw100Hz history and a known reference plan.

    Every history array is(N,401), newest first. Every future reference is
    (N,151), current first. Coordinates use metres/radians; future references
    retain their native units and are used only for the constant-plan gate.
    This does not expose a controller or forecast active hoisting.
    """
    def __init__(self,bundle=DEFAULT_BUNDLE):
        self.root=Path(bundle);self.manifest=json.loads((self.root/'manifest.json').read_text());self.cfg=json.loads((self.root/'task.json').read_text())
        for section,base in [('files',self.root),('source_hashes',ROOT)]:
            for name,digest in self.manifest[section].items():
                if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Frozen bundle/source hash mismatch: {name}')
        self.cases={}
        for name,case in self.manifest['cases'].items():
            self.cases[name]=dict(**case,calibration=json.loads((self.root/case['calibration_file']).read_text()),reference=load_reference(self.root/case['reference_file']))

    def forecast(self,history,future_reference,*,family,axis,known_plan):
        if known_plan is not True:raise ValueError('Future reference must be known at forecast time, not future logged feedback output')
        name=f'{family}_{axis}'
        if name not in self.cases:raise ValueError('Unsupported family/axis')
        coordinates=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if k=='x' else 'Jr' for k in coordinates]
        required=['L',axis]+coordinates+refs
        if not set(required)<=set(history) or set(future_reference)!=set(refs):raise ValueError('Missing history or invalid reference channels')
        if not set(history)<=set(['L','theta','beta']+coordinates+refs):raise ValueError('Unknown history channel')
        raw={k:np.asarray(history[k],dtype=float) for k in required};shape=raw['L'].shape
        if len(shape)!=2 or shape[1]!=401 or not shape[0]:raise ValueError('Nonempty (N,401) newest-first history required')
        if any(a.shape!=shape or not np.isfinite(a).all() for a in raw.values()) or (raw['L']<=0).any():raise ValueError('Finite aligned history and positive length required')
        future={k:np.asarray(future_reference[k],dtype=float) for k in refs}
        if any(a.shape!=(shape[0],151) or not np.isfinite(a).all() for a in future.values()):raise ValueError('Finite (N,151) current-first known reference required')
        eligible=np.ptp(raw['L'],axis=1)<=self.cfg['past_length_range_m']
        for coordinate,reference in zip(coordinates,refs):
            eligible &= np.ptp(raw[coordinate],axis=1)<=self.cfg['past_position_range']
            eligible &= np.ptp(future[reference],axis=1)<=self.cfg['future_reference_range']
            eligible &= np.abs(future[reference][:,0]-raw[reference][:,0])<=self.cfg['future_reference_range']
        case=self.cases[name];cal=case['calibration'];n=int(eligible.sum());prediction=np.full((shape[0],len(self.cfg['horizons_raw'])),np.nan);reference=prediction.copy();diagnostics={}
        if n:
            selected={k:v[eligible,::5] for k,v in raw.items()};theta=selected.pop(axis);length=selected.pop('L');omega=np.sqrt(9.81/np.maximum(cal['a']*length[:,0]+cal['b'],.01));p=dict(theta=theta,length=length,omega=omega,k=np.arange(n),**selected)
            y,diagnostics=modal(p,self.cfg,case['variant']);prediction[eligible]=y;reference[eligible]=predict_reference(case['reference'],p,self.cfg)
        return dict(forecast_rad=prediction,reference_arx_rad=reference,horizons_s=np.array(self.cfg['horizons_raw'])*.01,eligible=eligible,eligible_diagnostics=diagnostics,scope=self.manifest['scope'])
