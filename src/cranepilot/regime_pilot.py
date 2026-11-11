"""Separate experimental history-only forecast API; no controller/known-plan claim."""
import hashlib,json
from pathlib import Path
import numpy as np
from .modal_pilot import ROOT
from .reliability_pilot import FrozenReliabilityPilot
from .physics import LengthCalibration
from scripts import drive_conditioned as dc
from scripts.regime_fusion import context,expert_inputs,fuse

DEFAULT_REGIME_BUNDLE=ROOT/'artifacts/regime_bundle_20261107'

class FrozenRegimePilot:
    def __init__(self,bundle=DEFAULT_REGIME_BUNDLE):
        self.root=Path(bundle);self.manifest=json.loads((self.root/'manifest.json').read_text())
        for section,base in [('files',self.root),('source_hashes',ROOT)]:
            for name,digest in self.manifest[section].items():
                if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Regime bundle/source hash mismatch: {name}')
        self.pilot=FrozenReliabilityPilot(self.root/'legacy');self.models={name:dict(reference=case['reference'],long=dc.load(self.root/case['long']),short=dc.load(self.root/case['short']),prior=dc.load(self.root/case['prior'])) for name,case in self.manifest['cases'].items()}

    def forecast(self,history,*,family,axis):
        name=f'{family}_{axis}'
        if name not in self.manifest['cases']:raise ValueError('Unsupported family/axis')
        coordinates=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if k=='x' else 'Jr' for k in coordinates];required=['L',axis]+coordinates+refs
        if not set(required)<=set(history) or set(history)-set(['L','theta','beta']+coordinates+refs):raise ValueError('Only required past history channels allowed; no targets/future data')
        raw={k:np.asarray(v,dtype=float) for k,v in history.items()};shape=raw['L'].shape
        if len(shape)!=2 or not shape[0] or shape[1]!=401 or any(a.shape!=shape or not np.isfinite(a).all() for a in raw.values()) or (raw['L']<=0).any():raise ValueError('Finite aligned (N,401) newest-first100Hz histories and positive length required')
        case=self.pilot.cases[name];cal=LengthCalibration(**case['calibration']);p=dict(theta=raw[axis][:,::5],length=raw['L'][:,::5],omega=cal.omega(raw['L'][:,0]),k=np.arange(shape[0]),**{k:raw[k][:,::5] for k in coordinates+refs});ratio=context(raw,self.pilot.cfg['past_position_range']);e=expert_inputs(p,ratio,self.pilot,name,self.models[name]);y,w=fuse(e,self.manifest['selected']);return dict(forecast_rad=y,legacy_forecast_rad=e['legacy'],active_forecast_rad=e['active'],weights=w,context_ratio=ratio,quiet_preserved=ratio<=1,expert_forecasts_rad=e['forecasts'],prefix_errors_rad=e['errors'],old_prior=e['old_prior'],active_prior=e['new_prior'],horizons_s=np.array(self.pilot.cfg['horizons_raw'])*.01,scope=self.manifest['scope'])
