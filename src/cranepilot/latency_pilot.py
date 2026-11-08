"""Experimental known-angle-age wrapper; timestamps are input, not estimated truth."""
import json,hashlib
from pathlib import Path
import numpy as np
from .modal_pilot import ROOT
from .regime_pilot import FrozenRegimePilot
from .physics import LengthCalibration
from scripts import latency_observer as lo
from scripts import drive_conditioned as dc

DEFAULT_LATENCY_BUNDLE=ROOT/'artifacts/latency_bundle_20261108'

class FrozenLatencyPilot:
    def __init__(self,bundle=DEFAULT_LATENCY_BUNDLE):
        self.root=Path(bundle);self.manifest=json.loads((self.root/'manifest.json').read_text())
        for section,base in [('files',self.root),('source_hashes',ROOT)]:
            for key,digest in self.manifest[section].items():
                if hashlib.sha256((base/key).read_bytes()).hexdigest()!=digest:raise ValueError(f'Latency bundle/source hash mismatch: {key}')
        self.base=FrozenRegimePilot(self.root/'regime');self.reference_models={name:dc.load(self.root/case['reference']['file']) for name,case in self.manifest['cases'].items() if case['reference']['type']=='augmented'}

    def forecast(self,history,*,family,axis,angle_age_samples=0,include_reference=False):
        name=f'{family}_{axis}'
        if name not in self.manifest['cases']:raise ValueError('Unsupported family/axis')
        coordinates=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if k=='x' else 'Jr' for k in coordinates]
        if not set([axis,'L']+coordinates+refs)<=set(history):raise ValueError('Missing current-clock support/reference or dated angle history')
        raw,age=lo.validate(history,angle_age_samples,axis);y,diag=lo.predict(raw,age,axis,self.manifest['selected'],self.base,name);result=dict(forecast_rad=y,weights=diag['weights'],head_estimate_rad=diag['head'],backtest_valid=diag['valid_heads'],angle_age_samples=age,horizons_s=np.array(self.base.pilot.cfg['horizons_raw'])*.01,scope=self.manifest['scope'])
        if include_reference:
            choice=self.manifest['cases'][name]['reference']
            if choice['type']=='augmented':
                cal=LengthCalibration(**self.base.pilot.cases[name]['calibration']);p=lo.packet(raw,axis,cal,age);dated=lo.dated_modal(p['theta'],p['omega_sensor'],p['age_raw'],self.base.pilot.cfg);ref=lo.arx_prediction(self.reference_models[name],p,dated)
            else:
                other=diag if choice['method']==self.manifest['selected'] else lo.predict(raw,age,axis,choice['method'],self.base,name)[1];ref=other[choice['expert']]
            result['matched_arx_rad']=ref
        return result
