"""Separate experimental direct forecast with a known scalar acquisition age."""
import json,hashlib
from pathlib import Path
import numpy as np
from .modal_pilot import ROOT
from .regime_pilot import FrozenRegimePilot
from .physics import LengthCalibration
from scripts import dated_forecast as df,latency_observer as lo,drive_conditioned as dc

class FrozenDatedPilot:
    def __init__(self,bundle=ROOT/'artifacts/dated_bundle_20261108'):
        self.root=Path(bundle);self.manifest=json.loads((self.root/'manifest.json').read_text())
        for section,base in [('files',self.root),('source_hashes',ROOT)]:
            for key,digest in self.manifest[section].items():
                if hashlib.sha256((base/key).read_bytes()).hexdigest()!=digest:raise ValueError(f'Dated bundle/source hash mismatch: {key}')
        self.base=FrozenRegimePilot(self.root/'regime');self.models={rel:dc.load(self.root/rel) for case in self.manifest['cases'].values() for rel in case['models'].values()}
    def forecast(self,history,*,family,axis,angle_age_samples=0,include_reference=False,include_candidates=False):
        name=f'{family}_{axis}'
        if name not in self.manifest['cases']:raise ValueError('Unsupported family/axis')
        coordinates=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if k=='x' else 'Jr' for k in coordinates]
        if not set([axis,'L']+coordinates+refs)<=set(history):raise ValueError('Missing dated angle/current support/reference')
        raw=df.validate(history,angle_age_samples,axis);age=int(angle_age_samples);pred,diag=df.predictions(raw,age,axis,self.base,name,self.manifest['methods']);result=dict(forecast_rad=pred[self.manifest['selected']],angle_age_samples=age,horizons_s=np.array(self.base.pilot.cfg['horizons_raw'])*.01,scope=self.manifest['scope'])
        if include_candidates:result.update(candidates=pred,diagnostics=diag)
        if include_reference:
            case=self.manifest['cases'][name];known=min(self.manifest['train_ages_raw'],key=lambda a:(abs(a-age),a));key=case['reference'][str(known)]['key'];source,rest=key.split('/',1);cal=LengthCalibration(**self.base.pilot.cases[name]['calibration']);p=lo.packet(raw,axis,cal,age);dated=lo.dated_modal(p['theta'],p['omega_sensor'],p['age_raw'],self.base.pilot.cfg)
            result['matched_arx_rad']=df.arx(raw,age,axis,self.base,name,*rest.split('/')) if source=='aligned' else lo.arx_prediction(self.models[case['models'][key]],p,dated);result['augmented_arx_rad']=lo.arx_prediction(self.models[case['models']['new/'+case['augmented_file']]],p,dated);result['bare_dated_modal_rad']=dated
        return result
