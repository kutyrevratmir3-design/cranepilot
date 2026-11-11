"""Frozen causal blend; a separate experimental profile, no controller interface."""
import hashlib,json
from pathlib import Path
import numpy as np
from .modal_pilot import FrozenModalPilot,ROOT
from .physics import LengthCalibration
from scripts.modal_reference_challenge import load,predict
from scripts.modal_backtest_gate import sliced,backtest,compose

DEFAULT_RELIABILITY_BUNDLE=ROOT/'artifacts/reliability_bundle_20261107'

class FrozenReliabilityPilot(FrozenModalPilot):
    def __init__(self,bundle=DEFAULT_RELIABILITY_BUNDLE):
        root=Path(bundle);manifest=json.loads((root/'manifest.json').read_text())
        for section,base in [('files',root),('source_hashes',ROOT)]:
            for name,digest in manifest[section].items():
                if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Reliability bundle/source hash mismatch: {name}')
        super().__init__(root/'base');self.reliability_manifest=manifest;self.gate_cfg=json.loads((root/'gate.json').read_text());self.short_models={name:load(root/case['short_model_file']) for name,case in manifest['cases'].items()}

    def forecast(self,history,future_reference,*,family,axis,known_plan):
        result=super().forecast(history,future_reference,family=family,axis=axis,known_plan=known_plan);mask=result['eligible'];n=int(mask.sum());name=f'{family}_{axis}';result['modal_forecast_rad']=result['forecast_rad'].copy();result['modal_weight']=np.full(len(mask),np.nan);result['backtest_mae_rad']={key:np.full(len(mask),np.nan) for key in ['modal','arx']};result['scope']=self.reliability_manifest['scope']
        if n:
            theta=np.asarray(history[axis],dtype=float)[mask,::5];length=np.asarray(history['L'],dtype=float)[mask,::5];case=self.reliability_manifest['cases'][name];cal=LengthCalibration(**self.cases[name]['calibration']);p=dict(theta=theta,length=length,omega=cal.omega(length[:,0]),k=np.arange(n));model=self.short_models[name];evidence=backtest(p,self.cfg,self.gate_cfg,cal,model)
            reference=predict(model,sliced(p,0,61,cal),self.cfg) if case['reference']=='short' else result['reference_arx_rad'][mask].copy();y,w=compose(result['forecast_rad'][mask],reference,evidence,self.reliability_manifest['selected'])
            result['forecast_rad'][mask]=y;result['reference_arx_rad'][mask]=reference;result['modal_weight'][mask]=w;result['backtest_mae_rad']['modal'][mask]=evidence['modal_backtest_mae'];result['backtest_mae_rad']['arx'][mask]=evidence['arx_backtest_mae']
        return result
