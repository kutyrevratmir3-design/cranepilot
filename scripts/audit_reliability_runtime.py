"""Synthetic equilibrium and single-packet timing checks; no real-data score selection."""
import json,time
from pathlib import Path
import numpy as np
from cranepilot.data import sha256
from cranepilot.reliability_pilot import FrozenReliabilityPilot,DEFAULT_RELIABILITY_BUNDLE

def packet(name,length,wave=False,cal=None):
    family,axis=name.split('_');coordinates=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if c=='x' else 'Jr' for c in coordinates];q=np.zeros((1,401))
    if wave:
        w=np.sqrt(9.81/(cal['a']*length+cal['b']));q=.02*np.cos(w*(-np.arange(401)*.01))[None,:]
    history={key:np.zeros((1,401)) for key in coordinates+refs};history.update(L=np.full((1,401),length),**{axis:q});future={key:np.zeros((1,151)) for key in refs}
    return history,future,dict(family=family,axis=axis,known_plan=True)

def main():
    dest=Path('results/modal_backtest_gate_20261107/runtime_diagnostics.json')
    if dest.exists():raise FileExistsError('Retain earlier diagnostics')
    pilot=FrozenReliabilityPilot();zero=[];timing=[]
    for name,case in pilot.cases.items():
        for length in [.1,.4,.8]:
            history,future,kwargs=packet(name,length);r=pilot.forecast(history,future,**kwargs);assert r['eligible'].all();zero.append(dict(case=name,length_m=length,maximum_absolute_forecast_deg=float(np.abs(np.rad2deg(r['forecast_rad'])).max()),maximum_absolute_modal_deg=float(np.abs(np.rad2deg(r['modal_forecast_rad'])).max()),maximum_absolute_arx_deg=float(np.abs(np.rad2deg(r['reference_arx_rad'])).max()),modal_weight=float(r['modal_weight'][0])))
        history,future,kwargs=packet(name,.4,True,case['calibration']);values=[]
        for i in range(10):
            start=time.perf_counter();pilot.forecast(history,future,**kwargs);elapsed=1000*(time.perf_counter()-start)
            if i>=2:values.append(elapsed)
        timing.append(dict(case=name,single_packet_ms=values,median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95))))
    result=dict(source_sha256=sha256(Path(__file__)),bundle_manifest_sha256=sha256(DEFAULT_RELIABILITY_BUNDLE/'manifest.json'),zero_input=zero,timing=timing,zero_response_is_exact=bool(all(r['maximum_absolute_forecast_deg']==0 for r in zero)),scope='Fixed synthetic inputs, three positive cable lengths, five cases; pure modal equilibrium and ARX/blend bias explicitly compared. Timing first2warmups discarded; not a hardware or real-time guarantee. No model or threshold changed.')
    dest.write_text(json.dumps(result,indent=2)+'\n');print('MAX_ZERO_DEG',max(r['maximum_absolute_forecast_deg'] for r in zero));print('TIMING',json.dumps([{k:v for k,v in r.items() if k!='single_packet_ms'} for r in timing]))
if __name__=='__main__':main()
