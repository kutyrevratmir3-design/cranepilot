"""Replay every qualification window; independently recompute regime/metric tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.reliability_pilot import FrozenReliabilityPilot
from scripts.run_active_qualification import RUN,existing_records,groups
from scripts.active_qualification import qualification_packets,frozen_predictions
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

OUT=Path('results/active_qualification_20261107')
def main():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain earlier verification')
    p=json.loads((RUN/'provenance.json').read_text());cfg=p['config'];task=p['task'];r=dict(passed=False,checks=[],source_sha256=sha256(Path(__file__)),provenance_sha256=sha256(RUN/'provenance.json'),independent_confirmation=False)
    try:
        assert p['status']=='complete' and not p['fitting_performed'] and not p['selection_performed'] and not p['api_gate_changed'] and p['all_records_previously_opened'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);r['original_files']=original_files();assert r['original_files']['passed'];pilot=FrozenReliabilityPilot(cfg['bundle']);metrics=pd.read_csv(RUN/'metrics.csv');coverage=pd.read_csv(RUN/'coverage.csv')
        for family,fold in task['families'].items():
            records,roles=existing_records(task,fold)
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.cases[name]['calibration'])
                for sid,s in records.items():
                    packet,target,labels=qualification_packets(s,axis,task,cfg,cal);assert not(set(packet)-{'theta','length','omega','k','scenario','x','xr','J','Jr'});y,diag=frozen_predictions(packet,pilot,name);maximum=0.
                    with np.load(RUN/f'{name}_s{sid}.npz') as f:
                        np.testing.assert_array_equal(f['k'],packet['k'])
                        for key,value in labels.items():np.testing.assert_array_equal(value,f[key])
                        for key,value in dict(**y,**{'diagnostic_'+k:v for k,v in diag.items()}).items():
                            error=float(np.abs(value-f[key]).max());assert error<1e-12;maximum=max(maximum,error)
                    for group,mask in groups(labels,cfg).items():
                        count=int(mask.sum());c=coverage[(coverage['case']==name)&(coverage.scenario==sid)&(coverage['group']==group)].iloc[0];assert c.windows==count and c.total_windows==len(mask) and c.role==roles[sid]
                        for method,forecast in y.items():
                            frame=metrics[(metrics['case']==name)&(metrics.scenario==sid)&(metrics['group']==group)&(metrics.method==method)].sort_values('horizon_s');assert len(frame)==len(task['horizons_raw']) and (frame.windows==count).all() and (frame.role==roles[sid]).all()
                            if not count:assert frame.mae_deg.isna().all() and frame.rmse_deg.isna().all();continue
                            err=np.rad2deg(forecast[mask]-target[mask]);np.testing.assert_allclose(frame.mae_deg.to_numpy(),np.abs(err).mean(0),rtol=1e-12,atol=1e-12);np.testing.assert_allclose(frame.rmse_deg.to_numpy(),np.sqrt((err**2).mean(0)),rtol=1e-12,atol=1e-12)
                    small={k:v[:2].copy() for k,v in packet.items()};a,_=frozen_predictions(small,pilot,name);small['target']=np.full((2,7),999.)
                    try:frozen_predictions(small,pilot,name)
                    except ValueError:pass
                    else:raise AssertionError('Inference must reject target input')
                    del small['target'];b,_=frozen_predictions(small,pilot,name)
                    for method in a:np.testing.assert_array_equal(a[method],b[method])
                    r['checks'].append(dict(case=name,scenario=sid,role=roles[sid],packets=len(packet['k']),methods=len(y),maximum_replay_error=maximum,targets_kept_separate=True));print('VERIFIED',name,sid,maximum,flush=True)
        assert len(r['checks'])==53;assert sum(c['packets'] for c in r['checks'])==p['packets'];r.update(passed=True,axis_records=len(r['checks']),records=len(set(c['scenario'] for c in r['checks'])),forecast_comparisons=sum(c['packets']*c['methods'] for c in r['checks']),all_regime_metrics_recomputed=True,original_api_gate_unchanged=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
