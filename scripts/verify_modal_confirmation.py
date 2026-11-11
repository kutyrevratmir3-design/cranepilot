"""Exact frozen-method replay and independent metrics reconstruction, no model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import packets
from scripts.run_modal_confirmation import RUN,load_confirmation,forecasts
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

OUT=Path('results/modal_confirmation_20261107')
def main():
    OUT.mkdir(exist_ok=True);dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain every verification')
    p=json.loads((RUN/'provenance.json').read_text());cfg=p['config'];freeze=p['freeze'];parent=Path(freeze['parent_run']);selection=json.loads((parent/'selection.json').read_text());metrics=pd.read_csv(RUN/'metrics.csv');coverage=pd.read_csv(RUN/'coverage.csv')
    r=dict(passed=False,checks=[],independently_collected_confirmation=False,provenance_sha256=sha256(RUN/'provenance.json'),source_sha256=sha256(Path(__file__)))
    try:
        assert p['status']=='complete' and sorted(p['records_read'])==cfg['sealed_confirmation_ids'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(freeze['frozen_parent_hashes'])
        old=json.loads((parent/'provenance.json').read_text());verify_hashes(old['guarded_hashes']);verify_hashes(old['result_hashes'],parent);assert p['selection_unchanged_sha256']==sha256(parent/'selection.json')
        r['original_files']=original_files();assert r['original_files']['passed']
        empty=[]
        for family,fold in cfg['families'].items():
            records=load_confirmation(cfg,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';case=selection['cases'][name];cal=LengthCalibration(**json.loads((parent/name/'calibration.json').read_text()))
                for sid,seq in records.items():
                    part,counts=packets(seq,axis,cfg,cal);row=coverage[(coverage['case']==name)&(coverage.scenario==sid)].iloc[0]
                    for key,value in counts.items():
                        if isinstance(value,(int,float)):np.testing.assert_allclose(row[key],value,rtol=1e-12,atol=1e-12)
                    if not len(part['k']):empty.append(dict(case=name,scenario=sid));continue
                    y,diag=forecasts(part,cfg,case,parent/name)
                    with np.load(RUN/f'{name}_s{sid}.npz') as f:
                        np.testing.assert_array_equal(f['k'],part['k'])
                        for key,value in diag.items():np.testing.assert_array_equal(value,f['diagnostic_'+key])
                        for method,pred in y.items():
                            np.testing.assert_array_equal(pred,f[method])
                            for j,h in enumerate(cfg['horizons_raw']):
                                row=metrics[(metrics['case']==name)&(metrics.scenario==sid)&(metrics.method==method)&(np.abs(metrics.horizon_s-h*.01)<1e-10)].iloc[0];err=np.rad2deg(pred[:,j]-part['target'][:,j]);np.testing.assert_allclose([np.abs(err).mean(),np.sqrt((err**2).mean())],[row.mae_deg,row.rmse_deg],rtol=1e-12,atol=1e-12)
                    small={k:v[:2].copy() for k,v in part.items()};a,_=forecasts(small,cfg,case,parent/name);small['target'][:]=999.;b,_=forecasts(small,cfg,case,parent/name)
                    for method in a:np.testing.assert_array_equal(a[method],b[method])
                    r['checks'].append(dict(case=name,scenario=sid,packets=len(part['k']),methods=len(y),maximum_replay_error=0.,targets_ignored=True));print('VERIFIED',name,sid,flush=True)
        assert empty==p['empty_cases'];r.update(passed=True,records=len(set(c['scenario'] for c in r['checks'])),case_records=len(r['checks']),forecasts_replayed=sum(c['packets']*c['methods'] for c in r['checks']),empty_cases=empty,selection_unchanged=True,all_metrics_recomputed=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
