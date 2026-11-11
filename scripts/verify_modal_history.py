"""Replay all modal development fits and selected predictions; holdout loader remains shut."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import load_allowed,packets,concatenate,modal,predict_arx,load_arx,local_ar,score
from scripts.run_modal_history import RUN
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

OUT=Path('results/modal_history_20261107')
def main():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain previous verification')
    p=json.loads((RUN/'provenance.json').read_text());cfg=p['config'];s=json.loads((RUN/'selection.json').read_text());rows=pd.read_csv(RUN/'candidate_validation.csv')
    r=dict(passed=False,confirmation_records_read=False,independent_confirmation=False,checks=[],provenance_sha256=sha256(RUN/'provenance.json'),selection_sha256=sha256(RUN/'selection.json'),source_sha256=sha256(Path(__file__)))
    try:
        assert p['status']=='complete' and not p['confirmation_records_read'];assert not(set(p['records_read'])&set(cfg['sealed_confirmation_ids']))
        verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);r['original_files']=original_files();assert r['original_files']['passed']
        metrics=pd.read_csv(RUN/'validation_metrics.csv');coverage=pd.read_csv(RUN/'coverage.csv')
        for name,c in s['cases'].items():
            fold=cfg['families'][c['family']];cal=LengthCalibration(**json.loads((RUN/name/'calibration.json').read_text()));parts=[]
            for sid,seq in load_allowed(cfg,fold['validation'],'validation').items():
                part,counts=packets(seq,c['axis'],cfg,cal);parts.append(part);expected=coverage[(coverage['case']==name)&(coverage.role=='validation')&(coverage.scenario==sid)].iloc[0]
                for key,value in counts.items():
                    if isinstance(value,(int,float)):np.testing.assert_allclose(expected[key],value,rtol=1e-12,atol=1e-12)
            va=concatenate(parts);predictions={};arx_rows=rows[(rows['case']==name)&(rows.family=='arx')];assert len(arx_rows)==24
            for _,row in arx_rows.iterrows():
                y=predict_arx(load_arx(RUN/row.file),va,cfg);np.testing.assert_allclose(score(y,va),row.validation_mae_deg,rtol=1e-12,atol=1e-12)
            predictions['strong_arx']=predict_arx(load_arx(RUN/name/'strong_arx.npz'),va,cfg)
            np.testing.assert_allclose(score(predictions['strong_arx'],va),arx_rows.validation_mae_deg.min(),rtol=1e-12)
            for variant in cfg['modal_variants']:
                y,diag=modal(va,cfg,variant);predictions[variant]=y
                with np.load(RUN/name/f'{variant}_diagnostics.npz') as f:
                    for key,value in diag.items():np.testing.assert_array_equal(value,f[key])
            predictions['selected_modal']=predictions[c['modal']];predictions['local_ar']=local_ar(va,cfg,**c['local_ar']);predictions['persistence']=np.repeat(va['theta'][:,:1],len(cfg['horizons_raw']),axis=1)
            for method in ['modal','local_ar']:
                m=rows[(rows['case']==name)&(rows.family==method)]
                np.testing.assert_allclose(c['validation_scores']['selected_modal' if method=='modal' else 'local_ar'],m.validation_mae_deg.min(),rtol=1e-12)
            with np.load(RUN/name/'validation_forecasts.npz') as f:
                np.testing.assert_array_equal(f['k'],va['k']);np.testing.assert_array_equal(f['scenario'],va['scenario'])
                for method,y in predictions.items():
                    np.testing.assert_array_equal(y,f[method]);np.testing.assert_allclose(score(y,va),c['validation_scores'][method],rtol=1e-12,atol=1e-12)
                    for sid in fold['validation']:
                        mask=va['scenario']==sid
                        for j,h in enumerate(cfg['horizons_raw']):
                            row=metrics[(metrics['case']==name)&(metrics.scenario==sid)&(metrics.method==method)&(np.abs(metrics.horizon_s-h*.01)<1e-10)].iloc[0];err=np.rad2deg(y[mask,j]-va['target'][mask,j]);np.testing.assert_allclose([np.abs(err).mean(),np.sqrt((err**2).mean())],[row.mae_deg,row.rmse_deg],rtol=1e-12,atol=1e-12)
            small={k:v[:2].copy() for k,v in va.items()};a=modal(small,cfg,c['modal'])[0];b=predict_arx(load_arx(RUN/name/'strong_arx.npz'),small,cfg);small['target'][:]=999
            np.testing.assert_array_equal(a,modal(small,cfg,c['modal'])[0]);np.testing.assert_array_equal(b,predict_arx(load_arx(RUN/name/'strong_arx.npz'),small,cfg))
            r['checks'].append(dict(case=name,packets=len(va['k']),ridge_fits=24,modal_variants=3,local_ar_settings=10,forecast_methods=len(predictions),maximum_replay_error=0.,poisoned_targets_ignored=True));print('VERIFIED',name,flush=True)
        arx=np.array([c['validation_scores']['strong_arx'] for c in s['cases'].values()]);new=np.array([c['validation_scores']['selected_modal'] for c in s['cases'].values()]);local=np.array([c['validation_scores']['local_ar'] for c in s['cases'].values()]);gate=bool(new.mean()/arx.mean()<=cfg['development_gate']['maximum_mae_ratio'] and (new<arx).sum()>=cfg['development_gate']['minimum_case_wins'] and new.mean()<local.mean());assert gate==s['development_gate_passed']
        r.update(passed=True,ridge_fits_replayed=int(sum(c['ridge_fits'] for c in r['checks'])),forecasts_replayed=int(sum(c['packets']*c['forecast_methods'] for c in r['checks'])),development_gate_reproduced=gate)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
