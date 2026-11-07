"""Replay every development fit and all 90 comparisons, without opening old TEST."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from scripts.external_crane_data import load_sequence, packets, concatenate
from scripts.external_baselines import score
from scripts.matched_adaptation import linear_predict, load_linear
from scripts.shared_response import predict, load
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

RUN=Path('artifacts/runs/shared_response_20261107')
OUT=Path('results/shared_response_20261107')

def main():
    dest=OUT/'verification.json'
    if dest.exists(): raise FileExistsError('Retain prior verification')
    p=json.loads((RUN/'provenance.json').read_text());cfg=p['config'];s=json.loads((RUN/'selection.json').read_text())
    r=dict(passed=False,independent_confirmation=False,old_test_records_read=False,provenance_sha256=sha256(RUN/'provenance.json'),source_sha256=sha256(Path(__file__)))
    try:
        assert p['status']=='complete' and not p['independent_confirmation'] and not p['old_test_records_read']
        verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN)
        r['original_files']=original_files();assert r['original_files']['passed']
        val=concatenate([packets(load_sequence(cfg,sid,cfg['validation_ids']),cfg,s['offset_rad'],cfg['evaluation_stride']) for sid in cfg['validation_ids']])
        parent=Path(cfg['parent_run']);ref=linear_predict(load_linear(parent/'strong_arx.npz'),val,cfg)
        expected=dict(strong_arx=ref,previous_narx=linear_predict(load_linear(parent/'structured_narx.npz'),val,cfg))
        rows=pd.read_csv(RUN/'candidate_validation.csv');assert len(rows)==90
        values=[]
        for name,frame in rows.groupby('file',sort=False):
            y=predict(load(RUN/name),val,cfg)
            for _,row in frame.iterrows():
                value=score((1-row.blend)*ref+row.blend*y,val,cfg);np.testing.assert_allclose(value,row.validation_mae_deg,rtol=1e-12,atol=1e-12);values.append(value)
        expected['selected']=ref.copy() if s['selected']=='strong_arx' else (1-s['blend'])*ref+s['blend']*predict(load(RUN/'selected.npz'),val,cfg)
        np.testing.assert_allclose(min(values+[score(ref,val,cfg)]),s['validation_mae_deg'],rtol=1e-12)
        metrics=pd.read_csv(RUN/'validation_metrics.csv')
        with np.load(RUN/'validation_predictions.npz') as f:
            np.testing.assert_array_equal(f['k'],val['k']);np.testing.assert_array_equal(f['scenario'],val['scenario'])
            for name,y in expected.items():
                np.testing.assert_array_equal(f[name],y)
                for sid in cfg['validation_ids']:
                    mask=val['scenario']==sid
                    for h in cfg['horizon_index']:
                        row=metrics[(metrics.method==name)&(metrics.record==sid)&(metrics.horizon_steps==h+1)].iloc[0]
                        np.testing.assert_allclose(np.abs(np.rad2deg(y[mask,h]-val['target'][mask,h])).mean(),row.mae_deg,rtol=1e-12,atol=1e-12)
        small={k:v[:2].copy() for k,v in val.items()};a=predict(load(RUN/'selected.npz'),small,cfg);small['target'][:]=999;small['actual_horizon_s'][:]=-999
        np.testing.assert_array_equal(a,predict(load(RUN/'selected.npz'),small,cfg))
        r.update(passed=True,fits_replayed=rows.file.nunique(),comparisons_replayed=len(rows),forecasts_replayed=len(val['k'])*len(expected),maximum_replay_error=0.,selection_validation_only=True,targets_and_future_timestamps_ignored=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='original_files'},indent=2))

if __name__=='__main__':main()
