"""Replay development/retrospective outputs; never mark them independent confirmation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from scripts.external_crane_data import load_sequence,packets
from scripts.run_matched_adaptation import RUN,load_selected,all_predictions
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

OUT=Path('results/matched_adaptation_20261107')

def main():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain previous verification')
    prov=json.loads((RUN/'provenance.json').read_text());cfg=prov['config'];torch.set_num_threads(cfg['threads']);r=dict(passed=False,independent_confirmation=False,provenance_sha256=sha256(RUN/'provenance.json'),verifier_sha256=sha256(Path(__file__)),checks=[])
    try:
        assert prov['status']=='complete' and not prov['independent_confirmation'] and prov['test_records_previously_seen']
        for mapping,base in [('guarded_hashes',Path('.')),('development_hashes',RUN),('result_hashes',RUN)]:verify_hashes(prov[mapping],base)
        parent=Path(cfg['source_run']);previous=json.loads((parent/'provenance.json').read_text());verify_hashes(previous['guarded_hashes']);verify_hashes(previous['result_hashes'],parent)
        r['original_files']=original_files();assert r['original_files']['passed']
        selection,models=load_selected(RUN,cfg);assert prov['selection_sha256']==sha256(RUN/'selection.json')
        assert not(set(prov['development_reads'])&set(cfg['test_ids']))
        for group,ids in [('validation',cfg['validation_ids']),('retrospective',cfg['test_ids'])]:
            metrics=pd.read_csv(RUN/group/'metrics.csv')
            for sid in ids:
                p=packets(load_sequence(cfg,sid,ids),cfg,selection['offset_rad'],cfg['evaluation_stride']);y=all_predictions(models,p,cfg)
                maximum=0.
                with np.load(RUN/group/f'record_{sid}.npz') as f:
                    np.testing.assert_array_equal(f['k'],p['k'])
                    for method,pred in y.items():
                        difference=float(np.abs(np.rad2deg(pred-f[method])).max());maximum=max(maximum,difference);assert difference<5e-5
                        for h in cfg['horizon_index']:
                            row=metrics[(metrics.record==sid)&(metrics.method==method)&(metrics.horizon_steps==h+1)].iloc[0];err=np.rad2deg(f[method][:,h]-p['target'][:,h])
                            np.testing.assert_allclose([np.abs(err).mean(),np.sqrt((err**2).mean())],[row.mae_deg,row.rmse_deg],rtol=1e-12,atol=1e-12)
                small={k:v[:2].copy() for k,v in p.items()};a=all_predictions(models,small,cfg);small['target'][:]=999;small['actual_horizon_s'][:]=-999;b=all_predictions(models,small,cfg)
                for method in a:np.testing.assert_array_equal(a[method],b[method])
                r['checks'].append(dict(group=group,record=sid,packets=len(p['k']),methods=len(y),replay_max_deg=maximum,metrics_independently_recomputed=True,targets_and_future_timestamps_ignored=True));print('VERIFIED',group,sid,maximum,flush=True)
        table=pd.read_csv(RUN/'validation/metrics.csv').groupby(['record','method']).mae_deg.mean().groupby('method').mean()
        for name,value in table.items():np.testing.assert_allclose(value,selection['validation_scores'][name],rtol=1e-12,atol=1e-12)
        assert selection['candidate']==table.idxmin();r.update(passed=True,forecasts=sum(c['packets']*c['methods'] for c in r['checks']),records=len(r['checks']),selection_validation_only=True,no_default_changed=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
