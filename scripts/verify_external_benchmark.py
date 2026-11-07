"""Replay sealed external forecasts and independently recompute all errors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from scripts.external_crane_data import config,load_sequence,packets,train_offset
from scripts.run_external_benchmark import RUN,load_models,predictions,verify_hashes
from scripts.verify_finetune_campaign import original_files

OUT=Path('results/external_benchmark_20261107')

def main():
    out=OUT;out.mkdir(parents=True,exist_ok=True);dest=out/'verification.json'
    if dest.exists():raise FileExistsError('Retain previous verification')
    p=json.loads((RUN/'provenance.json').read_text());cfg=p['config'];torch.set_num_threads(cfg['threads'])
    result=dict(passed=False,provenance_sha256=sha256(RUN/'provenance.json'),verifier_sha256=sha256(Path(__file__)),checks=[])
    try:
        assert p['status']=='complete' and p['no_test_refitting'] and p['test_predictions_after_selection_lock']
        verify_hashes(p['guarded_hashes']);verify_hashes(p['training_hashes'],RUN);verify_hashes(p['result_hashes'],RUN)
        assert not(set(p['records_parsed_before_test'])&set(cfg['test_ids']))
        assert p['test_records_read']==cfg['test_ids'] and p['selection_sha256']==sha256(RUN/'selection.json')
        original=original_files();assert original['passed'];result['original_files']=original
        models=load_models(RUN,cfg);offset=models['selection']['offset_rad']
        independent_offset=train_offset([load_sequence(cfg,sid,cfg['train_ids']) for sid in cfg['train_ids']]);assert independent_offset==offset
        metrics=[]
        for stage,ids in [('validation',cfg['validation_ids']),('test',cfg['test_ids'])]:
            saved_metrics=pd.read_csv(RUN/stage/'metrics.csv')
            for sid in ids:
                seq=load_sequence(cfg,sid,ids);packet=packets(seq,cfg,offset,cfg['evaluation_stride']);replayed=predictions(models,packet,cfg)
                with np.load(RUN/stage/f'record_{sid}.npz') as f:saved={k:f[k] for k in f.files}
                np.testing.assert_array_equal(saved['k'],packet['k'])
                difference=0.
                for method,y in replayed.items():
                    delta=float(np.max(np.abs(np.rad2deg(y-saved[method]))));difference=max(difference,delta);assert delta<5e-5
                    e=np.rad2deg(saved[method]-packet['target'])
                    for h in cfg['horizon_index']:
                        row=saved_metrics[(saved_metrics.record==sid)&(saved_metrics.method==method)&(saved_metrics.horizon_steps==h+1)].iloc[0]
                        absolute=np.abs(e[:,h]);a=dict(mae_deg=float(absolute.mean()),rmse_deg=float(np.sqrt(np.square(e[:,h]).mean())),p95_absolute_error_deg=float(np.quantile(absolute,.95)))
                        for key,value in a.items():np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-12)
                        metrics.append(dict(stage=stage,record=sid,method=method,horizon_steps=h+1,**a))
                # A shared packet with poisoned targets/future timestamps cannot change ANY predictor.
                small={k:v[:2].copy() for k,v in packet.items()};clean=predictions(models,small,cfg)
                small['target'][:]=999.;small['actual_horizon_s'][:]=-999.;poisoned=predictions(models,small,cfg)
                for name in clean:np.testing.assert_array_equal(clean[name],poisoned[name])
                result['checks'].append(dict(stage=stage,record=sid,packets=len(packet['k']),methods=len(replayed),replay_max_deg=difference,targets_and_future_times_ignored=True,metric_recomputation=True))
                print('VERIFIED',stage,sid,'max_deg',difference,flush=True)
        pd.DataFrame(metrics).to_csv(out/'independent_metrics.csv',index=False)
        result.update(passed=True,methods=8,records=8,packets=sum(c['packets'] for c in result['checks']),forecast_trajectories=sum(c['packets']*c['methods'] for c in result['checks']),
            test_cases=4,test_used_for_refitting=False,raw_external_data_redistributed=False,closed_loop_control_evaluated=False,all_model_predictions_replayed=True)
    except Exception as e:result['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
