"""Development-only shared-horizon screen; no independent test or old-test scoring."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.benchmark import environment_record
from scripts.external_crane_data import load_sequence,packets,concatenate,train_offset
from scripts.external_baselines import score
from scripts.run_external_benchmark import write,verify_hashes
from scripts.matched_adaptation import linear_predict,load_linear
from scripts.shared_response import sufficient_statistics,fit,predict,save,load

CONFIG=Path('configs/shared_response_20261107.json');RUN=Path('artifacts/runs/shared_response_20261107')

def main():
    cfg=json.loads(CONFIG.read_text());source=Path(cfg['source_run']);parent=Path(cfg['parent_run']);old=json.loads((parent/'provenance.json').read_text());assert old['status']=='complete'
    verify_hashes(old['guarded_hashes']);verify_hashes(old['result_hashes'],parent);cfg['oscillator_parameters']=json.loads((source/'selection.json').read_text())['oscillator_parameters']
    locked=[CONFIG,Path('docs/SHARED_RESPONSE_DESIGN_20261107.md'),Path('scripts/shared_response.py'),Path('scripts/run_shared_response.py'),Path('tests/test_shared_response.py')]
    for path in locked:
        if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit first {path}')
    if RUN.exists():raise FileExistsError('Retain earlier attempts')
    RUN.mkdir(parents=True);paths=locked+[Path('scripts/external_crane_data.py'),Path('scripts/external_baselines.py'),Path('scripts/matched_adaptation.py'),parent/'provenance.json',parent/'strong_arx.npz',parent/'structured_narx.npz',source/'selection.json']
    p=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,guarded_hashes={str(p):sha256(p) for p in paths},environment=environment_record(),records_read=cfg['train_ids']+cfg['validation_ids'],old_test_records_read=False,independent_confirmation=False)
    write(RUN/'provenance.json',p)
    try:
        seq=[load_sequence(cfg,sid,cfg['train_ids']) for sid in cfg['train_ids']];offset=train_offset(seq);train=concatenate([packets(s,cfg,offset,cfg['train_stride']) for s in seq]);val=concatenate([packets(load_sequence(cfg,sid,cfg['validation_ids']),cfg,offset,cfg['evaluation_stride']) for sid in cfg['validation_ids']])
        reference=load_linear(parent/'strong_arx.npz');ref=linear_predict(reference,val,cfg);prior=linear_predict(load_linear(parent/'structured_narx.npz'),val,cfg);candidate_dir=RUN/'candidates';candidate_dir.mkdir();rows=[];best=(score(ref,val,cfg),'strong_arx',0.,None);predictions=dict(strong_arx=ref,previous_narx=prior)
        for variant in cfg['variants']:
            stats=sufficient_statistics(train,cfg,variant)
            for alpha in cfg['ridge_grid']:
                model=fit(stats,alpha);name=f'{variant}_{alpha:g}';path=candidate_dir/f'{name}.npz';save(path,model);forecast=predict(model,val,cfg)
                np.testing.assert_array_equal(forecast,predict(load(path),val,cfg))
                for blend in cfg['blend_grid']:
                    value=score((1-blend)*ref+blend*forecast,val,cfg);rows.append(dict(variant=variant,alpha=alpha,blend=blend,validation_mae_deg=value,parameters=len(model['coef']),file=str(path.relative_to(RUN))))
                    if value<best[0]:best=(value,variant,blend,path)
                print('RESPONSE',variant,alpha,'val',score(forecast,val,cfg),flush=True)
        if best[3] is not None:
            model=load(best[3]);save(RUN/'selected.npz',model);predictions['selected']=(1-best[2])*ref+best[2]*predict(model,val,cfg)
        else:predictions['selected']=ref.copy()
        metrics=[]
        for name,y in predictions.items():
            for sid in cfg['validation_ids']:
                mask=val['scenario']==sid
                for h in cfg['horizon_index']:metrics.append(dict(method=name,record=sid,horizon_steps=h+1,mae_deg=float(np.abs(np.rad2deg(y[mask,h]-val['target'][mask,h])).mean())))
        np.savez_compressed(RUN/'validation_predictions.npz',k=val['k'],scenario=val['scenario'],**predictions)
        pd.DataFrame(metrics).to_csv(RUN/'validation_metrics.csv',index=False);pd.DataFrame(rows).to_csv(RUN/'candidate_validation.csv',index=False)
        selection=dict(selected=best[1],blend=best[2],validation_mae_deg=best[0],reference_mae_deg=score(ref,val,cfg),change_percent=100*(best[0]/score(ref,val,cfg)-1),independent_confirmation=False,old_test_evaluated=False,offset_rad=offset)
        write(RUN/'selection.json',selection);p.update(status='complete',selection_sha256=sha256(RUN/'selection.json'),result_hashes={str(f.relative_to(RUN)):sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name!='provenance.json'})
        print('SELECTION',selection,flush=True)
    except Exception as e:p.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(RUN/'provenance.json',p)

if __name__=='__main__':main()
