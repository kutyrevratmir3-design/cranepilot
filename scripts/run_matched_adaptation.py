"""Train/validation-only matched development, with explicit retrospective evaluation."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from cranepilot.benchmark import environment_record
from scripts.external_crane_data import config as source_config,load_sequence,packets,concatenate,train_offset
from scripts.external_baselines import score
from scripts.run_broad_search import predict
from scripts.run_external_benchmark import verify_hashes,write
from scripts.matched_adaptation import CONFIG,fit_linear,linear_predict,save_linear,load_linear,angular_batch,make_angular,train_angular

RUN=Path('artifacts/runs/matched_adaptation_20261107')
DESIGN=Path('docs/MATCHED_ADAPTATION_DESIGN_20261107.md')


def get_config():
    cfg=json.loads(CONFIG.read_text());selected=json.loads((Path(cfg['source_run'])/'selection.json').read_text());cfg['oscillator_parameters']=selected['oscillator_parameters'];return cfg


def load_selected(out,cfg):
    selection=json.loads((out/'selection.json').read_text());models={}
    for name in ['strong_arx','physics_prior_arx','structured_narx']:models[name]=load_linear(out/f'{name}.npz')
    for name in ['adapted_neural','equilibrium_neural']:
        ensemble=[]
        for seed in cfg['seeds']:
            m=make_angular(seed,name=='equilibrium_neural');m.load_state_dict(torch.load(out/f'{name}_seed{seed}.pt',weights_only=True));ensemble.append(m.eval())
        models[name]=ensemble
    return selection,models


def all_predictions(models,p,cfg):
    b=angular_batch(p,cfg);result={}
    for name,m in models.items():result[name]=np.stack([predict(x,b) for x in m]).mean(0) if isinstance(m,list) else linear_predict(m,p,cfg)
    assert set(result)==set(cfg['methods'])
    if any(not np.isfinite(y).all() for y in result.values()):raise FloatingPointError('Nonfinite prediction')
    return result


def record(models,ids,cfg,offset,out,label):
    folder=out/label;folder.mkdir();rows=[]
    for sid in ids:
        p=packets(load_sequence(cfg,sid,ids),cfg,offset,cfg['evaluation_stride']);pred=all_predictions(models,p,cfg)
        np.savez_compressed(folder/f'record_{sid}.npz',k=p['k'],**pred)
        for name,y in pred.items():
            err=np.rad2deg(y-p['target'])
            for h in cfg['horizon_index']:rows.append(dict(group=label,record=sid,method=name,horizon_steps=h+1,mae_deg=float(np.abs(err[:,h]).mean()),rmse_deg=float(np.sqrt((err[:,h]**2).mean()))))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['development','retrospective'],required=True);ap.add_argument('--output',type=Path,default=RUN);a=ap.parse_args();out=a.output;cfg=get_config();torch.set_num_threads(cfg['threads'])
    locked=[CONFIG,DESIGN,Path('docs/SAME_RIG_DATA_ACCESS_20261107.md'),Path('scripts/matched_adaptation.py'),Path('scripts/run_matched_adaptation.py'),Path('tests/test_matched_adaptation.py')]
    if a.stage=='development':
        for path in locked:
            if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit first {path}')
        if out.exists():raise FileExistsError('Retain earlier attempts')
        parent=Path(cfg['source_run']);old=json.loads((parent/'provenance.json').read_text());verify_hashes(old['guarded_hashes']);verify_hashes(old['result_hashes'],parent)
        out.mkdir(parents=True);paths=locked+list(Path('src/cranepilot').glob('*.py'))+list(Path('artifacts/research_bundle_20261107').rglob('*'))+[Path('scripts/external_baselines.py'),Path('scripts/external_crane_data.py'),Path('scripts/predict_cranepilot.py'),Path('scripts/predict_research_bundle.py'),parent/'selection.json',parent/'provenance.json']
        prov=dict(status='development_running',config=cfg,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),guarded_hashes={str(p):sha256(p) for p in paths if p.is_file()},environment=environment_record(),independent_confirmation=False,confirmation_source=None,development_reads=cfg['train_ids']+cfg['validation_ids'],started_utc=pd.Timestamp.now(tz='UTC').isoformat())
        write(out/'provenance.json',prov)
        try:
            train_sequences=[load_sequence(cfg,sid,cfg['train_ids']) for sid in cfg['train_ids']];offset=train_offset(train_sequences)
            train=concatenate([packets(s,cfg,offset,cfg['train_stride']) for s in train_sequences]);val=concatenate([packets(load_sequence(cfg,sid,cfg['validation_ids']),cfg,offset,cfg['evaluation_stride']) for sid in cfg['validation_ids']])
            fits=[];choices={};folder=out/'ridge_candidates';folder.mkdir()
            for name,basis,bases in [('strong_arx','linear',['none','geometric']),('physics_prior_arx','linear',['fitted']),('structured_narx','structured',['fitted'])]:
                best=None
                for base in bases:
                    for alpha in cfg['ridge_grid']:
                        m=fit_linear(train,cfg,alpha,basis,base);value=score(linear_predict(m,val,cfg),val,cfg);file=folder/f'{name}_{base}_{alpha:g}.npz';save_linear(file,m)
                        fits.append(dict(method=name,base=base,alpha=alpha,validation_mae_deg=value,file=str(file.relative_to(out))))
                        if best is None or value<best[0]:best=(value,m)
                save_linear(out/f'{name}.npz',best[1]);choices[name]=dict(base=best[1]['base'],alpha=best[1]['alpha'],validation_mae_deg=best[0]);print('RIDGE',name,choices[name],flush=True)
            pd.DataFrame(fits).to_csv(out/'ridge_validation.csv',index=False);logs=[];meta=[]
            for name in ['adapted_neural','equilibrium_neural']:
                for seed in cfg['seeds']:
                    start=time.perf_counter();m,history,info=train_angular(train,val,cfg,seed,name=='equilibrium_neural');torch.save(m.state_dict(),out/f'{name}_seed{seed}.pt');info.update(method=name,fit_seconds=time.perf_counter()-start);meta.append(info)
                    logs.extend(dict(method=name,seed=seed,**row) for row in history);pd.DataFrame(logs).to_csv(out/'training.csv',index=False);write(out/'neural_fits.json',meta)
            selection=dict(offset_rad=offset,linear=choices,neural=meta,candidate=None,selection_criterion='minimum equal-record validation MAE among candidates and strong_arx',test_used_for_selection=False,independent_confirmation_available=False)
            write(out/'selection.json',selection);_,models=load_selected(out,cfg);record(models,cfg['validation_ids'],cfg,offset,out,'validation')
            frame=pd.read_csv(out/'validation/metrics.csv');scores=frame.groupby(['record','method']).mae_deg.mean().groupby('method').mean();selection['validation_scores']=scores.to_dict();selection['candidate']=str(scores.idxmin());write(out/'selection.json',selection)
            prov.update(status='selection_complete',selection_sha256=sha256(out/'selection.json'),locked_utc=pd.Timestamp.now(tz='UTC').isoformat(),development_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
        except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
        finally:write(out/'provenance.json',prov)
    else:
        prov=json.loads((out/'provenance.json').read_text());assert prov['status']=='selection_complete';verify_hashes(prov['guarded_hashes']);verify_hashes(prov['development_hashes'],out)
        try:
            selection,models=load_selected(out,cfg);record(models,cfg['test_ids'],cfg,selection['offset_rad'],out,'retrospective')
            prov.update(status='complete',independent_confirmation=False,test_records_previously_seen=True,no_refit_after_validation_lock=True,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),result_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
        except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
        finally:write(out/'provenance.json',prov)
    print('COMPLETE',a.stage,flush=True)

if __name__=='__main__':main()
