"""Preregistered external real-camera comparison with a sealed four-record test."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cranepilot.data import sha256
from cranepilot.benchmark import environment_record
from scripts.external_crane_data import CONFIG,config,load_sequence,packets,concatenate,train_offset
from scripts.external_baselines import geometric,oscillator,fit_oscillator,fit_arx,predict_arx,save_arx,load_arx,ResidualLSTM,fit_lstm,predict_lstm,score,FrozenAngular

DESIGN=Path('docs/EXTERNAL_BENCHMARK_DESIGN_20261107.md')
RUN=Path('artifacts/runs/external_benchmark_20261107')

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def verify_hashes(mapping,base=Path('.')):
    for name,digest in mapping.items():
        if sha256(base/name)!=digest:raise ValueError(f'Changed frozen input {name}')


def load_models(out,cfg):
    selected=json.loads((out/'selection.json').read_text());models=[]
    for seed in cfg['seeds']:
        m=ResidualLSTM(cfg['lstm']['hidden']);m.load_state_dict(torch.load(out/f'lstm_seed{seed}.pt',weights_only=True));models.append(m.eval())
    return dict(selection=selected,arx=load_arx(out/'arx.npz'),lstm=models,
                frozen={profile:FrozenAngular(profile) for profile in ['retained','candidate']})


def predictions(models,p,cfg):
    prediction=dict(zero=np.zeros_like(p['target']),persistence=np.repeat(p['theta'][:,:1],cfg['steps'],axis=1),geometric=geometric(p,cfg),
        fitted_oscillator=oscillator(p,cfg,models['selection']['oscillator_parameters']),arx_residual=predict_arx(models['arx'],p,cfg),
        lstm_residual=np.stack([predict_lstm(m,p,cfg) for m in models['lstm']]).mean(0))
    for profile,model in models['frozen'].items():prediction['cranepilot_'+profile]=model.forecast(p,cfg)
    if set(prediction)!=set(cfg['methods']):raise AssertionError('Fixed comparator set')
    if any(y.shape!=p['target'].shape or not np.isfinite(y).all() for y in prediction.values()):raise FloatingPointError('Invalid prediction')
    return prediction


def metric_rows(pred,p,cfg,stage):
    rows=[]
    for method,y in pred.items():
        error=np.rad2deg(y-p['target'])
        for j,h in enumerate(cfg['horizon_index']):
            rows.append(dict(stage=stage,record=int(p['scenario'][0]),method=method,horizon_steps=h+1,nominal_horizon_s=(h+1)*cfg['dt_s'],
                packets=len(y),mae_deg=float(np.abs(error[:,h]).mean()),rmse_deg=float(np.sqrt(np.mean(error[:,h]**2))),
                p95_absolute_error_deg=float(np.quantile(np.abs(error[:,h]),.95)),actual_horizon_median_s=float(np.median(p['actual_horizon_s'][:,h])),actual_horizon_min_s=float(p['actual_horizon_s'][:,h].min()),actual_horizon_max_s=float(p['actual_horizon_s'][:,h].max())))
    return rows


def evaluate_group(models,ids,cfg,offset,stride,out,stage):
    rows=[];qualification=[];reads=[];folder=out/stage;folder.mkdir()
    for sid in ids:
        s=load_sequence(cfg,sid,ids);reads.append(sid);p=packets(s,cfg,offset,stride);pred=predictions(models,p,cfg)
        np.savez_compressed(folder/f'record_{sid}.npz',k=p['k'],**pred)
        rows+=metric_rows(pred,p,cfg,stage)
        qualification.append(dict(stage=stage,record=sid,samples=len(s['time']),packets=len(p['k']),raw_timestamp_span_s=float(s['time'][-1]),nominal_span_s=(len(s['time'])-1)*.05,
            maximum_frame_interval_s=float(np.diff(s['time']).max()),minimum_frame_interval_s=float(np.diff(s['time']).min()),median_frame_interval_s=float(np.median(np.diff(s['time']))),
            no_future_angle_filter=True,whole_record_test_mean_used=False,input_file_simulated_yd_ignored=True))
        print('RECORD',stage,sid,len(p['k']),flush=True)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(qualification).to_csv(folder/'qualification.csv',index=False)
    return reads


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['train','evaluate'],required=True);ap.add_argument('--output',type=Path,default=RUN);a=ap.parse_args()
    cfg=config();torch.set_num_threads(cfg['threads']);out=a.output
    locked=[CONFIG,DESIGN,Path('scripts/external_crane_data.py'),Path('scripts/external_baselines.py'),Path('scripts/run_external_benchmark.py'),Path('scripts/fetch_external_crane.py'),Path('tests/test_external_benchmark.py'),Path('scripts/verify_external_benchmark.py')]
    if a.stage=='train':
        for p in locked:
            if subprocess.check_output(['git','show',f'HEAD:{p}'])!=p.read_bytes():raise ValueError(f'Commit before fitting {p}')
        if out.exists():raise FileExistsError('Retain previous attempts')
        out.mkdir(parents=True)
        additional=list(Path('src/cranepilot').glob('*.py'))+list(Path('artifacts/research_bundle_20261107').rglob('*'))+[Path('scripts/predict_research_bundle.py'),Path('scripts/predict_cranepilot.py'),Path('scripts/run_broad_search.py'),Path('scripts/residual_adapter.py'),Path('scripts/stable_latent.py')]
        paths=locked+additional
        prov=dict(status='training',config=cfg,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),guarded_hashes={str(p):sha256(p) for p in paths if p.is_file()},
            environment=environment_record(),started_utc=pd.Timestamp.now(tz='UTC').isoformat(),external_test_seen=False,source_commit=cfg['commit'],historical_primary_records_not_reused=True)
        source_manifest=Path(cfg['cache'])/'download_manifest.json';write(out/'external_source_manifest.json',json.loads(source_manifest.read_text()));prov['external_source_manifest_sha256']=sha256(out/'external_source_manifest.json')
        write(out/'provenance.json',prov)
        try:
            sequences=[load_sequence(cfg,sid,cfg['train_ids']) for sid in cfg['train_ids']];offset=train_offset(sequences)
            train=concatenate([packets(s,cfg,offset,cfg['train_stride']) for s in sequences])
            validation=concatenate([packets(load_sequence(cfg,sid,cfg['validation_ids']),cfg,offset,cfg['evaluation_stride']) for sid in cfg['validation_ids']])
            parameters,fits=fit_oscillator(train,cfg);write(out/'oscillator_fits.json',fits);print('OSCILLATOR',parameters,flush=True)
            arx_scores=[];best=None
            for alpha in cfg['arx_ridge']:
                model=fit_arx(train,cfg,alpha);value=score(predict_arx(model,validation,cfg),validation,cfg);arx_scores.append(dict(alpha=alpha,validation_mae_deg=value))
                if best is None or value<best[0]:best=(value,alpha,model)
            save_arx(out/'arx.npz',best[2]);meta=[];logs=[]
            for seed in cfg['seeds']:
                started=time.perf_counter();m,history,info=fit_lstm(train,validation,cfg,seed);torch.save(m.state_dict(),out/f'lstm_seed{seed}.pt')
                info['fit_seconds']=time.perf_counter()-started;meta.append(info);logs.extend(dict(seed=seed,**row) for row in history);pd.DataFrame(logs).to_csv(out/'training.csv',index=False)
            selected=dict(offset_rad=offset,offset_source='equal-record mean of TRAIN raw-camera tangential angle only',oscillator_parameters=parameters,arx_alpha=best[1],arx_validation=arx_scores,lstm=meta,ensemble_all_three_seeds=True,test_ids_sealed=cfg['test_ids'],test_used_for_selection=False)
            write(out/'selection.json',selected);models=load_models(out,cfg)
            evaluate_group(models,cfg['validation_ids'],cfg,offset,cfg['evaluation_stride'],out,'validation')
            prov.update(status='selection_complete',selection_sha256=sha256(out/'selection.json'),records_parsed_before_test=cfg['train_ids']+cfg['validation_ids'],selection_locked_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                training_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
        except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
        finally:write(out/'provenance.json',prov)
    else:
        prov=json.loads((out/'provenance.json').read_text());assert prov['status']=='selection_complete'
        verify_hashes(prov['guarded_hashes']);verify_hashes(prov['training_hashes'],out)
        try:
            models=load_models(out,cfg);reads=evaluate_group(models,cfg['test_ids'],cfg,models['selection']['offset_rad'],cfg['evaluation_stride'],out,'test')
            prov.update(status='complete',test_records_read=reads,external_test_seen=True,test_predictions_after_selection_lock=True,no_test_refitting=True,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                result_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
        except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
        finally:write(out/'provenance.json',prov)
    print('COMPLETE',a.stage,flush=True)

if __name__=='__main__':main()
