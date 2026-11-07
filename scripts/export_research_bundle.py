"""Copy selected research weights into a small independently loadable bundle.

Export/replay reads local frozen runs and data. The resulting inference loader
needs neither. No training, git mutation, network access or archive upload.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch

from cranepilot.data import load_dataset
from scripts.predict_research_bundle import (ResearchBundle,DEFAULT_BUNDLE,PROJECT_ROOT,
                                            FOLDS,PROFILES,file_sha256)


STRUCTURED=Path('artifacts/runs/structured_models_20261107')
FINETUNE=Path('artifacts/runs/finetune_campaign_20261107')
GLOBAL=Path('artifacts/runs/global_research_20261107')
TEMPORAL=Path('artifacts/runs/temporal_20261106')
TOLERANCE_DEG=5e-5


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _verify(mapping,base=PROJECT_ROOT):
    for name,expected in mapping.items():
        path=Path(base)/name.replace('\\','/')
        if file_sha256(path)!=expected:raise ValueError(f'Changed frozen source: {path}')


def _completed(path):
    provenance=_read(path/'provenance.json')
    if provenance['status']!='complete':raise ValueError(f'Incomplete parent: {path}')
    _verify(provenance['guarded_hashes']);_verify(provenance['checkpoint_hashes'],path)
    if 'result_hashes' in provenance:_verify(provenance['result_hashes'],path)
    if 'selection_sha256' in provenance:_verify({'selection.json':provenance['selection_sha256']},path)
    return provenance


def _code_files():
    scripts=['predict_research_bundle.py','predict_cranepilot.py','run_broad_search.py',
             'run_cleanroom.py','run_temporal.py','residual_adapter.py','stable_latent.py',
             'finetune_strategies.py','global_delay_lift.py']
    return [*sorted(Path('src/cranepilot').glob('*.py')),*[Path('scripts')/name for name in scripts]]


def export_bundle(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Retain previous exported bundle; choose a new output directory')
    structured=_completed(STRUCTURED);finetune=_completed(FINETUNE)
    global_run=_completed(GLOBAL);temporal=_completed(TEMPORAL)
    selected=_read(STRUCTURED/'selection.json');retained=_read(FINETUNE/'selection.json')
    older=_read(GLOBAL/'selection.json')
    expected_new={'primary':'adapter_direct8','short_transfer':'previous_local','long_transfer':'latent_residual6'}
    expected_old={'primary':'phase_space','short_transfer':'previous_local','long_transfer':'incumbent'}
    if {f:selected['folds'][f]['model'] for f in FOLDS}!=expected_new:
        raise ValueError('Structured selection differs from registered bundle recipe')
    if {f:retained['folds'][f]['model'] for f in FOLDS}!=expected_old:
        raise ValueError('Retained selection differs from registered bundle recipe')
    if older['folds']['short_transfer']['model']!='lpv_linear_actor_r1':
        raise ValueError('Expected retained short-transfer LPV actor alpha .01')
    metric=pd.read_csv(STRUCTURED/'evaluate_metrics.csv')
    load=metric[(metric.fold=='primary')&(metric.group=='load_stress')&(metric.regime=='all')]
    load_values=load.groupby(['model','scenario']).mae_deg.mean().groupby('model').mean()
    before=float(load_values['previous_local']);after=float(load_values['adapter_direct8'])
    if after/before<=1.01:raise ValueError('Bundle caution must be updated if adoption evidence changes')
    output.mkdir(parents=True)
    files={};sources={};folds={}
    def copy_weight(source,relative):
        source=Path(str(source).replace('\\','/'));target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True);digest=file_sha256(source)
        shutil.copyfile(source,target)
        if file_sha256(target)!=digest:raise AssertionError('Copied weight checksum mismatch')
        files[relative]=digest;sources[relative]={'path':str(source),'sha256':digest}
        return relative
    seeds=structured['config']['seeds']
    if seeds!=[0,1,2]:raise ValueError('Expected three original seeds')
    for fold in FOLDS:
        info=temporal['inputs'][fold]
        actor=copy_weight(info['actor'],f'{fold}/actor.pt')
        split=temporal['config']['folds'][fold]
        if sorted(item['sid'] for item in info['calibration']['evidence'])!=sorted(split['train']):
            raise ValueError('Calibration does not match fold TRAIN')
        if fold=='primary':
            old_spec=dict(architecture='soft_strategy',method='phase_space',seeds=seeds,
                weights=[copy_weight(FINETUNE/fold/f'phase_space_seed{s}.pt',f'{fold}/retained_seed{s}.pt') for s in seeds])
        elif fold=='long_transfer':
            old_spec=dict(architecture='innovation',method='incumbent',seeds=seeds,
                weights=[copy_weight(path,f'{fold}/retained_seed{s}.pt') for s,path in zip(seeds,info['weights'])])
        else:
            old_spec=dict(architecture='physical_plus_lpv',method='lpv_linear_actor_r1',seeds=[],
                weights=[copy_weight(GLOBAL/fold/'lpv_linear_actor_r1.npz',f'{fold}/retained_lpv.npz')])
        if fold=='short_transfer':new_spec=copy.deepcopy(old_spec)
        else:
            name=expected_new[fold];choice=structured['config']['variants'][name]
            new_spec=dict(architecture='residual_adapter' if choice['family']=='adapter' else 'stable_latent',
                          method=name,choice=choice,seeds=seeds,
                          weights=[copy_weight(STRUCTURED/fold/f'{name}_seed{s}.pt',f'{fold}/candidate_seed{s}.pt') for s in seeds])
        folds[fold]=dict(actor=actor,calibration=info['calibration'],
                        split={key:split[key] for key in ['train','validation','evaluation','load_stress']},
                        profiles={'retained':old_spec,'candidate':new_spec})
    manifest=dict(schema_version=1,created_date='2026-11-07',default_profile='retained',
                  scope='Retrospective fixed-history forecasting research; not a controller or automatic length router.',
                  fold_argument_required=True,candidate_promoted=False,
                  input=dict(theta='N by 41 radians, newest first at 10 ms',x='N by 41 metres, newest first',
                             reference='N by 41 recorded native command scale, newest first',
                             length_m='N positive fixed cable lengths in metres',
                             future_reference='N by 30 prescribed native commands at 50 ms, includes matching origin command'),
                  output=dict(prediction_rad='N by 30 at .05,...,1.50 seconds',
                              seed_std_rad='descriptive three-seed spread, not calibrated uncertainty; absent for LPV'),
                  candidate_gate=dict(promoted=False,passed=False,
                      reason='Primary candidate load-stress all-regime MAE exceeds retained by more than the allowed 1%; candidate is explicit opt-in.',
                      primary_load_retained_mae_deg=before,primary_load_candidate_mae_deg=after,
                      ratio=after/before,maximum_ratio=1.01),
                  folds=folds,file_hashes=files,source_weights=sources,
                  code_hashes_lf={str(path):file_sha256(path,canonical_source=True) for path in _code_files()},
                  code_checksum_convention='SHA256 after CRLF to LF normalization; no other content transformation',
                  source_provenance_hashes={str(path/'provenance.json'):file_sha256(path/'provenance.json') for path in [STRUCTURED,FINETUNE,GLOBAL,TEMPORAL]},
                  export_git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  environment=dict(python=platform.python_version(),platform=platform.platform(),
                                   torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__),
                  verification={'status':'not_replayed'},
                  data_required_for_inference=False,ancestral_runs_required_for_inference=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return manifest


def verify_bundle(output):
    """Replay both profiles on all18 saved real records, using history only."""
    output=Path(output);manifest=_read(output/'manifest.json')
    data=load_dataset(Path('data/processed'));checks=[];packets=0;records=0
    torch.set_num_threads(4)
    for fold in FOLDS:
        models={profile:ResearchBundle(output,fold=fold,profile=profile) for profile in PROFILES}
        for path in sorted((STRUCTURED/fold).glob('predictions_s*.npz')):
            sid=int(path.stem.split('_s')[-1]);scenario=data[sid]
            with np.load(path,allow_pickle=False) as saved:
                k=saved['k'];past=k[:,None]-np.arange(41);future=k[:,None]+5*np.arange(30)
                packet=dict(theta=scenario.signals['theta'][past],x=scenario.signals['x'][past],
                            reference=scenario.signals['xr'][past],length_m=scenario.signals['L'][k],
                            future_reference=scenario.signals['xr'][future])
                for profile,model in models.items():
                    expected_name='previous_local' if profile=='retained' else manifest['folds'][fold]['profiles']['candidate']['method']
                    if fold=='short_transfer':expected_name='previous_local'
                    result=model.forecast(packet)
                    error=float(np.abs(np.rad2deg(result['prediction_rad']-saved[expected_name])).max())
                    actor_error=float(np.abs(result['acceleration_m_s2']-saved['acceleration']).max())
                    if error>TOLERANCE_DEG or actor_error>1e-6:
                        raise AssertionError(('bundle_replay',fold,sid,profile,error,actor_error))
                    checks.append(dict(fold=fold,scenario=sid,profile=profile,method=expected_name,
                                       packets=len(k),max_error_deg=error,max_acceleration_error_m_s2=actor_error,
                                       reference_sha256=file_sha256(path)))
            records+=1;packets+=len(k)
            print(f'BUNDLE REPLAY {fold} scenario={sid} profiles=2 packets={len(k)}',flush=True)
    if records!=18 or len(checks)!=36:raise AssertionError('Expected18 real files and36 profile comparisons')
    report=dict(passed=True,records=records,unique_packets=packets,profile_comparisons=len(checks),
                max_error_deg=max(item['max_error_deg'] for item in checks),tolerance_deg=TOLERANCE_DEG,
                history_only_inputs=True,checks=checks)
    report_path=output/'verification.json'
    if report_path.exists():raise FileExistsError('Retain previous bundle verification')
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest['file_hashes']['verification.json']=file_sha256(report_path)
    manifest['verification']={key:value for key,value in report.items() if key!='checks'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--verify',action='store_true',help='Replay all18 retained records before publishing the bundle')
    args=parser.parse_args()
    if Path.cwd().resolve()!=PROJECT_ROOT:raise ValueError('Run from the main project root')
    manifest=export_bundle(args.output)
    if args.verify:
        report=verify_bundle(args.output)
        print(json.dumps({key:value for key,value in report.items() if key!='checks'}),flush=True)
    print(f'Exported {len(manifest["file_hashes"])} model files to {args.output}; retained is default.',flush=True)


if __name__=='__main__':main()
