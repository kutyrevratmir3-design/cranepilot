"""History-only inference from the small, self-contained research weight bundle.

No dataset or artifacts/runs directory is read.  Folds are explicit research
splits, not a deployment router.  The retained profile is the default; the
candidate profile is exploratory and has not passed the load adoption gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cranepilot.broad_strategies import make_model
from cranepilot.forced_observer import refine_windows,analytic_rollout
from cranepilot.innovation import InnovationForecaster
from cranepilot.physics import LengthCalibration
from cranepilot.rollout import NeuralActuator,tensors,HORIZON_INDEX,HORIZONS_S
from scripts.global_delay_lift import load_model,predict_residual
from scripts.predict_cranepilot import build_packet
from scripts.residual_adapter import make_adapter
from scripts.run_broad_search import inputs,predict
from scripts.stable_latent import make_stable_latent


PROJECT_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE=PROJECT_ROOT/'artifacts/research_bundle_20261107'
FOLDS=('primary','short_transfer','long_transfer')
PROFILES=('retained','candidate')


def file_sha256(path, *, canonical_source=False):
    content=Path(path).read_bytes()
    if canonical_source:content=content.replace(b'\r\n',b'\n')
    return hashlib.sha256(content).hexdigest()


def bundle_file(directory, relative):
    """Manifest paths stay inside the copied bundle, including symlink targets."""
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Bundle entries must be relative paths within the bundle')
    root=Path(directory).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Bundle entry escapes bundle directory')
    return path


def _torch_state(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state or not all(isinstance(value,torch.Tensor) for value in state.values()):
        raise ValueError('Expected a tensor-only state_dict')
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite bundled state_dict')
    return state


def _model(spec):
    architecture=spec['architecture']
    if architecture=='innovation':return InnovationForecaster(driven=True)
    if architecture=='soft_strategy':return make_model({'symmetry_mix':.5})
    if architecture=='residual_adapter':
        return make_adapter(make_model({'symmetry_mix':.5}),spec['choice'])
    if architecture=='stable_latent':
        base=make_model({'symmetry_mix':.5}) if spec['choice']['mode']=='residual' else None
        return make_stable_latent(spec['choice'],base)
    raise ValueError(f'Unsupported bundled architecture: {architecture}')


class ResearchBundle:
    def __init__(self,directory=DEFAULT_BUNDLE,*,fold,profile='retained'):
        if fold not in FOLDS:raise ValueError('Choose an explicit research fold')
        if profile not in PROFILES:raise ValueError('Profile must be retained or candidate')
        self.directory=Path(directory).resolve();self.fold=fold;self.profile=profile
        manifest_path=self.directory/'manifest.json'
        self.manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('schema_version')!=1:raise ValueError('Unsupported research bundle schema')
        if self.manifest.get('default_profile')!='retained':raise ValueError('Unexpected bundle default profile')
        self.manifest_sha256=file_sha256(manifest_path)
        for name,expected in self.manifest['file_hashes'].items():
            if file_sha256(bundle_file(self.directory,name))!=expected:
                raise ValueError(f'Bundled file changed: {name}')
        for name,expected in self.manifest['code_hashes_lf'].items():
            if file_sha256(bundle_file(PROJECT_ROOT,name),canonical_source=True)!=expected:
                raise ValueError(f'Inference source differs from bundle: {name}')
        spec=self.manifest['folds'][fold]
        self.calibration=LengthCalibration(**spec['calibration'])
        actor_path=bundle_file(self.directory,spec['actor'])
        if spec['actor'] not in self.manifest['file_hashes']:raise ValueError('Unhashed actor')
        self.actor=NeuralActuator();self.actor.load_state_dict(_torch_state(actor_path));self.actor.eval()
        self.spec=spec['profiles'][profile]
        self.models=[];self.delay_model=None
        for name in self.spec['weights']:
            if name not in self.manifest['file_hashes']:raise ValueError('Unhashed model')
            path=bundle_file(self.directory,name)
            if self.spec['architecture']=='physical_plus_lpv':
                if len(self.spec['weights'])!=1:raise ValueError('One LPV model expected')
                self.delay_model=load_model(path)
                if (self.delay_model.basis,self.delay_model.input_mode,self.delay_model.alpha)!=('lpv_linear','actor',.01):
                    raise ValueError('Unexpected retained LPV specification')
            else:
                model=_model(self.spec);model.load_state_dict(_torch_state(path));model.eval();self.models.append(model)
        if (self.delay_model is None and len(self.models)!=3) or (self.delay_model is not None and self.models):
            raise ValueError('Expected three neural seeds or one deterministic LPV model')

    def forecast(self,packet):
        """Read only theta/x/reference/length_m/future_reference; return all30 steps."""
        raw=build_packet(packet['theta'],packet['x'],packet['reference'],packet['length_m'],
                         packet['future_reference'],self.calibration)
        if not len(raw['omega']):raise ValueError('Nonempty history packet required')
        with torch.no_grad():acceleration=self.actor(tensors(raw))[1].numpy()
        if self.delay_model is not None:
            refined=refine_windows(raw)
            physical=analytic_rollout(refined['initial_q'],refined['initial_p'],raw['omega'],acceleration.astype(float))
            prediction=physical+predict_residual(self.delay_model,raw,acceleration.astype(float))
            spread=None
        else:
            batch=inputs(raw);batch['acceleration']=torch.as_tensor(acceleration,dtype=torch.float32)
            seeds=np.stack([predict(model,batch) for model in self.models])
            prediction=seeds.mean(0);spread=seeds.std(0)
        if prediction.shape!=(len(raw['omega']),30) or not np.isfinite(prediction).all():
            raise ValueError('Invalid bundled forecast')
        metadata=dict(fold=self.fold,profile=self.profile,method=self.spec['method'],
                      architecture=self.spec['architecture'],manifest_sha256=self.manifest_sha256,
                      fold_is_research_split_not_operational_router=True,
                      seed_spread='descriptive_not_calibrated_uncertainty' if spread is not None else 'not_available_single_deterministic_model',
                      candidate_promoted=False,closed_loop_control_evaluated=False,
                      gate_warning=self.manifest['candidate_gate'] if self.profile=='candidate' else None)
        return dict(prediction_rad=prediction,acceleration_m_s2=acceleration,seed_std_rad=spread,metadata=metadata)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--fold',choices=FOLDS,required=True)
    parser.add_argument('--profile',choices=PROFILES,default='retained')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='CSV with seven benchmark horizons')
    args=parser.parse_args();metadata_path=args.output.with_suffix('.metadata.json')
    if args.output.exists() or metadata_path.exists():raise FileExistsError('Choose fresh prediction and metadata paths')
    torch.set_num_threads(4)
    model=ResearchBundle(args.bundle,fold=args.fold,profile=args.profile)
    with np.load(args.input,allow_pickle=False) as packet:result=model.forecast(packet)
    prediction=np.rad2deg(result['prediction_rad'][:,HORIZON_INDEX])
    frame=pd.DataFrame({f'angle_deg_{h:g}s':prediction[:,i] for i,h in enumerate(HORIZONS_S)})
    if result['seed_std_rad'] is not None:
        spread=np.rad2deg(result['seed_std_rad'][:,HORIZON_INDEX])
        for i,h in enumerate(HORIZONS_S):frame[f'seed_std_deg_{h:g}s']=spread[:,i]
    frame['profile']=args.profile;frame['fold']=args.fold;frame['method']=result['metadata']['method']
    args.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output,index=False)
    metadata_path.write_text(json.dumps(result['metadata'],indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if args.profile=='candidate':print('Candidate is exploratory: primary load adoption gate failed; retained remains default.')
    print(f'Saved {len(frame)} forecasts and metadata; no dataset or ancestral run directory was read.')


if __name__=='__main__':main()
