"""Independently replay structured models and audit frozen bases/stable operators.

Uses full history-only inputs for every retained real packet. No fitting occurs.
Every run and original data/source hash is checked before checkpoint loading.
The latent stability audit does not certify the plant or the composed controller.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import numpy as np
from cranepilot.data import load_dataset, sha256
from scripts.verify_finetune_campaign import (
    PROJECT_ROOT, TOLERANCE_DEG, ACCELERATION_TOLERANCE,
    utc_now, verify_mapping, original_files, finite,
)


def audit_stability(model, frequencies):
    """Double-precision operator checks, not inferred closed-loop guarantees."""
    import torch
    from scipy.linalg import expm
    from scripts.stable_latent import StableLatentForecaster
    from scripts.residual_adapter import oscillator_transition
    probe = copy.deepcopy(model).double().eval()
    omega = torch.as_tensor(np.unique(frequencies), dtype=torch.float64)
    if isinstance(probe, StableLatentForecaster):
        with torch.no_grad():
            transition, port = probe.discrete_operators(omega)
            generator = probe.generator()
            damping = torch.nn.functional.softplus(probe.damping_raw)+1e-3
            identity = torch.eye(probe.latent_dim, dtype=torch.float64)[None]
            left = identity-.025*omega[:, None, None]*generator
            inverse = torch.linalg.inv(left)
            observed = identity-transition.transpose(1, 2)@transition
            expected = inverse.transpose(1, 2)@(.1*omega[:, None, None]*torch.diag(damping))@inverse
            identity_error = float((observed-expected).abs().max())
            singular = float(torch.linalg.svdvals(transition)[:, 0].max())
            spectral = float(torch.linalg.eigvals(transition).abs().max())
            symmetric = float(torch.linalg.eigvalsh(.5*(generator+generator.T)).max())
            min_margin = float(torch.linalg.eigvalsh(observed).min())
            if not (identity_error < 1e-11 and singular < 1 and spectral < 1
                    and symmetric < 0 and min_margin > 0 and torch.isfinite(port).all()):
                raise AssertionError('Saved latent operator fails independent contraction audit')
            input_norm = float(torch.linalg.svdvals(port)[:, 0].max())
            readout_norm = float(torch.linalg.vector_norm(probe.readout_anchor+probe.readout_delta))
        return dict(passed=True, kind='latent_cayley', omega_values=omega.tolist(),
                    max_transition_singular_value=singular, max_transition_spectral_radius=spectral,
                    max_generator_symmetric_eigenvalue=symmetric, min_contraction_matrix_eigenvalue=min_margin,
                    contraction_identity_max_abs_error=identity_error,
                    max_discrete_input_operator_norm=input_norm, readout_norm=readout_norm,
                    latent_homogeneous_only=True, plant_or_controller_certificate=False)
    if probe.correction == 'direct':
        return dict(passed=True, kind='direct_bounded_head', recurrence_present=False,
                    head_bound_radians=probe.bound, plant_or_controller_certificate=False)
    with torch.no_grad():
        transition, port = oscillator_transition(omega, probe.damping_ratio)
        generator = np.array([[0., 1.], [-1., -2*probe.damping_ratio]])
        reference = np.stack([expm(w*.05*generator) for w in omega.numpy()])
        ref_port = np.column_stack([1-reference[:, 0, 0], -reference[:, 1, 0]])
        error = float(np.abs(transition.numpy()-reference).max())
        port_error = float(np.abs(port.numpy()-ref_port).max())
        singular = float(torch.linalg.svdvals(transition)[:, 0].max())
        spectral = float(torch.linalg.eigvals(transition).abs().max())
        if not (error < 1e-11 and port_error < 1e-11 and singular <= 1+1e-12 and spectral < 1):
            raise AssertionError('Saved adapter filter fails independent exact-flow audit')
    return dict(passed=True, kind='fixed_damped_oscillator', omega_values=omega.tolist(),
                scipy_expm_max_abs_error=error, scipy_input_port_max_abs_error=port_error,
                max_transition_singular_value=singular, max_transition_spectral_radius=spectral,
                head_bound_radians=probe.bound, filter_homogeneous_only=True,
                bound_applies_to_head_not_filtered_angle=True, plant_or_controller_certificate=False)


def audit_saved_parameters(run, provenance):
    """Reconstruct initialization with exact factory RNG; audit all 72 states."""
    import torch
    from cranepilot.physics import LengthCalibration
    from scripts.structured_candidate import make_structured_model
    cfg = provenance['config']; broad = PROJECT_ROOT/cfg['broad_run']
    broad_path = broad/'provenance.json'; broad_prov = json.loads(broad_path.read_text())
    if broad_prov['status'] != 'complete':raise ValueError('Completed warm-start research required')
    expected_keys = {(fold, name, int(seed)) for fold in cfg['folds']
                     for name in cfg['variants'] for seed in cfg['seeds']}
    if len(expected_keys) != 72:raise ValueError('Expected seventy-two structured models')
    fits = json.loads((run/'fits.json').read_text()); fit_by_key = {}
    for row in fits:
        key = (row['fold'], row['model'], int(row['seed']))
        if key in fit_by_key:raise ValueError('Duplicate fit metadata')
        fit_by_key[key] = row
    if set(fit_by_key) != expected_keys:raise ValueError('Fit metadata does not cover all models')
    warm_paths = {f'{fold}/soft_symmetry_seed{seed}.pt' for fold in cfg['folds'] for seed in cfg['seeds']}
    warm_integrity = verify_mapping({name:broad_prov['checkpoint_hashes'][name] for name in warm_paths}, broad)
    if not warm_integrity['passed']:raise AssertionError('Warm-start checkpoints changed')
    source = json.loads((PROJECT_ROOT/cfg['source_run']/'provenance.json').read_text())
    data = load_dataset(PROJECT_ROOT/'data/processed'); frequencies = {}
    for fold in cfg['folds']:
        split = source['config']['folds'][fold]
        ids = set(split['train']+split['validation']+split['evaluation']+split['load_stress'])
        lengths = [float(data[sid].signals['L'][0]) for sid in sorted(ids)]+cfg['simulation']['length_m']
        calibration = LengthCalibration(**source['inputs'][fold]['calibration'])
        # Include every real/simulation nominal frequency and a wider operator probe.
        frequencies[fold] = np.unique(np.r_[calibration.omega(lengths), [.01, .1, .5, 1, 2, 3, 4, 6, 10, 20, 40, 70],
                                            np.geomspace(.1,100,32)])
    models=[]; frozen_checks=0; fallback_checks=0; buffer_checks=0
    for fold, name, seed in sorted(expected_keys):
        choice = cfg['variants'][name]; meta = fit_by_key[(fold, name, seed)]
        selected_epoch = int(meta['selected_epoch'])
        if selected_epoch < -1 or int(meta['best_epoch']) != selected_epoch:
            raise AssertionError('Inconsistent selected epoch metadata')
        if selected_epoch >= int(meta['epochs_executed']):raise AssertionError('Selected nonexistent epoch')
        torch.manual_seed(seed)
        model = make_structured_model(cfg, fold, seed, choice)
        reference = copy.deepcopy(model.state_dict())
        parameters_by_name = dict(model.named_parameters())
        buffers_by_name = dict(model.named_buffers())
        trainable = {key for key, value in parameters_by_name.items() if value.requires_grad}
        if set(meta['trainable_parameter_names']) != trainable:
            raise AssertionError('Trainable-name metadata disagrees with architecture')
        total_count = sum(value.numel() for value in parameters_by_name.values())
        trainable_count = sum(parameters_by_name[key].numel() for key in trainable)
        if total_count != meta['total_parameter_count'] or trainable_count != meta['trainable_parameter_count']:
            raise AssertionError('Parameter metadata counts differ; buffers are excluded')
        current = torch.load(run/fold/f'{name}_seed{seed}.pt', map_location='cpu', weights_only=True)
        if current.keys() != reference.keys():raise AssertionError('Saved state key mismatch')
        additive = choice['family'] == 'adapter' or choice.get('mode') == 'residual'
        if additive:
            source_state = torch.load(broad/fold/f'soft_symmetry_seed{seed}.pt', map_location='cpu', weights_only=True)
            if {key[5:] for key in current if key.startswith('base.')} != set(source_state):
                raise AssertionError('Saved base coverage differs from frozen source')
            for key, value in source_state.items():
                if not torch.equal(current['base.'+key], value):raise AssertionError('Frozen soft base changed')
        elif any(key.startswith('base.') for key in current):
            raise AssertionError('Standalone model unexpectedly includes a base')
        rows=[]; changed=0; sum_sq=0.; frozen=0; fallback=0; buffers=0
        for key in sorted(reference):
            before=reference[key]; after=current[key]
            if not isinstance(after, torch.Tensor) or after.shape != before.shape or after.dtype != before.dtype:
                raise AssertionError('Checkpoint tensor shape/dtype differs from factory')
            if not torch.isfinite(after).all() or not torch.isfinite(before).all():raise AssertionError('Nonfinite saved tensor')
            exact = bool(torch.equal(before, after)); is_frozen = key not in trainable
            is_buffer = key in buffers_by_name
            if is_buffer:buffers += 1
            if is_frozen:
                frozen += 1
                if not exact:raise AssertionError(f'Frozen parameter or buffer changed: {fold}/{name}/{seed}/{key}')
            if selected_epoch == -1:
                fallback += 1
                if not exact:raise AssertionError(f'Epoch -1 differs from seeded factory: {fold}/{name}/{seed}/{key}')
            delta = (after.double()-before.double()).reshape(-1)
            squared = float(torch.dot(delta, delta)); sum_sq += squared
            count = int(torch.count_nonzero(before != after)); changed += count
            rows.append(dict(name=key, shape=list(after.shape), dtype=str(after.dtype),
                             elements=after.numel(), buffer=is_buffer, frozen=is_frozen,
                             exactly_equal=exact, changed_elements=count, l2_delta=float(np.sqrt(squared))))
        model.load_state_dict(current); model.eval()
        stability = audit_stability(model, frequencies[fold])
        frozen_checks += frozen; fallback_checks += fallback; buffer_checks += buffers
        models.append(dict(fold=fold, model=name, seed=seed, family=choice['family'],
                           selected_epoch=selected_epoch, additive_frozen_base=additive,
                           checkpoint=f'{fold}/{name}_seed{seed}.pt',
                           source_checkpoint=f'{fold}/soft_symmetry_seed{seed}.pt' if additive else None,
                           total_parameter_count=total_count, trainable_parameter_count=trainable_count,
                           frozen_tensors_checked=frozen, fallback_tensors_checked=fallback,
                           buffers_checked=buffers, changed_elements=changed,
                           total_l2_delta=float(np.sqrt(sum_sq)), all_weights_finite=True,
                           stability=stability, parameters=rows))
    return dict(passed=True, models_checked=len(models), warm_start_integrity=warm_integrity,
                warm_start_provenance_sha256=sha256(broad_path),
                initialization_rule='torch.manual_seed(seed), then exact make_structured_model factory',
                comparison='torch.equal; drift in float64; buffers audited separately from parameter counts',
                frozen_tensor_checks=frozen_checks, fallback_tensor_checks=fallback_checks,
                buffer_checks=buffer_checks, stability_checks=len(models),
                initial_checkpoint_selections=sum(m['selected_epoch']==-1 for m in models),
                unchanged_models=sum(m['changed_elements']==0 for m in models),
                latent_homogeneous_contraction_checks=sum(m['family']=='latent' for m in models),
                max_total_l2_delta=max(m['total_l2_delta'] for m in models), models=models)


def replay(run,provenance,selection,report):
    import torch
    from cranepilot.rollout import command_windows
    from scripts.structured_candidate import StructuredCandidate, CONTROLS
    from scripts.predict_cranepilot import build_packet
    torch.set_num_threads(4)
    cfg=provenance['config'];variants=set(cfg['variants']);seeds=list(cfg['seeds'])
    if len(cfg['folds'])!=3 or len(set(cfg['folds']))!=3 or len(variants)!=8 or len(seeds)!=3 or len(set(seeds))!=3:
        raise ValueError('Expected fixed three folds, three seeds and eight structured variants')
    expected_models=set(CONTROLS)|variants
    if len(expected_models)!=12:raise ValueError('Expected twelve real ensemble/control forecasts')
    expected_checkpoints={f'{fold}/{name}_seed{seed}.pt' for fold in cfg['folds'] for name in variants for seed in seeds}
    if set(provenance['checkpoint_hashes'])!=expected_checkpoints:raise ValueError('Checkpoint coverage differs from seventy-two frozen fits')
    source=json.loads((PROJECT_ROOT/cfg['source_run']/'provenance.json').read_text())
    if source['status']!='complete':raise ValueError('Completed split source required')
    data=load_dataset(PROJECT_ROOT/'data/processed')
    report.update(folds=list(cfg['folds']),seeds=seeds,variants=sorted(variants),fitted_models_total=72,
                  real_methods_per_fold=12,history_only_inputs=True,history_samples=41,prediction_steps=30,
                  target_comparison='exact_array_equal',records=[],checks=[])
    for fold in cfg['folds']:
        folder=run/fold;split=source['config']['folds'][fold];groups={}
        for group in ('validation','evaluation','load_stress'):
            for sid in split[group]:
                if sid in groups:raise ValueError(f'Repeated scenario {fold}/{sid}')
                groups[sid]=group
        by_sid={}
        for path in folder.glob('predictions_s*.npz'):
            match=re.fullmatch(r'predictions_s([0-9]+)\.npz',path.name)
            if match is None:raise ValueError(f'Unexpected real prediction filename: {path}')
            sid=int(match.group(1))
            if sid in by_sid:raise ValueError('Repeated retained scenario')
            by_sid[sid]=path
        if set(by_sid)!=set(groups):raise AssertionError(f'Real-record coverage mismatch in {fold}')
        predictor=StructuredCandidate(cfg,fold,folder)
        if set(predictor.ensembles)!=variants:raise AssertionError('Loaded ensemble set differs from fixed variants')
        for sid in sorted(by_sid):
            path=by_sid[sid];relative=str(path.relative_to(run));scenario=data[sid]
            if relative not in provenance['result_hashes']:raise AssertionError('Prediction file lacks result hash')
            with np.load(path,allow_pickle=False) as saved:
                k=saved['k']
                if k.ndim!=1 or not len(k) or not np.issubdtype(k.dtype,np.integer):raise AssertionError('Integer origins required')
                expected_k=np.arange(40,len(scenario.time)-150,cfg['evaluation_stride'])
                np.testing.assert_array_equal(k,expected_k,err_msg=relative)
                n=len(k);raw=command_windows(scenario,predictor.calibration,k)
                target=finite(saved['target'],(n,30),relative+'/target')
                np.testing.assert_array_equal(target,raw['target_q'],err_msg=relative)
                saved_acc=finite(saved['acceleration'],(n,30),relative+'/acceleration')
                past=k[:,None]-np.arange(41);future=k[:,None]+np.arange(30)*5
                packet=build_packet(scenario.signals['theta'][past],scenario.signals['x'][past],
                                    scenario.signals['xr'][past],scenario.signals['L'][k],
                                    scenario.signals['xr'][future],predictor.calibration)
                actual,seed_predictions,acceleration=predictor.forecast(packet)
                if set(actual)!=expected_models or set(seed_predictions)!=variants:raise AssertionError('Forecast model sets changed')
                acceleration=finite(acceleration,(n,30),'replayed acceleration')
                acc_error=float(np.max(np.abs(acceleration-saved_acc)))
                if acc_error>ACCELERATION_TOLERANCE:raise AssertionError(f'Actuator replay differs: {acc_error}')
                aliases=set() if groups[sid]=='validation' else {'selected','common'}
                expected_fields={'k','target','acceleration'}|expected_models|aliases|{name+'__seeds' for name in variants}
                if set(saved.files)!=expected_fields:raise AssertionError(f'Saved field coverage differs: {relative}')
                resolved={'selected':selection['folds'][fold]['model'],'common':selection['common']['model']}
                for alias in aliases:
                    if resolved[alias] not in actual:raise AssertionError('Selected method missing from fixed predictions')
                    actual[alias]=actual[resolved[alias]]
                maximum=0.;ensemble_count=0;seed_count=0
                for name in sorted(expected_models|aliases):
                    expected=finite(saved[name],(n,30),relative+'/'+name)
                    predicted=finite(actual[name],(n,30),'replayed '+name)
                    error=float(np.max(np.abs(np.rad2deg(predicted-expected))))
                    if error>TOLERANCE_DEG:raise AssertionError(f'{relative}/{name}: {error} degrees')
                    maximum=max(maximum,error);ensemble_count+=1
                    report['checks'].append(dict(kind='ensemble_or_control',fold=fold,scenario=sid,group=groups[sid],
                        model=name,resolved_model=resolved.get(name,name),packets=n,max_error_deg=error))
                for name in sorted(variants):
                    expected=finite(saved[name+'__seeds'],(len(seeds),n,30),relative+'/'+name+'__seeds')
                    predicted=finite(seed_predictions[name],(len(seeds),n,30),'replayed '+name+' seeds')
                    np.testing.assert_array_equal(actual[name],predicted.mean(0),err_msg=name+' ensemble mean')
                    np.testing.assert_array_equal(saved[name],expected.mean(0),err_msg=name+' retained ensemble mean')
                    for index,seed in enumerate(seeds):
                        error=float(np.max(np.abs(np.rad2deg(predicted[index]-expected[index]))))
                        if error>TOLERANCE_DEG:raise AssertionError(f'{relative}/{name}/seed{seed}: {error} degrees')
                        maximum=max(maximum,error);seed_count+=1
                        report['checks'].append(dict(kind='individual_seed',fold=fold,scenario=sid,group=groups[sid],
                            model=name,seed=seed,packets=n,max_error_deg=error))
                report['records'].append(dict(fold=fold,scenario=sid,group=groups[sid],file=relative,packets=n,
                    ensemble_comparisons=ensemble_count,seed_comparisons=seed_count,targets_exact=True,
                    ensemble_means_exact=True,all_shapes_finite=True,max_error_deg=maximum,
                    acceleration_max_abs_difference_m_s2=acc_error))
            print(f'REPLAY {fold} scenario={sid} packets={n} ensembles={ensemble_count} seeds={seed_count} max_deg={maximum:.9g}',flush=True)
    report.update(real_record_files=len(report['records']),packet_count=sum(r['packets'] for r in report['records']),
                  prediction_comparisons=len(report['checks']),
                  ensemble_prediction_comparisons=sum(r['ensemble_comparisons'] for r in report['records']),
                  individual_seed_comparisons=sum(r['seed_comparisons'] for r in report['records']),
                  max_error_deg=max(r['max_error_deg'] for r in report['checks']))


def main():
    os.chdir(PROJECT_ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/runs/structured_models_20261107'))
    parser.add_argument('--output',type=Path,default=Path('results/structured_models_20261107/replay_verification.json'))
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Preserve previous verification; choose a fresh output')
    provenance_path=run/'provenance.json';provenance=json.loads(provenance_path.read_text())
    if provenance['status']!='complete':raise ValueError('Completed training and evaluation required')
    report=dict(passed=False,run=str(run),started_utc=utc_now(),tolerance_deg=TOLERANCE_DEG,relative_tolerance=0.,
                acceleration_tolerance_m_s2=ACCELERATION_TOLERANCE,fitting_performed=False,
                verifier_sha256=sha256(Path(__file__)),provenance_sha256=sha256(provenance_path))
    try:
        integrity={key:verify_mapping(provenance[key],base) for key,base in
                   [('guarded_hashes',PROJECT_ROOT),('checkpoint_hashes',run),('result_hashes',run)]}
        integrity['selection']=verify_mapping({'selection.json':provenance['selection_sha256']},run)
        parent=PROJECT_ROOT/provenance['config']['parent_run']
        parent_provenance=json.loads((parent/'provenance.json').read_text())
        if parent_provenance['status']!='complete':raise AssertionError('Parent run incomplete')
        integrity['parent_checkpoints']=verify_mapping(parent_provenance['checkpoint_hashes'],parent)
        integrity['parent_selection']=verify_mapping({'selection.json':parent_provenance['selection_sha256']},parent)
        # The parent previous_local includes retained LPV matrices from the grandparent.
        grandparent=PROJECT_ROOT/parent_provenance['config']['parent_run']
        grandparent_provenance=json.loads((grandparent/'provenance.json').read_text())
        if grandparent_provenance['status']!='complete':raise AssertionError('Grandparent run incomplete')
        integrity['grandparent_guarded']=verify_mapping(grandparent_provenance['guarded_hashes'],PROJECT_ROOT)
        integrity['grandparent_checkpoints']=verify_mapping(grandparent_provenance['checkpoint_hashes'],grandparent)
        integrity['grandparent_selection']=verify_mapping({'selection.json':grandparent_provenance['selection_sha256']},grandparent)
        report['integrity']=integrity;report['original_files']=original_files()
        if not all(v['passed'] for v in integrity.values()):raise AssertionError('Retained evidence hash verification failed')
        if not report['original_files']['passed']:raise AssertionError('Original main/APHYNITY files changed')
        print('HASHES verified; original files main=1111 aphynity=170',flush=True)
        report['parameter_audit']=audit_saved_parameters(run,provenance)
        print('PARAMETERS verified; 72 states, frozen tensors, buffers, epoch -1 and stable operators',flush=True)
        selection=json.loads((run/'selection.json').read_text());replay(run,provenance,selection,report)
        if report['real_record_files']!=18:raise AssertionError('Expected all eighteen real records')
        report['passed']=True
    except Exception as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        report['finished_utc']=utc_now();output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({key:report[key] for key in ['passed','real_record_files','packet_count','prediction_comparisons',
          'ensemble_prediction_comparisons','individual_seed_comparisons','max_error_deg']},indent=2),flush=True)


if __name__=='__main__':main()

