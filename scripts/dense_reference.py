"""Dense TRAIN reference challenge; frozen primary and old evidence stay untouched."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cranepilot.benchmark import environment_record
from cranepilot.data import sha256
from cranepilot.dated_pilot import FrozenDatedPilot
from cranepilot.physics import LengthCalibration
from scripts import dated_forecast as dated
from scripts import drive_conditioned as drive
from scripts import latency_observer as latency
from scripts.modal_backtest_gate import sliced
from scripts.modal_history import load_allowed, modal
from scripts.modal_reference_challenge import predict as legacy_predict
from scripts.run_dated_forecast import fit_case
from scripts.run_external_benchmark import verify_hashes, write
from scripts.run_latency_observer import add_rows, balanced, case_data
from scripts.run_modal_confirmation import load_confirmation
from scripts.verify_latency_observer import model_names, same

CONFIG = Path('configs/dense_reference_20261108.json')
RUN = Path('artifacts/runs/dense_reference_20261108')
OUT = Path('results/dense_reference_20261108')
LOCKED = [CONFIG] + [Path(p) for p in [
    'docs/DENSE_REFERENCE_DESIGN_20261108.md', 'scripts/dense_reference.py',
    'scripts/verify_dense_reference.py', 'tests/test_dense_reference.py',
    'scripts/build_dense_reference_bundle.py', 'src/cranepilot/matched_dated_pilot.py',
]]


def setup():
    cfg = json.loads(CONFIG.read_text())
    if sha256(Path(cfg['candidate_bundle']) / 'manifest.json') != cfg['candidate_manifest_sha256']:
        raise ValueError('Frozen primary manifest changed')
    primary = FrozenDatedPilot(cfg['candidate_bundle'])
    if primary.manifest['selected'] != cfg['candidate_selected']:
        raise ValueError('Frozen primary changed')
    task = primary.base.pilot.cfg
    base = json.loads(Path(cfg['qualification']).read_text())
    grids = {}
    for role in ['train', 'validation', 'evaluation']:
        grids[role] = dict(base, origin_start_s=cfg['origin_start_s'],
                           origin_stride_raw=cfg[role + '_origin_stride_raw'])
    return cfg, grids, primary, task


def require_committed(paths):
    for path in paths:
        try:
            committed = subprocess.check_output(['git', 'show', f'HEAD:{path}'], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            raise ValueError(f'Commit source/config before campaign: {path}') from error
        if committed != path.read_bytes():
            raise ValueError(f'Commit source/config before campaign: {path}')


def nearest(known_ages, age):
    return min(known_ages, key=lambda value: (abs(value - age), value))


def choose_reference(scores):
    """scores[age][option] is the list of two fixed noise-condition scores."""
    out = {}
    for age, options in scores.items():
        means = {}
        for key, values in options.items():
            if len(values) != 2 or not np.isfinite(values).all():
                raise ValueError('Every reference needs both finite VAL noise conditions')
            means[key] = float(np.mean(values))
        if not means:
            raise ValueError('Empty reference pool')
        key = min(means, key=lambda option: (means[option], option))
        out[str(age)] = dict(key=key, validation_scores=means)
    return out


def pool_specs(cfg, name, new_root=RUN):
    """Describe all options explicitly; never follow the parent's selected-only route."""
    specs = {}
    for source, root in [('dense', new_root), ('dated', Path(cfg['dated_development'])),
                         ('latency', Path(cfg['latency_development']))]:
        for filename in model_names(cfg):
            key = f'augmented/{source}/{filename}'
            specs[key] = dict(kind='augmented', source=source, file=str(root / name / filename), history_samples=81)
    original = Path(cfg['drive_development']) / name
    # This grid has modal_residual, whereas the age-augmented grid has modal_age_residual.
    for representation in cfg['representations']:
        for form in ['direct', 'modal_residual']:
            for alpha in cfg['ridge_grid']:
                filename = f'arx_{representation}_{form}_{alpha:g}.npz'
                for head in cfg['head_methods']:
                    specs[f'drive4/{head}/{filename}'] = dict(kind='drive', source='original_dense4',
                        file=str(original / filename), history_samples=81, head=head)
    legacy = Path(cfg['candidate_bundle']) / 'regime' / 'legacy'
    for head in cfg['head_methods']:
        specs[f'drive3/{head}'] = dict(kind='drive', source='original_dense3',
            file=str(original / 'short_arx.npz'), history_samples=61, head=head)
        specs[f'legacy3/{head}'] = dict(kind='legacy', source='legacy3',
            file=str(legacy / name / 'short_arx.npz'), history_samples=61, head=head)
        specs[f'legacy4/{head}'] = dict(kind='legacy', source='legacy4',
            file=str(legacy / 'base' / name / 'reference_arx.npz'), history_samples=81, head=head)
        for expert in ['frozen_arx', 'active_arx']:
            specs[f'previous/{head}/{expert}'] = dict(kind='previous', source='previous_named',
                head=head, expert=expert, history_samples=None)
    if len(specs) != 304:
        raise AssertionError('Incomplete preregistered reference pool')
    return specs


def load_pool(specs):
    # Multiple adapters share one physical file; do not reload it eight times.
    return {path: drive.load(Path(path)) for path in sorted({s['file'] for s in specs.values() if 'file' in s})}


def reference_prediction(spec, models, raw, age, axis, pilot, name, cache=None):
    """Predict from past inputs only; caller never passes targets to this function."""
    cache = {} if cache is None else cache
    raw = dated.validate(raw, age, axis)
    cal = LengthCalibration(**pilot.pilot.cases[name]['calibration'])
    task = pilot.pilot.cfg
    kind = spec['kind']
    if kind == 'previous':
        return dated.arx(raw, age, axis, pilot, name, spec['head'], spec['expert'])
    model = models[spec['file']]
    if kind == 'augmented':
        if 'dated_packet' not in cache:
            cache['dated_packet'] = latency.packet(raw, axis, cal, age)
        packet = cache['dated_packet']
        if model['form'] == 'modal_age_residual':
            if 'dated_modal' not in cache:
                cache['dated_modal'] = latency.dated_modal(packet['theta'], packet['omega_sensor'], packet['age_raw'], task)
            base = cache['dated_modal']
        else:
            base = np.zeros((len(raw[axis]), 7))
        return latency.arx_prediction(model, packet, base)
    if kind not in ['drive', 'legacy']:
        raise ValueError('Unknown fixed reference kind')
    key = ('aligned', spec['head'], spec['history_samples'])
    if key not in cache:
        aligned = dated.extended_head(raw, age, axis, spec['head'], pilot, name)
        packet = latency.packet(aligned, axis, cal)
        packet.pop('age_raw'); packet.pop('omega_sensor')
        if spec['history_samples'] == 61:
            packet = sliced(packet, 0, 61, cal)
        elif spec['history_samples'] != 81:
            raise ValueError('Only original 3s/4s references are declared')
        cache[key] = packet
    packet = cache[key]
    if kind == 'legacy':
        return legacy_predict(model, packet, task)
    free_key = ('free',) + key
    if model['form'] == 'modal_residual':
        if free_key not in cache:
            cache[free_key] = modal(packet, task, 'two_mode')[0]
        free = cache[free_key]
    else:
        free = np.zeros((len(raw[axis]), 7))
    return drive.arx_predict(model, packet, free)


def all_predictions(specs, models, raw, age, axis, pilot, name):
    cache = {}
    predictions = {key: reference_prediction(spec, models, raw, age, axis, pilot, name, cache)
                   for key, spec in specs.items()}
    if not all(y.shape == (len(raw[axis]), 7) and np.isfinite(y).all() for y in predictions.values()):
        raise ValueError('Invalid option forecast; retain failed run, do not drop the option')
    return predictions


def origin_indices(samples, start_s, stride, task):
    first = max((task['history_samples'] - 1) * task['history_spacing_raw'], int(round(start_s / .01)))
    return np.arange(first, samples - max(task['horizons_raw']), stride, dtype=np.int64)


def support_audit(records, name, task, cfg):
    """Describe old/new source support without exposing raw numerical observations."""
    parent = Path(cfg['drive_development'])
    development = json.loads((parent / 'development.json').read_text())
    case = json.loads((parent / 'selection.json').read_text())['cases'][name]
    family = name.split('_')[0]
    if case['train_records'] != task['families'][family]['train'] or sorted(records) != sorted(case['train_records']):
        raise ValueError('TRAIN roles differ from original physical prior')
    old = development['qualification']
    if old['origin_start_s'] != 4 or old['origin_stride_raw'] != 25:
        raise ValueError('Unexpected original dense source grid')
    rows = []
    for sid, record in records.items():
        previous = origin_indices(len(record.time), 4., 25, task)
        current = origin_indices(len(record.time), cfg['origin_start_s'], cfg['train_origin_stride_raw'], task)
        np.testing.assert_array_equal(current, previous[previous >= 425])
        if len(current) == 0 or current.min() - 400 - max(cfg['evaluation_ages_raw']) < 0:
            raise ValueError('Genuine prehistory missing at maximum declared age')
        horizons = np.asarray(task['horizons_raw'])
        if not np.isin(current[:, None] + horizons, previous[:, None] + horizons).all():
            raise ValueError('New TRAIN targets exceed original dense target support')
        if current.min() - 400 - max(cfg['train_ages_raw']) < previous.min() - 400 or current.max() > previous.max():
            raise ValueError('New TRAIN raw histories exceed original dense raw support')
        rows.append(dict(case=name, scenario=int(sid), role='train',
            source_file=record.metadata.get('file_name'), source_sha256=record.metadata.get('sha256'),
            original_origins=previous.tolist(), augmented_origins=current.tolist(),
            original_origin_sha256=hashlib.sha256(previous.astype('<i8').tobytes()).hexdigest(),
            augmented_origin_sha256=hashlib.sha256(current.astype('<i8').tobytes()).hexdigest(),
            original_windows=len(previous), augmented_base_windows=len(current),
            learned_history_spacing_raw=int(task['history_spacing_raw']),
            learned_history_samples=int(task['history_samples']),
            augmented_angle_offsets_raw={str(age): (np.arange(task['history_samples']) * task['history_spacing_raw'] + age).tolist()
                                         for age in cfg['train_ages_raw']},
            original_only_origins=sorted(set(previous.tolist()) - set(current.tolist())),
            original_history_raw_span=[int(previous.min() - 400), int(previous.max())],
            augmented_history_raw_span=[int(current.min() - 400 - max(cfg['train_ages_raw'])), int(current.max())],
            augmented_targets_subset_original_targets=True, augmented_origins_subset_original=True))
    expected = json.loads((parent / name / 'prior_diagnostics.json').read_text())['train_windows']
    if sum(r['original_windows'] for r in rows) != expected:
        raise ValueError('Source physical-prior TRAIN window count differs')
    return rows


def source_paths(cfg, primary):
    paths = set(LOCKED + [Path(cfg['qualification']), Path('scripts/run_dated_forecast.py'),
        Path('scripts/run_latency_observer.py'), Path('scripts/drive_conditioned.py'),
        Path('scripts/modal_reference_challenge.py'), Path('scripts/modal_backtest_gate.py'),
        Path('scripts/active_qualification.py'), Path('scripts/run_modal_confirmation.py'),
        Path('scripts/run_drive_conditioned.py'), Path('scripts/run_active_qualification.py'),
        Path('scripts/run_external_benchmark.py'), Path('scripts/verify_latency_observer.py'),
        Path('scripts/verify_finetune_campaign.py'), Path('src/cranepilot/physics.py'),
        Path('src/cranepilot/data.py'),
        Path(primary.base.pilot.cfg['data']) / 'dataset.npz', Path(primary.base.pilot.cfg['data']) / 'manifest.json'])
    for base in [Path(cfg['candidate_bundle']), Path(cfg['drive_development']),
                 Path(cfg['dated_development']), Path(cfg['latency_development'])]:
        paths.update(f for f in base.rglob('*') if f.is_file())
    for source_key in ['drive_development', 'dated_development', 'latency_development']:
        root = Path(cfg[source_key])
        if sha256(root / 'development.json') != cfg['source_development_hashes'][source_key]:
            raise ValueError('Frozen source development manifest changed')
        parent = json.loads((root / 'development.json').read_text())
        if parent['status'] != 'complete':raise ValueError('Incomplete frozen source run')
        if parent['task']['families'] != primary.base.pilot.cfg['families']:
            raise ValueError('Reused reference TRAIN/VAL roles differ from the primary')
        verify_hashes(parent['guarded_hashes']); verify_hashes(parent['result_hashes'], root)
    # Guard every transitive numerical source already guarded by the frozen primary.
    old = json.loads((Path(cfg['dated_development']) / 'development.json').read_text())
    paths.update(Path(p) for p in old['guarded_hashes'])
    return sorted(paths, key=str)


def develop():
    cfg, grids, primary, task = setup()
    require_committed(LOCKED)
    if RUN.exists():
        raise FileExistsError('Retain every previous attempt')
    RUN.mkdir(parents=True)
    provenance = dict(status='running', git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        config=cfg, grids=grids, task=task, environment=environment_record(),
        guarded_hashes={str(f): sha256(f) for f in source_paths(cfg, primary)},
        primary_manifest_sha256=sha256(Path(cfg['candidate_bundle']) / 'manifest.json'),
        primary_selected=primary.manifest['selected'], primary_unchanged=True,
        old_evaluation_read_during_development=False, independent_confirmation=False, final_goal_achieved=False)
    write(RUN / 'development.json', provenance)
    choices = {}; support = []; option_rows = []; grid_rows = []; metric_rows = []; index = 0
    try:
        for family, fold in task['families'].items():
            train = load_allowed(task, fold['train'], 'train')
            validation = load_allowed(task, fold['validation'], 'validation')
            for axis in fold['axes']:
                name = f'{family}_{axis}'; dest = RUN / name; dest.mkdir()
                cal = LengthCalibration(**primary.base.pilot.cases[name]['calibration'])
                support.extend(support_audit(train, name, task, cfg))
                fitted, count = fit_case(train, axis, family, task, grids['train'], cal, cfg, index)
                for filename, model in fitted.items():
                    drive.save(dest / filename, model)
                    grid_rows.append(dict(case=name, file=filename, representation=model['representation'],
                                          form=model['form'], alpha=model['alpha'], train_windows=count))
                specs = pool_specs(cfg, name); models = load_pool(specs)
                for spec in specs.values():
                    if 'file' in spec:spec['source_sha256'] = sha256(Path(spec['file']))
                scores = {age: {} for age in cfg['train_ages_raw']}; batches = []
                for sid, age, noise, current, target, labels, raw, packet, free in case_data(
                        validation, axis, family, task, grids['validation'], cal, cfg, index, cfg['train_ages_raw']):
                    forecasts = all_predictions(specs, models, raw, age, axis, primary.base, name)
                    for key, y in forecasts.items():
                        score = balanced(y, target, labels['drive_active'])
                        scores[age].setdefault(key, []).append(score)
                        option_rows.append(dict(case=name, scenario=int(sid), age_raw=age, noise_quantum=noise,
                                                option=key, validation_mae_deg=score))
                    old_file = Path(cfg['dated_development']) / name / f'validation_s{sid}_a{age}_n{noise:g}.npz'
                    with np.load(old_file) as old:
                        np.testing.assert_array_equal(current['k'], old['k'])
                        for key, value in labels.items():np.testing.assert_array_equal(value, old[key])
                        selected = old[primary.manifest['selected']].copy()
                    with np.load(Path(cfg['dated_development']) / name / ('reference_' + old_file.name)) as old:
                        previous_reference = old['matched_arx'].copy()
                    batches.append((sid, age, noise, current['k'], target, labels, forecasts, selected, previous_reference))
                    print('REFERENCE_VAL', name, age, noise, 'options', len(forecasts), flush=True)
                reference = choose_reference(scores)
                choices[name] = dict(family=family, axis=axis, reference=reference, specs=specs,
                                     augmented_train_windows=count, source_support_records=len(train))
                for sid, age, noise, k, target, labels, forecasts, selected, previous in batches:
                    key = reference[str(age)]['key']; y = forecasts[key]
                    np.savez_compressed(dest / f'validation_s{sid}_a{age}_n{noise:g}.npz', k=k, **labels,
                        selected=selected, previous_matched_arx=previous, dense_matched_arx=y,
                        reference_key=np.array(key), matched_train_age_raw=np.array(age))
                    add_rows(metric_rows, dict(selected=selected, previous_matched_arx=previous, dense_matched_arx=y),
                             target, labels, name, sid, 'validation', task, grids['validation'], age, noise)
                index += 1
        selection = dict(cases=choices, primary_selected=primary.manifest['selected'], primary_unchanged=True,
            primary_manifest_sha256=provenance['primary_manifest_sha256'], options_per_case=304,
            new_arx_fits=len(grid_rows), validation_choices=sum(len(c['reference']) for c in choices.values()),
            independent_confirmation=False, final_goal_achieved=False)
        write(RUN / 'selection.json', selection); write(RUN / 'train_support.json', dict(records=support,
            same_source_records_roles_channels=True, new_dense_origins_subset_original=True,
            original_first_origins_retained_in_source_models=True, identical_transformed_training_examples_claimed=False,
            identical_consumed_angle_sample_indices_claimed=False))
        pd.DataFrame(option_rows).to_csv(RUN / 'validation_option_scores.csv', index=False)
        pd.DataFrame(grid_rows).to_csv(RUN / 'arx_grid.csv', index=False)
        pd.DataFrame(metric_rows).to_csv(RUN / 'validation_metrics.csv', index=False)
        provenance.update(status='complete', selection_sha256=sha256(RUN / 'selection.json'),
            result_hashes={str(f.relative_to(RUN)): sha256(f) for f in RUN.rglob('*') if f.is_file() and f.name != 'development.json'})
        print('REFERENCE_FROZEN', len(grid_rows), 'fits', selection['validation_choices'], 'choices', flush=True)
    except Exception as error:
        provenance.update(status='failed', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(RUN / 'development.json', provenance)


def require_frozen_development():
    p = json.loads((RUN / 'development.json').read_text())
    if p['status'] != 'complete':raise ValueError('Complete development required')
    verify_hashes(p['guarded_hashes']); verify_hashes(p['result_hashes'], RUN)
    for rel in ['development.json'] + list(p['result_hashes']):
        path = OUT / 'frozen_development' / rel
        if path.read_bytes() != (RUN / rel).read_bytes():raise ValueError('Development mirror differs')
        require_committed([path])
    return p


def evaluate():
    cfg, grids, primary, task = setup()
    development = require_frozen_development()
    selection = json.loads((RUN / 'selection.json').read_text())
    out = RUN / 'retrospective'
    if out.exists():raise FileExistsError('Retain prior replay')
    out.mkdir(); provenance = dict(status='running', selection_sha256=sha256(RUN / 'selection.json'),
        development_sha256=sha256(RUN / 'development.json'), primary_unchanged=True,
        independent_confirmation=False, final_goal_achieved=False)
    write(out / 'provenance.json', provenance); rows = []; index = 0
    try:
        for family, fold in task['families'].items():
            records = load_confirmation(task, fold['confirmation'])
            for axis in fold['axes']:
                name = f'{family}_{axis}'; case = selection['cases'][name]
                cal = LengthCalibration(**primary.base.pilot.cases[name]['calibration'])
                models = load_pool({v['key']: case['specs'][v['key']] for v in case['reference'].values()})
                for sid, age, noise, current, target, labels, raw, packet, free in case_data(
                        records, axis, family, task, grids['evaluation'], cal, cfg, index, cfg['evaluation_ages_raw']):
                    known_age = nearest(cfg['train_ages_raw'], age); key = case['reference'][str(known_age)]['key']
                    y = reference_prediction(case['specs'][key], models, raw, age, axis, primary.base, name)
                    original_path = Path(cfg['dated_replay']) / f'{name}_s{sid}_a{age}_n{noise:g}.npz'
                    with np.load(original_path) as old:retained = {k: old[k].copy() for k in old.files}
                    np.testing.assert_array_equal(current['k'], retained['k'])
                    for k, value in labels.items():np.testing.assert_array_equal(value, retained[k])
                    actual = primary.forecast(raw, family=family, axis=axis, angle_age_samples=age, include_candidates=True)
                    same(actual['forecast_rad'], retained['selected'])
                    for method, value in actual['candidates'].items():same(value, retained[method])
                    retained.update(dense_matched_arx=y, dense_reference_key=np.array(key),
                                    dense_matched_train_age_raw=np.array(known_age))
                    np.savez_compressed(out / original_path.name, **retained)
                    methods = primary.manifest['methods'] + ['selected', 'matched_arx', 'augmented_arx', 'bare_dated_modal', 'dense_matched_arx']
                    add_rows(rows, {m: retained[m] for m in methods}, target, labels, name, sid, 'old_evaluation',
                             task, grids['evaluation'], age, noise)
                    print('REFERENCE_REPLAY', name, sid, age, noise, key, flush=True)
                index += 1
        pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
        provenance.update(status='complete', result_hashes={str(f.relative_to(out)): sha256(f)
            for f in out.rglob('*') if f.is_file() and f.name != 'provenance.json'})
    except Exception as error:
        provenance.update(status='failed', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(out / 'provenance.json', provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--evaluate', action='store_true')
    evaluate() if parser.parse_args().evaluate else develop()
