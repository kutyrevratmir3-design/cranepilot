"""Rebuild dense TRAIN fits/all VAL choices, then check source/raw/portable replay."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts import drive_conditioned as drive
from scripts.dense_reference import (CONFIG, OUT, RUN, all_predictions, choose_reference,
    load_pool, nearest, pool_specs, reference_prediction, setup, support_audit)
from scripts.modal_history import load_allowed
from scripts.run_dated_forecast import fit_case
from scripts.run_external_benchmark import verify_hashes, write
from scripts.run_latency_observer import add_rows, balanced, case_data, current_packet, raw_history
from scripts.run_modal_confirmation import load_confirmation
from scripts.verify_finetune_campaign import original_files
from scripts.verify_latency_observer import compare_rows, model_names, same


def development():
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'development_verification.json'
    if dest.exists():raise FileExistsError('Retain previous verification')
    result = dict(passed=False, checks=[], train_refits=[], source_sha256=sha256(Path(__file__)),
                  independent_confirmation=False, final_goal_achieved=False)
    try:
        cfg, grids, primary, task = setup()
        provenance = json.loads((RUN / 'development.json').read_text())
        selection = json.loads((RUN / 'selection.json').read_text())
        assert provenance['status'] == 'complete' and not provenance['old_evaluation_read_during_development']
        verify_hashes(provenance['guarded_hashes']); verify_hashes(provenance['result_hashes'], RUN)
        index = 0; rows = []; options = []; support = []; chosen_count = 0
        for family, fold in task['families'].items():
            train = load_allowed(task, fold['train'], 'train')
            val = load_allowed(task, fold['validation'], 'validation')
            for axis in fold['axes']:
                name = f'{family}_{axis}'; case = selection['cases'][name]
                cal = LengthCalibration(**primary.base.pilot.cases[name]['calibration'])
                support.extend(support_audit(train, name, task, cfg))
                rebuilt, count = fit_case(train, axis, family, task, grids['train'], cal, cfg, index)
                assert count == case['augmented_train_windows']
                for filename, model in rebuilt.items():
                    actual = drive.load(RUN / name / filename)
                    assert model['representation'] == actual['representation'] and model['form'] == actual['form']
                    assert model['alpha'] == actual['alpha']
                    for key in ['mean', 'scale', 'coef']:
                        np.testing.assert_allclose(model[key], actual[key], rtol=1e-12, atol=1e-12)
                result['train_refits'].append(dict(case=name, fits=len(rebuilt), train_windows=count))
                specs = pool_specs(cfg, name)
                for spec in specs.values():
                    if 'file' in spec:spec['source_sha256'] = sha256(Path(spec['file']))
                assert specs == case['specs']
                models = load_pool(specs); scores = {age: {} for age in cfg['train_ages_raw']}
                for sid, age, noise, current, target, labels, raw, packet, free in case_data(
                        val, axis, family, task, grids['validation'], cal, cfg, index, cfg['train_ages_raw']):
                    forecasts = all_predictions(specs, models, raw, age, axis, primary.base, name)
                    for key, value in forecasts.items():
                        score = balanced(value, target, labels['drive_active'])
                        scores[age].setdefault(key, []).append(score)
                        options.append(dict(case=name, scenario=int(sid), age_raw=age, noise_quantum=noise,
                                            option=key, validation_mae_deg=score))
                    actual = primary.forecast(raw, family=family, axis=axis, angle_age_samples=age,
                                              include_reference=True, include_candidates=True)
                    key = case['reference'][str(age)]['key']
                    values = dict(selected=actual['forecast_rad'], previous_matched_arx=actual['matched_arx_rad'],
                                  dense_matched_arx=forecasts[key])
                    maximum = 0.
                    with np.load(RUN / name / f'validation_s{sid}_a{age}_n{noise:g}.npz') as z:
                        np.testing.assert_array_equal(current['k'], z['k'])
                        for field, value in labels.items():np.testing.assert_array_equal(value, z[field])
                        for field, value in values.items():maximum = max(maximum, same(value, z[field]))
                        assert str(z['reference_key']) == key and int(z['matched_train_age_raw']) == age
                    with np.load(Path(cfg['dated_development']) / name / f'validation_s{sid}_a{age}_n{noise:g}.npz') as z:
                        for method, value in actual['candidates'].items():same(value, z[method])
                    add_rows(rows, values, target, labels, name, sid, 'validation', task, grids['validation'], age, noise)
                    result['checks'].append(dict(case=name, scenario=int(sid), age_raw=age, noise=noise,
                                                options=len(forecasts), windows=len(target), maximum_error_rad=maximum))
                choices = choose_reference(scores)
                assert set(choices) == set(case['reference'])
                for age, choice in choices.items():
                    assert choice['key'] == case['reference'][age]['key']
                    for key, value in choice['validation_scores'].items():
                        np.testing.assert_allclose(value, case['reference'][age]['validation_scores'][key], rtol=1e-12, atol=1e-12)
                chosen_count += len(choices); index += 1
                print('DENSE_VERIFIED_VAL', name, len(choices), flush=True)
        assert sum(r['fits'] for r in result['train_refits']) == selection['new_arx_fits'] == 120
        assert chosen_count == selection['validation_choices'] == 30
        assert support == json.loads((RUN / 'train_support.json').read_text())['records']
        keys = ['case', 'scenario', 'age_raw', 'noise_quantum', 'option']
        pd.testing.assert_frame_equal(pd.DataFrame(options).sort_values(keys).reset_index(drop=True),
            pd.read_csv(RUN / 'validation_option_scores.csv').sort_values(keys).reset_index(drop=True),
            check_exact=False, check_dtype=False, rtol=1e-12, atol=1e-12)
        grid = pd.read_csv(RUN / 'arx_grid.csv')
        assert len(grid) == 120 and set(grid['case']) == set(selection['cases'])
        for name, case in selection['cases'].items():
            assert set(grid[grid['case'] == name].file) == set(model_names(cfg))
            assert set(grid[grid['case'] == name].train_windows) == {case['augmented_train_windows']}
        compare_rows(rows, pd.read_csv(RUN / 'validation_metrics.csv'))
        result.update(passed=True, development_sha256=sha256(RUN / 'development.json'),
            all_120_train_fits_rebuilt=True, all_30_reference_choices_reproduced=True,
            all_validation_option_scores_recomputed=True, validation_option_scores=len(options),
            dense_source_support_verified=True, original_first_origins_retained=True, primary_unchanged=True)
    except Exception as error:
        result['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write(dest, result)


def full():
    from cranepilot.matched_dated_pilot import FrozenDenseReference
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'verification.json'
    if dest.exists():raise FileExistsError('Retain previous verification')
    result = dict(passed=False, checks=[], clock_probes=[], source_sha256=sha256(Path(__file__)),
                  independent_confirmation=False, final_goal_achieved=False)
    try:
        cfg, grids, primary, task = setup()
        p = json.loads((RUN / 'development.json').read_text())
        q = json.loads((RUN / 'retrospective' / 'provenance.json').read_text())
        s = json.loads((RUN / 'selection.json').read_text())
        dev = json.loads((OUT / 'development_verification.json').read_text())
        assert dev['passed'] and dev['development_sha256'] == sha256(RUN / 'development.json')
        assert p['status'] == q['status'] == 'complete' and q['selection_sha256'] == sha256(RUN / 'selection.json')
        verify_hashes(p['guarded_hashes']); verify_hashes(p['result_hashes'], RUN)
        verify_hashes(q['result_hashes'], RUN / 'retrospective')
        result['original_files'] = original_files(); assert result['original_files']['passed']
        bundle = Path('artifacts/dense_reference_bundle_20261108').resolve()
        primary_bundle = Path(cfg['candidate_bundle']).resolve(); np_load = np.load
        def restricted(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if not any(resolved.is_relative_to(root) for root in [bundle, primary_bundle]):
                raise AssertionError('Portable reference read outside the two frozen bundles')
            return np_load(path, *args, **kwargs)
        with patch('numpy.load', restricted):api = FrozenDenseReference(bundle, primary_bundle)
        val_rows = []; old_rows = []; index = 0
        for family, fold in task['families'].items():
            validation = load_allowed(task, fold['validation'], 'validation')
            records = load_confirmation(task, fold['confirmation'])
            for axis in fold['axes']:
                name = f'{family}_{axis}'; case = s['cases'][name]
                cal = LengthCalibration(**primary.base.pilot.cases[name]['calibration'])
                needed = {value['key']: case['specs'][value['key']] for value in case['reference'].values()}
                models = load_pool(needed)
                for role, source, grid, ages in [('validation', validation, grids['validation'], cfg['train_ages_raw']),
                        ('old_evaluation', records, grids['evaluation'], cfg['evaluation_ages_raw'])]:
                    for sid, age, noise, current, target, labels, raw, packet, free in case_data(
                            source, axis, family, task, grid, cal, cfg, index, ages):
                        known = nearest(cfg['train_ages_raw'], age); key = case['reference'][str(known)]['key']
                        source_y = reference_prediction(case['specs'][key], models, raw, age, axis, primary.base, name)
                        with patch('numpy.load', restricted):
                            portable = api.forecast(raw, family=family, axis=axis, angle_age_samples=age)
                            original = primary.forecast(raw, family=family, axis=axis, angle_age_samples=age,
                                                        include_reference=True, include_candidates=True)
                        maximum = same(source_y, portable['forecast_rad'])
                        assert portable['reference_key'] == key and portable['matched_train_age_raw'] == known
                        if role == 'validation':
                            values = dict(selected=original['forecast_rad'], previous_matched_arx=original['matched_arx_rad'],
                                          dense_matched_arx=portable['forecast_rad'])
                            file = RUN / name / f'validation_s{sid}_a{age}_n{noise:g}.npz'
                            with np.load(file) as z:
                                np.testing.assert_array_equal(z['k'], current['k'])
                                for field, value in labels.items():np.testing.assert_array_equal(value, z[field])
                                for field, value in values.items():maximum = max(maximum, same(value, z[field]))
                            add_rows(val_rows, values, target, labels, name, sid, role, task, grid, age, noise)
                        else:
                            filename = f'{name}_s{sid}_a{age}_n{noise:g}.npz'
                            with np.load(RUN / 'retrospective' / filename) as z:
                                retained = {field: z[field].copy() for field in z.files}
                            with np.load(Path(cfg['dated_replay']) / filename) as z:
                                for field in z.files:
                                    np.testing.assert_array_equal(retained[field], z[field])
                            np.testing.assert_array_equal(retained['k'], current['k'])
                            for field, value in labels.items():np.testing.assert_array_equal(value, retained[field])
                            maximum = max(maximum, same(portable['forecast_rad'], retained['dense_matched_arx']),
                                          same(original['forecast_rad'], retained['selected']))
                            for method, value in original['candidates'].items():maximum = max(maximum, same(value, retained[method]))
                            for field, actual in [('matched_arx', 'matched_arx_rad'), ('augmented_arx', 'augmented_arx_rad'),
                                                  ('bare_dated_modal', 'bare_dated_modal_rad')]:
                                maximum = max(maximum, same(original[actual], retained[field]))
                            assert str(retained['dense_reference_key']) == key and int(retained['dense_matched_train_age_raw']) == known
                            methods = primary.manifest['methods'] + ['selected', 'matched_arx', 'augmented_arx',
                                                                     'bare_dated_modal', 'dense_matched_arx']
                            values = {method: retained[method] for method in methods}
                            add_rows(old_rows, values, target, labels, name, sid, role, task, grid, age, noise)
                        result['checks'].append(dict(role=role, case=name, scenario=int(sid), age_raw=age,
                            noise=noise, windows=len(target), methods=len(values), maximum_error_rad=maximum))
                # Poison unreceived angle/future support in a copied recording, not the original.
                sid = next(iter(validation)); record = validation[sid]
                packet, _, _ = current_packet(record, axis, task, grids['validation'], cal)
                k = packet['k'][:1]; cutoff = int(k[0])
                for age in [0, 5, 15, 25]:
                    before = raw_history(record, k, axis, family, age, 0, cfg, index)
                    changed = copy.deepcopy(record)
                    for field, value in changed.signals.items():value[(cutoff - age + 1 if field == axis else cutoff + 1):] = 1e6
                    after = raw_history(changed, k, axis, family, age, 0, cfg, index)
                    for field in before:np.testing.assert_array_equal(before[field], after[field])
                    with patch('numpy.load', restricted):
                        a = api.forecast(before, family=family, axis=axis, angle_age_samples=age)['forecast_rad']
                        b = api.forecast(after, family=family, axis=axis, angle_age_samples=age)['forecast_rad']
                    np.testing.assert_array_equal(a, b)
                    result['clock_probes'].append(dict(case=name, age_raw=age, maximum_error_rad=0.))
                index += 1
                print('DENSE_VERIFIED_API', name, flush=True)
        compare_rows(val_rows, pd.read_csv(RUN / 'validation_metrics.csv'))
        compare_rows(old_rows, pd.read_csv(RUN / 'retrospective' / 'metrics.csv'))
        original_rows = [row for row in old_rows if row['method'] != 'dense_matched_arx']
        compare_rows(original_rows, pd.read_csv(Path(cfg['dated_replay']) / 'metrics.csv'))
        result.update(passed=True, development_sha256=sha256(RUN / 'development.json'),
            retrospective_sha256=sha256(RUN / 'retrospective' / 'provenance.json'),
            comparator_bundle_manifest_sha256=sha256(bundle / 'manifest.json'),
            primary_bundle_manifest_sha256=sha256(primary_bundle / 'manifest.json'),
            primary_unchanged=True, every_original_replay_array_retained=True,
            original_and_new_metrics_recomputed=True, portable_reads_limited_to_two_bundles=True,
            forecast_vectors_checked=sum(c['windows'] * c['methods'] for c in result['checks']),
            maximum_error_rad=max(c['maximum_error_rad'] for c in result['checks']))
    except Exception as error:
        result['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write(dest, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--development-only', action='store_true')
    development() if parser.parse_args().development_only else full()
