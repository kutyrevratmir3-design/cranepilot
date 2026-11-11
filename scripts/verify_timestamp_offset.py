"""Audit saved timestamp-offset arrays without forecasting, fitting, or selection.

The partial-batch mode never writes. Full mode requires the completed producer
manifest and writes only a new, separate audit JSON (never into its input run).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_modal_confirmation import load_confirmation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path('configs/timestamp_offset_20261109.json')
RUN = Path('results/timestamp_offset_20261109')
OUTPUT = Path('results/timestamp_offset_audit_20261109/verification.json')
METHODS = ('selected', 'dense_matched_arx')
VALUES = ['mae_deg', 'paired_delta_from_zero_deg']
AXIS_KEYS = ['family', 'axis', 'scenario', 'true_age_raw', 'offset_raw', 'horizon_raw', 'method']
ROUTE_KEYS = ['family', 'axis', 'scenario', 'true_age_raw', 'offset_raw']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def check_hashes(mapping, base=Path('.')):
    require(bool(mapping), 'Empty provenance hash mapping')
    for name, expected in mapping.items():
        require(digest(base / name) == expected, f'SHA256 mismatch: {base / name}')
    return len(mapping)


def supported(age, offsets):
    return [offset for offset in offsets if 0 <= age + offset <= 25]


def context(partial):
    cfg = read_json(CONFIG)
    provenance = read_json(RUN / 'provenance.json')
    require(provenance['status'] in (('running', 'complete') if partial else ('complete',)),
            'Full audit requires a completed run; failed runs cannot pass partial audit')
    for name, expected in {'retrospective': True, 'independent_confirmation': False,
                           'hardware_success_claimed': False, 'model_selection_or_fit': False,
                           'final_goal_achieved': False}.items():
        require(provenance.get(name) is expected, f'Provenance qualification changed: {name}')
    locked = {str(CONFIG), 'docs/TIMESTAMP_OFFSET_DESIGN_20261109.md',
              'scripts/timestamp_offset.py', 'tests/test_timestamp_offset.py'}
    require(set(provenance['source_hashes']) == locked, 'Incomplete locked source manifest')
    counts = {'producer_sources': check_hashes(provenance['source_hashes'])}
    evidence_path = Path(cfg['evidence_manifest'])
    evidence_sha = digest(evidence_path)
    require(evidence_sha == cfg['evidence_manifest_sha256'] == provenance['input_manifest_sha256'],
            'Frozen evidence manifest digest mismatch')
    evidence = read_json(evidence_path)['files']
    counts['frozen_evidence'] = check_hashes(evidence)
    require(cfg['development'] in evidence, 'Archived development absent from frozen evidence')
    development = read_json(cfg['development'])
    require(development['status'] == 'complete', 'Archived development is incomplete')
    counts['old_guarded_inputs'] = check_hashes(development['guarded_hashes'])
    task = development['task']
    for name in ['dataset.npz', 'manifest.json']:
        path = str(Path(task['data']) / name)
        require(path in development['guarded_hashes'], f'Processed input is not guarded: {path}')
    np.testing.assert_array_equal(task['horizons_raw'], cfg['horizons_raw'])
    require(len(set(cfg['true_ages_raw'])) == len(cfg['true_ages_raw']), 'Duplicate true ages')
    require(len(set(cfg['offsets_raw'])) == len(cfg['offsets_raw']) and 0 in cfg['offsets_raw'],
            'Duplicate offsets or missing zero offset')
    require(cfg['origin_stride_raw'] > 0 and cfg['history_samples'] > 0, 'Invalid packet geometry')
    declared_ids = []
    expected_batches = {}
    for family, spec in cfg['families'].items():
        require(spec['records'] == task['families'][family]['confirmation'], 'Record set changed')
        require(spec['axes'] == task['families'][family]['axes'], 'Axes changed')
        declared_ids.extend(spec['records'])
        for sid in spec['records']:
            for axis in spec['axes']:
                for age in cfg['true_ages_raw']:
                    require(0 <= age <= 25, 'Unsupported true age')
                    expected_batches[f'{family}_{axis}_s{sid}_a{age}.npz'] = (family, axis, sid, age)
    require(len(declared_ids) == len(set(declared_ids)), 'Physical records span multiple families')
    require(set(declared_ids) == set(task['sealed_confirmation_ids']), 'Old-only record scope changed')
    primary_path = Path(cfg['primary_bundle']) / 'manifest.json'
    reference_path = Path(cfg['reference_bundle']) / 'manifest.json'
    for path in [primary_path, reference_path]:
        require(str(path) in evidence, f'Bundle manifest absent from frozen evidence: {path}')
    primary, reference = read_json(primary_path), read_json(reference_path)
    require(primary['selected'] == cfg['expected_primary'] == reference['primary_selected'],
            'Primary model identity changed')
    require(reference['primary_manifest_sha256'] == digest(primary_path), 'Reference primary digest changed')
    for label, bundle, manifest in [('primary', cfg['primary_bundle'], primary),
                                     ('reference', cfg['reference_bundle'], reference)]:
        counts[label + '_bundle_files'] = check_hashes(manifest['files'], Path(bundle))
        counts[label + '_source_files'] = check_hashes(manifest['source_hashes'])
    require(reference['train_ages_raw'] and len(set(reference['train_ages_raw'])) == len(reference['train_ages_raw']),
            'Invalid routing age grid')
    if not partial:
        counts['saved_results'] = check_hashes(provenance['result_hashes'], RUN)
        actual = {str(p.relative_to(RUN)) for p in RUN.rglob('*')
                  if p.is_file() and p != RUN / 'provenance.json'}
        require(actual == set(provenance['result_hashes']), 'Saved-result manifest coverage mismatch')
    # This existing loader reads only the declared old evaluation records.
    records = load_confirmation(task, declared_ids)
    return cfg, provenance, evidence, reference, records, expected_batches, counts


def integer_array(array, expected, name):
    require(np.issubdtype(array.dtype, np.integer), f'{name} must use integer dtype')
    require(array.shape == np.asarray(expected).shape, f'{name} shape mismatch')
    np.testing.assert_array_equal(array, expected, err_msg=name)


def finite_float(array, shape, name):
    require(array.shape == shape and np.issubdtype(array.dtype, np.floating), f'{name} shape/dtype mismatch')
    require(np.isfinite(array).all(), f'{name} contains nonfinite values')


def verify_batch(path, spec, cfg, evidence, reference, records):
    before = digest(path)
    with np.load(path, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    require(before == digest(path), f'Batch changed during read: {path}')
    require(set(saved) == {'k', 'true_age_raw', 'offsets_raw', 'horizons_raw', 'target_rad',
                           'reference_key', 'reference_route_age_raw', *METHODS}, 'Unexpected NPZ schema')
    family, axis, sid, age = spec
    seq = records[sid]
    h = np.asarray(cfg['horizons_raw'], dtype=int)
    k = np.arange(cfg['first_origin_raw'], len(seq.signals[axis]) - h.max(), cfg['origin_stride_raw'])
    offsets = supported(age, cfg['offsets_raw'])
    require(len(k) > 0 and k[0] - age - cfg['history_samples'] + 1 >= 0, 'Incomplete real history support')
    integer_array(saved['k'], k, 'k')
    integer_array(saved['true_age_raw'], age, 'true_age_raw')
    integer_array(saved['offsets_raw'], offsets, 'offsets_raw')
    integer_array(saved['horizons_raw'], h, 'horizons_raw')
    target = seq.signals[axis][k[:, None] + h]
    finite_float(saved['target_rad'], target.shape, 'target_rad')
    np.testing.assert_array_equal(saved['target_rad'], target, err_msg='Current-clock target mismatch')
    require(saved['reference_key'].shape == (len(offsets),) and saved['reference_key'].dtype.kind == 'U',
            'Reference key shape/dtype mismatch')
    train_ages = reference['train_ages_raw']
    routed_ages = [min(train_ages, key=lambda known: (abs(known - (age + offset)), known)) for offset in offsets]
    routed_keys = [reference['cases'][f'{family}_{axis}']['reference'][str(a)] for a in routed_ages]
    integer_array(saved['reference_route_age_raw'], routed_ages, 'reference_route_age_raw')
    np.testing.assert_array_equal(saved['reference_key'], routed_keys, err_msg='Nearest-age route mismatch')
    zero = offsets.index(0)
    old_path = Path(cfg['replay']) / (path.stem + '_n0.npz')
    require(str(old_path) in evidence, f'Zero-offset archive absent from evidence: {old_path}')
    maximum = 0.0
    with np.load(old_path, allow_pickle=False) as old:
        np.testing.assert_array_equal(old['k'], k)
        np.testing.assert_array_equal(old['time_s'], seq.time[k])
        require(str(old['dense_reference_key']) == routed_keys[zero], 'Old reference key mismatch')
        require(int(old['dense_matched_train_age_raw']) == routed_ages[zero], 'Old reference route mismatch')
        for method in METHODS:
            finite_float(saved[method], (len(offsets), len(k), len(h)), method)
            finite_float(old[method], (len(k), len(h)), 'old ' + method)
            delta = float(np.max(np.abs(saved[method][zero] - old[method])))
            require(delta <= cfg['zero_offset_tolerance_rad'], f'Zero-offset parity failed: {method}, {delta}')
            maximum = max(maximum, delta)
    rows, routes = [], []
    for index, offset in enumerate(offsets):
        identity = dict(family=family, axis=axis, scenario=sid, true_age_raw=age,
                        offset_raw=offset, declared_age_raw=age + offset)
        routes.append(dict(**identity, reference_key=routed_keys[index],
                           matched_train_age_raw=routed_ages[index],
                           changed_weight_key=routed_keys[index] != routed_keys[zero]))
        for method in METHODS:
            error = np.abs(saved[method][index] - target) * (180.0 / np.pi)
            zero_error = np.abs(saved[method][zero] - target) * (180.0 / np.pi)
            # Paired window-level differences, independently of producer summaries.
            mae, paired = error.mean(axis=0), (error - zero_error).mean(axis=0)
            for head, value, change in zip(h, mae, paired):
                rows.append(dict(**identity, horizon_raw=int(head), method=method,
                                 windows=len(k), mae_deg=float(value), paired_delta_from_zero_deg=float(change)))
    checked = dict(batch=str(path), sha256=before, windows=len(k), supported_offsets=offsets,
                   target_exact=True, nearest_age_route_exact=True, maximum_zero_offset_error_rad=maximum)
    return rows, routes, checked


def rebuilt_tables(rows, routes):
    axis = pd.DataFrame(rows)
    record_keys = ['true_age_raw', 'offset_raw', 'horizon_raw', 'scenario', 'method']
    record = axis.groupby(record_keys, as_index=False)[VALUES].mean()
    horizon_keys = ['true_age_raw', 'offset_raw', 'horizon_raw', 'method']
    horizon = record.groupby(horizon_keys, as_index=False)[VALUES].mean()
    cell_keys = ['true_age_raw', 'offset_raw', 'method']
    cells = horizon.groupby(cell_keys, as_index=False)[VALUES].mean()
    record_all = record.groupby(['true_age_raw', 'offset_raw', 'scenario', 'method'], as_index=False).mae_deg.mean()
    paired_keys = ['true_age_raw', 'offset_raw', 'scenario']
    paired = record_all.pivot(index=paired_keys, columns='method', values='mae_deg').reset_index()
    paired['selected_minus_dense_deg'] = paired['selected'] - paired['dense_matched_arx']
    age_sets = [set(group.true_age_raw) for _, group in cells.groupby('offset_raw')]
    common = sorted(set.intersection(*age_sets))
    require(bool(common), 'No common interior support')
    tables = {
        'axis_horizon_metrics.csv': (axis, AXIS_KEYS, VALUES),
        'record_horizon_metrics.csv': (record, record_keys, VALUES),
        'mean_by_horizon.csv': (horizon, horizon_keys, VALUES),
        'paired_record_all_horizons.csv': (paired, paired_keys, [*METHODS, 'selected_minus_dense_deg']),
        'mean_by_true_age_offset.csv': (cells, cell_keys, VALUES),
        'reference_routes.csv': (pd.DataFrame(routes), ROUTE_KEYS, []),
    }
    for scope, subset in [('available_support', cells), ('common_interior', cells[cells.true_age_raw.isin(common)])]:
        grouped = subset.groupby(['offset_raw', 'method'], as_index=False).agg(
            mae_deg=('mae_deg', 'mean'), paired_delta_from_zero_deg=('paired_delta_from_zero_deg', 'mean'),
            true_age_count=('true_age_raw', 'nunique'))
        tables[scope + '_offset_means.csv'] = (grouped, ['offset_raw', 'method'], VALUES)
    return tables, common


def compare_csv(name, expected, keys, floating):
    actual = pd.read_csv(RUN / name)
    require(set(actual.columns) == set(expected.columns), f'CSV columns differ: {name}')
    require(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), f'Duplicate cells: {name}')
    require(len(actual) == len(expected), f'CSV coverage differs: {name}')
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    maximum = 0.0
    for column in expected.columns:
        if column in floating:
            require(np.isfinite(actual[column].to_numpy()).all(), f'Nonfinite CSV value: {name}/{column}')
            np.testing.assert_allclose(actual[column], expected[column], rtol=1e-12, atol=1e-10,
                                       err_msg=f'{name}/{column}')
            maximum = max(maximum, float(np.max(np.abs(actual[column] - expected[column]))))
        else:
            np.testing.assert_array_equal(actual[column], expected[column], err_msg=f'{name}/{column}')
    return dict(file=name, rows=len(actual), maximum_error_deg=maximum)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-batch', type=Path, help='Audit one completed NPZ; print only, never write')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='New full-audit JSON outside producer results')
    args = parser.parse_args(argv)
    require(Path.cwd().resolve() == ROOT, 'Run from repository root using the restored Python launcher')
    partial = args.verify_batch is not None
    if not partial:
        require(not args.output.resolve().is_relative_to(RUN.resolve()), 'Audit output must be outside producer run')
        require(not args.output.exists(), f'Retain prior audit: {args.output}')
    cfg, provenance, evidence, reference, records, expected, hashes = context(partial)
    if partial:
        path = args.verify_batch
        require(path.resolve().parent == (RUN / 'batches').resolve() and path.name in expected,
                'Partial batch must be an expected producer NPZ')
        rows, routes, checked = verify_batch(path, expected[path.name], cfg, evidence, reference, records)
        print(json.dumps(dict(passed=True, mode='one_batch_only', full_run_verified=False,
                              summary_csvs_verified=False, **checked, recomputed_metric_cells=len(rows))))
        return
    paths = sorted((RUN / 'batches').glob('*.npz'))
    require({path.name for path in paths} == set(expected), 'Incomplete or extra NPZ batch coverage')
    rows, routes, checks = [], [], []
    for path in paths:
        batch_rows, batch_routes, checked = verify_batch(path, expected[path.name], cfg, evidence, reference, records)
        rows.extend(batch_rows)
        routes.extend(batch_routes)
        checks.append(checked)
    tables, common = rebuilt_tables(rows, routes)
    csv_checks = [compare_csv(name, *spec) for name, spec in tables.items()]
    vectors = sum(check['windows'] * len(check['supported_offsets']) * len(METHODS) for check in checks)
    zero_vectors = sum(check['windows'] * len(METHODS) for check in checks)
    maximum = max(check['maximum_zero_offset_error_rad'] for check in checks)
    for name, value in {'paired_angle_history_batches': len(routes), 'forecast_vectors': vectors,
                         'zero_offset_vectors_checked': zero_vectors,
                         'maximum_zero_offset_error_rad': maximum, 'common_true_ages_raw': common}.items():
        require(provenance[name] == value, f'Producer total mismatch: {name}')
    # Recheck the completed manifest after consuming every array and CSV.
    check_hashes(provenance['result_hashes'], RUN)
    require(read_json(RUN / 'provenance.json') == provenance, 'Producer provenance changed during audit')
    report = dict(passed=True, mode='full_saved_array_audit', source_sha256=digest(Path(__file__)),
                  provenance_sha256=digest(RUN / 'provenance.json'), input_manifest_sha256=cfg['evidence_manifest_sha256'],
                  hashes_verified=hashes, batches=len(checks), physical_records=len(records),
                  paired_angle_history_batches=len(routes), forecast_vectors=vectors,
                  zero_offset_vectors_checked=zero_vectors, maximum_zero_offset_error_rad=maximum,
                  axis_horizon_metric_cells=len(rows), common_true_ages_raw=common,
                  exact_processed_targets=True, complete_paired_coverage=True, csv_checks=csv_checks,
                  batch_checks=checks, model_calls=0, model_fits=0, retrospective=True,
                  independent_confirmation=False, hardware_success_claimed=False, final_goal_achieved=False,
                  limitation='Nonzero forecasts are audited as archived finite arrays; they are not independently rerun. '
                             'Timestamp histories are not saved in these NPZs, so unchanged-history construction '
                             'is covered by the locked producer and its tests, not reconstructed from saved histories.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, mode='full_saved_array_audit', output=str(args.output),
                          batches=len(checks), forecast_vectors=vectors, csv_files=len(csv_checks),
                          maximum_zero_offset_error_rad=maximum)))


if __name__ == '__main__':
    main()
