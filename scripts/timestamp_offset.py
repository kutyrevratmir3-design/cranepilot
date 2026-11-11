"""Run an immutable metadata-only age stress against the two frozen APIs."""
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from cranepilot.data import sha256
from cranepilot.dated_pilot import FrozenDatedPilot
from cranepilot.matched_dated_pilot import FrozenDenseReference
from scripts.run_external_benchmark import verify_hashes, write
from scripts.run_latency_observer import raw_history
from scripts.run_modal_confirmation import load_confirmation

CONFIG = Path('configs/timestamp_offset_20261109.json')
OUT = Path('results/timestamp_offset_20261109')
LOCKED = [
    CONFIG, Path('docs/TIMESTAMP_OFFSET_DESIGN_20261109.md'),
    Path('scripts/timestamp_offset.py'), Path('tests/test_timestamp_offset.py'),
]
METHODS = ('selected', 'dense_matched_arx')


def valid_offsets(age, offsets):
    return [offset for offset in offsets if 0 <= age + offset <= 25]


def new_packet(s, axis, family, age, k, horizons):
    history = raw_history(s, k, axis, family, age, 0, {}, 0)
    target = s.signals[axis][k[:, None] + horizons]
    return history, target


def score_batch(history, target, primary, reference, family, axis, declared, h):
    outputs = []
    for api in [primary, reference]:
        out = api.forecast({key: x.copy() for key, x in history.items()},
                           family=family, axis=axis, angle_age_samples=declared)
        if out['angle_age_samples'] != declared:
            raise ValueError('Forecast used a different declared age')
        np.testing.assert_allclose(out['horizons_s'], h * .01, atol=1e-12, rtol=0)
        v = out['forecast_rad']
        if v.shape != target.shape or not np.isfinite(v).all():
            raise ValueError('Nonfinite forecast or incomplete window coverage')
        outputs.append(out)
    return outputs


def summarize(rows, out):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'axis_horizon_metrics.csv', index=False)
    keys = ['true_age_raw', 'offset_raw', 'horizon_raw', 'scenario', 'method']
    values = ['mae_deg', 'paired_delta_from_zero_deg']
    records = frame.groupby(keys, as_index=False)[values].mean()
    records.to_csv(out / 'record_horizon_metrics.csv', index=False)
    mean = records.groupby(keys[:3] + ['method'], as_index=False)[values].mean()
    mean.to_csv(out / 'mean_by_horizon.csv', index=False)
    record_all = records.groupby(['true_age_raw', 'offset_raw', 'scenario', 'method'], as_index=False)[values].mean()
    pivot = record_all.pivot(index=['true_age_raw', 'offset_raw', 'scenario'], columns='method', values='mae_deg').reset_index()
    pivot['selected_minus_dense_deg'] = pivot.selected - pivot.dense_matched_arx
    pivot.to_csv(out / 'paired_record_all_horizons.csv', index=False)
    cells = mean.groupby(['true_age_raw', 'offset_raw', 'method'], as_index=False)[values].mean()
    cells.to_csv(out / 'mean_by_true_age_offset.csv', index=False)
    ages_per_offset = cells.groupby('offset_raw').true_age_raw.agg(lambda v: sorted(v.unique()))
    common = sorted(set.intersection(*(set(a) for a in ages_per_offset)))
    assert common
    for scope, subset in [('available_support', cells), ('common_interior', cells[cells.true_age_raw.isin(common)])]:
        agg = subset.groupby(['offset_raw', 'method'], as_index=False).agg(
            mae_deg=('mae_deg', 'mean'), paired_delta_from_zero_deg=('paired_delta_from_zero_deg', 'mean'),
            true_age_count=('true_age_raw', 'nunique'))
        agg.to_csv(out / (scope + '_offset_means.csv'), index=False)
    return common


def run():
    for p in LOCKED:
        if subprocess.check_output(['git', 'show', f'HEAD:{p}']) != p.read_bytes():
            raise ValueError(f'Commit design/config/source before replay: {p}')
    cfg = json.loads(CONFIG.read_text())
    mf = Path(cfg['evidence_manifest'])
    if sha256(mf) != cfg['evidence_manifest_sha256']:
        raise ValueError('Frozen evidence manifest changed')
    frozen = json.loads(mf.read_text())['files']
    verify_hashes(frozen)
    dev = json.loads(Path(cfg['development']).read_text())
    verify_hashes(dev['guarded_hashes'])
    out = OUT
    out.mkdir()
    batches = out / 'batches'
    batches.mkdir()
    v = dict(status='running', git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             source_hashes={str(p): sha256(p) for p in LOCKED}, input_manifest_sha256=sha256(mf),
             retrospective=True, independent_confirmation=False, hardware_success_claimed=False,
             model_selection_or_fit=False, final_goal_achieved=False)
    write(out / 'provenance.json', v)
    start = perf_counter()
    rows, routing = [], []
    max_error = 0.
    comparisons = 0
    try:
        primary = FrozenDatedPilot(Path(cfg['primary_bundle']))
        reference = FrozenDenseReference(Path(cfg['reference_bundle']), Path(cfg['primary_bundle']))
        if primary.manifest['selected'] != cfg['expected_primary']:
            raise ValueError('Primary was replaced')
        task = primary.base.pilot.cfg
        h = np.array(cfg['horizons_raw'])
        np.testing.assert_array_equal(task['horizons_raw'], h)
        archive = Path(cfg['replay'])
        for family, spec in cfg['families'].items():
            fold = task['families'][family]
            assert list(fold['confirmation']) == spec['records'] and list(fold['axes']) == spec['axes']
            for sid, s in load_confirmation(task, spec['records']).items():
                for axis in spec['axes']:
                    k = np.arange(cfg['first_origin_raw'], len(s.signals[axis]) - h.max(), cfg['origin_stride_raw'])
                    for age in cfg['true_ages_raw']:
                        history, target = new_packet(s, axis, family, age, k, h)
                        assert len(k) == 94 and all(a.shape == (len(k), cfg['history_samples']) for a in history.values())
                        name = f'{family}_{axis}_s{sid}_a{age}'
                        offsets = valid_offsets(age, cfg['offsets_raw'])
                        predicted = {}
                        details = {}
                        for off in [0] + [d for d in offsets if d]:
                            values = score_batch(history, target, primary, reference, family, axis, age + off, h)
                            predicted[off] = {key: p['forecast_rad'] for key, p in zip(METHODS, values)}
                            details[off] = values[1]
                            if off == 0:
                                with np.load(archive / (name + '_n0.npz'), allow_pickle=False) as saved:
                                    np.testing.assert_array_equal(k, saved['k'])
                                    np.testing.assert_array_equal(s.time[k], saved['time_s'])
                                    assert values[1]['reference_key'] == str(saved['dense_reference_key'])
                                    assert values[1]['matched_train_age_raw'] == int(saved['dense_matched_train_age_raw'])
                                    for key, val in predicted[0].items():
                                        delta = float(np.max(np.abs(val - saved[key])))
                                        assert delta <= cfg['zero_offset_tolerance_rad'], (name, key, delta)
                                        max_error = max(max_error, delta)
                                        comparisons += len(k)
                        np.savez_compressed(batches / (name + '.npz'), k=k, true_age_raw=age,
                            offsets_raw=offsets, horizons_raw=h, target_rad=target,
                            reference_key=np.array([details[d]['reference_key'] for d in offsets]),
                            reference_route_age_raw=np.array([details[d]['matched_train_age_raw'] for d in offsets]),
                            **{key: np.stack([predicted[d][key] for d in offsets]) for key in METHODS})
                        for off in offsets:
                            routing.append(dict(family=family, axis=axis, scenario=sid, true_age_raw=age,
                                offset_raw=off, declared_age_raw=age + off, reference_key=details[off]['reference_key'],
                                matched_train_age_raw=details[off]['matched_train_age_raw'],
                                changed_weight_key=details[off]['reference_key'] != details[0]['reference_key']))
                            for key in METHODS:
                                mae = np.rad2deg(np.abs(predicted[off][key] - target)).mean(axis=0)
                                base = np.rad2deg(np.abs(predicted[0][key] - target)).mean(axis=0)
                                for head, error, change in zip(h, mae, mae - base):
                                    rows.append(dict(family=family, axis=axis, scenario=sid, true_age_raw=age,
                                        offset_raw=off, declared_age_raw=age + off, horizon_raw=int(head),
                                        method=key, windows=len(k), mae_deg=float(error), paired_delta_from_zero_deg=float(change)))
                    print('TIMESTAMP_OFFSET', family, sid, axis, flush=True)
        pair_count = sum(len(valid_offsets(a, cfg['offsets_raw'])) for a in cfg['true_ages_raw'])
        axis_count = sum(len(f['records'])*len(f['axes']) for f in cfg['families'].values())
        assert len(rows) == pair_count * axis_count * len(h) * len(METHODS)
        assert len(routing) == pair_count * axis_count
        pd.DataFrame(routing).to_csv(out / 'reference_routes.csv', index=False)
        common = summarize(rows, out)
        v.update(status='complete', paired_angle_history_batches=len(routing), forecast_vectors=len(routing)*94*2,
                 zero_offset_vectors_checked=comparisons, maximum_zero_offset_error_rad=max_error,
                 common_true_ages_raw=common, elapsed_s=perf_counter()-start,
                 result_hashes={str(f.relative_to(out)): sha256(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='provenance.json'})
        print('TIMESTAMP_OFFSET_COMPLETE', len(routing), comparisons, max_error, flush=True)
    except Exception as error:
        v.update(status='failed', failure=f'{type(error).__name__}: {error}', elapsed_s=perf_counter()-start)
        raise
    finally:
        write(out / 'provenance.json', v)


if __name__ == '__main__':
    run()
