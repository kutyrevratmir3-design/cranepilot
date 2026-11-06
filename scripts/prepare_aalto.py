"""Reconstruct causal Aalto observations from events, retaining log ages and gaps.

No forecasts, train/test split, command interpretation or inferred cable length.
"""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scripts.audit_aalto import EXPECTED_MD5


SIGNALS = {
    'trolley_angle_rad': 'Status.Trolley.RopeAngle.Angle_rad',
    'bridge_angle_rad': 'Status.Bridge.RopeAngle.Angle_rad',
    'trolley_position_m': 'Status.Trolley.Position.Position_m',
    'bridge_position_m': 'Status.Bridge.Position.Position_m',
    'hoist_position_m': 'Status.Hoist.Position.Position_m',
    'trolley_speed_mmin': 'Status.Trolley.ControlSignals.SpeedFeedback_mmin',
    'bridge_speed_mmin': 'Status.Bridge.ControlSignals.SpeedFeedback_mmin',
    'trolley_speed_request_native': 'Status.Trolley.ControlSignals.SpeedRequest',
    'bridge_speed_request_native': 'Status.Bridge.ControlSignals.SpeedRequest',
    'sling_setting_mm': 'RadioSelection.SwayControl_SlingLength_mm',
}


def causal_hold(source_time, available_time, values, grid):
    """Times share any numeric unit; ages use that unit. No future event is used."""
    source_time, available_time, values, grid = [np.asarray(x) for x in
        (source_time, available_time, values, grid)]
    if source_time.shape != available_time.shape or source_time.shape != values.shape:
        raise ValueError('Event arrays must have matching shapes')
    if any(x.ndim != 1 for x in (source_time, available_time, values, grid)):
        raise ValueError('One-dimensional event arrays required')
    available_time = np.maximum(source_time, available_time)
    order = np.argsort(available_time, kind='stable')
    index = np.searchsorted(available_time[order], grid, side='right')-1
    observed = index >= 0
    out = np.full(len(grid), np.nan)
    age = np.full(len(grid), np.nan)
    if observed.any():
        picked = order[index[observed]]
        out[observed] = values[picked]
        age[observed] = grid[observed]-source_time[picked]
        if np.any(available_time[picked] > grid[observed]):
            raise AssertionError('A future event entered a causal observation')
    return out, age


def timestamp_ns(values):
    return pd.DatetimeIndex(pd.to_datetime(values, format='mixed', utc=True)).as_unit('ns').asi8


def main():
    archive = Path('data/raw/aalto_6907569/crane_dataset.zip')
    if hashlib.md5(archive.read_bytes()).hexdigest() != EXPECTED_MD5:
        raise ValueError('Unexpected source archive')
    out = Path('data/processed/aalto_6907569')
    if out.exists():
        raise FileExistsError('Preserve the existing reconstruction')
    records = {}
    with ZipFile(archive) as z:
        combined = pd.read_csv(z.open('combined_csv.csv'), usecols=['Date', 'Cycle'])
        for name in z.namelist():
            if not name.endswith('tagconfig.txt'):
                continue
            cycle = int(re.search(r'/Cycle(\d+)/', name).group(1))
            mapping = {}
            for line in z.read(name).decode().splitlines():
                parts = line.split('\t')
                if len(parts) >= 7:
                    for key, suffix in SIGNALS.items():
                        if parts[-1].endswith('.'+suffix):
                            mapping[int(parts[0])] = key
            if not mapping:
                continue
            directory = name.rsplit('/', 1)[0]
            names = [n for n in z.namelist() if n.rsplit('/', 1)[0] == directory
                     and n.endswith('.csv') and 'DataLogger' in n]
            raw = pd.concat([pd.read_csv(z.open(n)) for n in names], ignore_index=True)
            for primary_key, key in mapping.items():
                events = raw[(raw.PrimaryKey == primary_key) & (raw.StatusCode == '0x00000000')]
                values = pd.to_numeric(events.Value, errors='coerce').to_numpy()
                valid = np.isfinite(values)
                records[cycle, key] = (timestamp_ns(events.SourceTimeStamp)[valid],
                    timestamp_ns(events.ServerTimeStamp)[valid], values[valid])
    arrays, summaries = {}, []
    for cycle, group in combined.groupby('Cycle', sort=True):
        grid = timestamp_ns(group.Date.str.strip())
        prefix = f'cycle{cycle}'
        arrays[prefix+'_timestamp_ns'] = grid
        for key in SIGNALS:
            events = records.get((cycle, key), ([], [], []))
            value, age = causal_hold(*events, grid)
            arrays[prefix+'_'+key] = value
            arrays[prefix+'_'+key+'_log_age_s'] = age/1e9
            summaries.append({'cycle': int(cycle), 'signal': key, 'rows': len(grid),
                'missing_rows': int(np.isnan(value).sum()), 'events': len(events[0]),
                'median_log_age_s': float(np.nanmedian(age)/1e9) if np.isfinite(age).any() else None,
                'max_log_age_s': float(np.nanmax(age)/1e9) if np.isfinite(age).any() else None})
    out.mkdir(parents=True)
    np.savez_compressed(out/'observations.npz', **arrays)
    provenance = {'source_doi': '10.5281/zenodo.6907569', 'license': 'CC BY 4.0',
        'source_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256((out/'observations.npz').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'signals': SIGNALS, 'summaries': summaries, 'model_fits': 0, 'forecast_scores': 0,
        'rule': 'Last GOOD finite raw event with max(source,server) timestamp <= grid time.',
        'grid_source': 'Only Date and Cycle columns of combined_csv.csv; no exported signal values used.',
        'caveats': ['Log age is age since the last logged event, not necessarily sensor age.',
            'No backward fill or synthetic fresh high-frequency observations.',
            'Original year 2103 timestamps retained.',
            'Native speed requests are not identified as force or known future planned commands.',
            'Hoist position plus sling setting is not an established cable-length formula.',
            'Not a plug-in dataset for the current 100 Hz model.']}
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    Path('results/latest/aalto_20261106_causal_reconstruction.json').write_text(json.dumps(provenance, indent=2))
    pd.DataFrame(summaries).to_csv('results/latest/aalto_20261106_causal_signals.csv', index=False)
    print(pd.DataFrame(summaries).groupby('signal')[['rows', 'events', 'missing_rows']].sum().to_string())


if __name__ == '__main__':
    main()
