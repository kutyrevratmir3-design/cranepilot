"""Fetch/verify and audit the separate Aalto data without fitting or scoring models."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json
import re
import urllib.request
import numpy as np
import pandas as pd


RECORD = 'https://zenodo.org/api/records/6907569'
EXPECTED_MD5 = '2ad915f66f38ba7a5a61f8e70183b09c'
EXPECTED_BYTES = 5855928


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    a = parser.parse_args()
    root = Path('data/raw/aalto_6907569')
    root.mkdir(parents=True, exist_ok=True)
    archive = root/'crane_dataset.zip'
    if a.download:
        with urllib.request.urlopen(RECORD, timeout=30) as response:
            metadata = json.load(response)
        entry = next(f for f in metadata['files'] if f['key'] == 'crane_dataset.zip')
        if entry['checksum'] != 'md5:'+EXPECTED_MD5 or entry['size'] != EXPECTED_BYTES:
            raise ValueError('Published record changed; inspect before importing')
        (root/'metadata.json').write_text(json.dumps(metadata, indent=2))
        if not archive.exists():
            url = entry['links']['self']
            if url != RECORD+'/files/crane_dataset.zip/content':
                raise ValueError('Unexpected file endpoint')
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if len(content) != EXPECTED_BYTES or hashlib.md5(content).hexdigest() != EXPECTED_MD5:
                raise ValueError('Downloaded content checksum/size mismatch')
            archive.write_bytes(content)
    content = archive.read_bytes()
    if len(content) != EXPECTED_BYTES or hashlib.md5(content).hexdigest() != EXPECTED_MD5:
        raise ValueError('Local archive checksum/size mismatch')
    metadata = json.loads((root/'metadata.json').read_text())
    cycles, raw_rows, alignments, future_checks, future_examples, tag_names = [], [], [], [], [], set()
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Zip CRC mismatch')
        frame = pd.read_csv(z.open('combined_csv.csv'))
        parameters = z.read('Raw data/cycle_parameters.txt.txt').decode().replace('\r', '')
        for cycle, group in frame.groupby('Cycle', sort=True):
            dates = pd.to_datetime(group.Date.str.strip(), format='mixed', utc=True)
            delta = dates.diff().dt.total_seconds().dropna()
            cycles.append({'cycle': int(cycle), 'rows': len(group),
                'payload_kg': [0, 0, 120, 120, 500, 500, 1000, 1000][int(cycle)-1],
                'anti_sway': 'on' if cycle % 2 else 'off',
                'median_step_s': float(delta.median()), 'max_step_s': float(delta.max()),
                'nonpositive_steps': int((delta <= 0).sum()),
                'duration_s': float((dates.iloc[-1]-dates.iloc[0]).total_seconds()),
                'clock_years': ','.join(map(str, sorted(dates.dt.year.unique()))),
                'trolley_angle_unchanged_fraction': float((group.TrolleyRopeAngle.diff().iloc[1:] == 0).mean())})
        for name in z.namelist():
            if not name.endswith('tagconfig.txt'):
                continue
            tag_map = {}
            for line in z.read(name).decode('utf-8', errors='replace').splitlines():
                fields = line.split('\t')
                if len(fields) >= 7:
                    tag = fields[-1].split(';s=', 1)[-1]
                    tag_names.add(tag)
                    tag_map[int(fields[0])] = tag
            relevant = {key: value for key, value in tag_map.items() if any(
                word in value for word in ['Angle_rad', 'SpeedRequest', 'SlingLength', 'Controls.Trolley.Speed'])}
            if not relevant:
                continue
            directory = name.rsplit('/', 1)[0]
            csv_names = [n for n in z.namelist() if n.rsplit('/', 1)[0] == directory
                         and n.endswith('.csv') and 'DataLogger' in n]
            raw = pd.concat([pd.read_csv(z.open(n)) for n in csv_names], ignore_index=True)
            cycle = int(re.search(r'/Cycle(\d+)/', name).group(1))
            for key, tag in relevant.items():
                group = raw[raw.PrimaryKey == key]
                dates = pd.to_datetime(group.SourceTimeStamp, format='mixed', utc=True)
                diffs = dates.sort_values().diff().dt.total_seconds().dropna()
                raw_rows.append({'cycle': cycle, 'tag': tag, 'events': len(group),
                    'unique_values': int(group.Value.nunique()),
                    'non_good_status_events': int((group.StatusCode != '0x00000000').sum()),
                    'median_event_interval_s': float(diffs.median()) if len(diffs) else None,
                    'max_event_interval_s': float(diffs.max()) if len(diffs) else None,
                    'source_files': '|'.join(csv_names)})
                if 'RopeAngle.Angle_rad' in tag and len(group):
                    column = 'TrolleyRopeAngle' if '.Trolley.' in tag else 'BridgeRopeAngle'
                    exported = frame[frame.Cycle == cycle]
                    grid = pd.DataFrame({'time': pd.to_datetime(exported.Date.str.strip(),
                        format='mixed', utc=True), 'exported': exported[column].to_numpy()}).sort_values('time')
                    events = pd.DataFrame({'time': dates.to_numpy(),
                        'raw_value': pd.to_numeric(group.Value).to_numpy()})
                    events = events.sort_values('time').drop_duplicates('time', keep='last')
                    events['event_time'] = events['time']
                    for direction in ['backward', 'forward', 'nearest']:
                        aligned = pd.merge_asof(grid, events, on='time', direction=direction)
                        valid = aligned.raw_value.notna()
                        difference = abs(aligned.loc[valid, 'exported']-aligned.loc[valid, 'raw_value'])
                        alignments.append({'cycle': cycle, 'column': column, 'direction': direction,
                            'matched_rows': int(valid.sum()), 'total_rows': len(grid),
                            'equal_within_1e-9_fraction': float((difference < 1e-9).mean()),
                            'max_difference_rad': float(difference.max()),
                            'mean_difference_rad': float(difference.mean())})
                    before = pd.merge_asof(grid, events, on='time', direction='backward')
                    after = pd.merge_asof(grid, events, on='time', direction='forward')
                    ahead = (after.event_time-after.time).dt.total_seconds()
                    suspicious = ((abs(after.exported-after.raw_value) < 1e-9)
                        & (abs(before.exported-before.raw_value) > 1e-9) & (ahead > 0))
                    future_checks.append({'cycle': cycle, 'column': column,
                        'future_match_past_mismatch_rows': int(suspicious.sum()),
                        'total_rows': len(grid), 'fraction': float(suspicious.mean()),
                        'median_lookahead_s': float(ahead[suspicious].median())})
                    for index in after.index[suspicious][:3]:
                        future_examples.append({'cycle': cycle, 'column': column,
                            'export_timestamp': str(after.loc[index, 'time']),
                            'exported_angle_rad': float(after.loc[index, 'exported']),
                            'previous_event_timestamp': str(before.loc[index, 'event_time']),
                            'previous_angle_rad': float(before.loc[index, 'raw_value']),
                            'next_event_timestamp': str(after.loc[index, 'event_time']),
                            'next_angle_rad': float(after.loc[index, 'raw_value']),
                            'lookahead_s': float(ahead.loc[index])})
        member_hashes = {n: hashlib.sha256(z.read(n)).hexdigest() for n in
            ['combined_csv.csv', 'README.txt', 'LICENCE.txt', 'CONTRIBUTORS.txt',
             'Raw data/cycle_parameters.txt.txt']}
    out = Path('results/latest')
    cycle_frame = pd.DataFrame(cycles)
    cycle_frame.to_csv(out/'aalto_20261106_cycles.csv', index=False)
    pd.DataFrame(raw_rows).to_csv(out/'aalto_20261106_raw_signal_audit.csv', index=False)
    pd.DataFrame(alignments).to_csv(out/'aalto_20261106_alignment_audit.csv', index=False)
    pd.DataFrame(future_checks).to_csv(out/'aalto_20261106_future_match_audit.csv', index=False)
    pd.DataFrame(future_examples).to_csv(out/'aalto_20261106_future_match_examples.csv', index=False)
    audit = {'doi': metadata['doi'], 'version': metadata['metadata']['version'],
        'source': RECORD, 'license': metadata['metadata']['license']['id'],
        'bytes': len(content), 'md5': EXPECTED_MD5,
        'sha256': hashlib.sha256(content).hexdigest(), 'member_hashes': member_hashes,
        'rows': len(frame), 'columns': frame.columns.tolist(),
        'missing_values': {k: int(v) for k, v in frame.isna().sum().items()},
        'cycles': cycles, 'cycle_description': parameters,
        'raw_tag_names': sorted(tag_names), 'future_match_audit': future_checks,
        'models_fitted': 0, 'forecast_scores_computed': 0,
        'limitations': [
            'Combined table is at 250 ms, not the 10 ms contract of the current model.',
            'Combined table lacks prescribed reference and explicit physical cable length.',
            'Raw tags include speed requests and sling-length setting; their semantics need verification.',
            'Hoist position is not automatically suspension length; speed feedback is not automatically m/s.',
            'Raw timestamps have year 2103. Preserve originals; do not invent a corrected calendar date.',
            'Preprocessing of the combined table and age of event-based signals require a causal reconstruction audit.',
            'Angle_rad tag documents units but does not alone establish an independent physical sensor measurement.',
            'No direct plug-in, 100 Hz upsampling claim, transfer score or hardware validation is justified by this audit.']}
    (out/'aalto_20261106_audit.json').write_text(json.dumps(audit, indent=2))
    print(cycle_frame.to_string(index=False))
    print(pd.DataFrame(raw_rows)[['cycle', 'tag', 'events', 'median_event_interval_s']].to_string(index=False))


if __name__ == '__main__':
    main()
