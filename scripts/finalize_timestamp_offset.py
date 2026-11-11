"""Recover timestamp stress metadata, or replay with its JSON type conversion."""
import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import patch

from scripts import timestamp_offset as producer
from scripts import verify_timestamp_offset as audit

RECOVERY = Path('results/timestamp_offset_recovery_20261109')
LOG = Path('../../handoff/restoration_20261107/timestamp_offset_20261109.log')


def replay():
    """Identical predictions and summaries; convert NumPy age labels to JSON ints."""
    summarize = producer.summarize

    def json_safe_ages(rows, out):
        return [int(age) for age in summarize(rows, out)]

    with patch.object(producer, 'summarize', json_safe_ages):
        producer.run()


def recover():
    run = audit.RUN
    metadata = run / 'provenance.json'
    if RECOVERY.exists():
        raise FileExistsError('Retain the previous recovery record')
    initial_bytes = metadata.read_bytes()
    error_log = LOG.read_text()
    if ('TIMESTAMP_OFFSET_COMPLETE 372 15792 0.0' not in error_log or
        'TypeError: Object of type int64 is not JSON serializable' not in error_log):
        raise ValueError('Recovery is only for the observed final serialization error')
    cfg, initial, evidence, reference, records, expected, _ = audit.context(partial=True)
    if initial['status'] != 'running':
        raise ValueError('Expected the original start record left by serialization failure')
    result_files = [f for f in sorted(run.rglob('*')) if f.is_file() and f != metadata]
    before = {str(f.relative_to(run)): audit.digest(f) for f in result_files}
    paths = sorted((run / 'batches').glob('*.npz'))
    if {p.name for p in paths} != set(expected):
        raise ValueError('Saved predictions do not cover the original design')
    rows, routes, checks = [], [], []
    for path in paths:
        part, route, check = audit.verify_batch(path, expected[path.name], cfg, evidence, reference, records)
        rows.extend(part)
        routes.extend(route)
        checks.append(check)
    tables, common = audit.rebuilt_tables(rows, routes)
    for name, spec in tables.items():
        audit.compare_csv(name, *spec)
    zero = sum(c['windows']*len(audit.METHODS) for c in checks)
    maximum = max(c['maximum_zero_offset_error_rad'] for c in checks)
    assert len(routes)==372 and zero==15792 and maximum==0
    audit.check_hashes(before, run)
    assert metadata.read_bytes() == initial_bytes
    RECOVERY.mkdir()
    saved_start = RECOVERY/'initial_running_provenance.json'
    saved_start.write_bytes(initial_bytes)
    shutil.copyfile(LOG, RECOVERY/'producer_exit_1.log')
    complete = dict(initial, status='complete', paired_angle_history_batches=len(routes),
        forecast_vectors=sum(c['windows']*len(c['supported_offsets'])*len(audit.METHODS) for c in checks),
        zero_offset_vectors_checked=zero, maximum_zero_offset_error_rad=maximum,
        common_true_ages_raw=[int(a) for a in common], result_hashes=before,
        recovered_metadata_only=True, original_producer_exit_code=1,
        recovery_source_sha256=audit.digest(Path(__file__)),
        initial_provenance_sha256=audit.digest(saved_start),
        original_failure_log_sha256=audit.digest(RECOVERY/'producer_exit_1.log'),
        recovery_note='All forecasts and summary CSVs were already written. Only the terminal JSON failed to serialize a NumPy integer; no result array or metric was rewritten.')
    final_bytes = (json.dumps(complete,indent=2,allow_nan=False)+'\n').encode()
    temp = metadata.with_suffix('.recovered.tmp')
    with temp.open('xb') as stream:
        stream.write(final_bytes)
    temp.replace(metadata)
    audit.check_hashes(before, run)
    print('TIMESTAMP_METADATA_RECOVERED', len(paths), len(routes), complete['forecast_vectors'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='Use the JSON-safe wrapper for a new run in a clean checkout with the required data')
    args = parser.parse_args()
    replay() if args.replay else recover()
