"""Private intake declarations and post-lock potential-duplicate screening.

No loader, predictor, training, scoring or eligibility decision lives here.
``metadata_preflight`` hashes bytes without parsing their numerical contents.
Complete declarations still require a separately locked numerical-schema audit.

Manifest v1 has exactly ``schema_version``, ``source``, ``records``. Source:
``source_id, rig_id, physical_recordings=True, provenance_evidence,
research_use_permitted=True, research_use_evidence``; raw_release_permitted
defaults to False. Evidence strings are declarations, not verified proof.

Record: record_id/session_id/trial_id/condition_id/independent_cluster_id, relative file/sha256,
family, axes, sample_rate_hz=100, clock_scope, angle_age_samples,
history_s>=4+age*.01, target_tail_s>=1.5, genuine_history=True, channels,
encoder_quantum_rad (positive shared scalar or per-axis mapping), filters, acquisition,
arrival. IDs identify unique physical trials, not overlapping forecast windows.
Repeated conditions/clusters are allowed; the session/trial tuple is unique.
Family requires theta for trolley and theta+beta for jib/simult, recorded once
per physical trial. Clusters are declarations; software cannot certify them.

Channel declarations have column and normalized unit. Angle/J: rad; L/x: m.
Each reference also needs semantics, training_semantics, training_unit and
compatibility_evidence. Merely renaming motor voltage or a position setpoint
to native Inteco xr/Jr never establishes compatibility.
Optional training_contract_sha256 binds to a separately frozen reviewed contract;
its syntax is checked here and the specialized lock must check the actual hash.
An explicitly opaque 'native' unit may be retained with identical declared
training semantics/unit and evidence for the exact exporter/rig/firmware/schema.
It is not a statement that two different rigs' native units are equivalent.

Clock scopes:
* synchronous_acquisition_offline: angle_age_samples explicitly 0 describes
  acquisition-grid alignment, NOT an assumed physical arrival delay. Acquisition
  requires synchronized=True plus synchronization_evidence; arrival may be None.
* known_arrival_age: actual acquisition/arrival clock declarations, evidence and
  nonangle_availability_evidence; age is a known integer 0..25. Different clock
  IDs require clock_relation_evidence. Unknown age is never changed to zero.
  arrival.nonangle_time_columns maps every required nonangle channel to its
  explicit availability-array column. Optional angle_time_columns maps every
  selected angle to its arrival column; otherwise shared time_column is declared.

Acquisition/arrival objects use time_column, clock_id, unit='s', evidence.
Filters declare kind ('none' or 'causal'), description, evidence; causal filters
also need delay_model. Noncausal/unknown filters are blocked for this profile.

Numerical functions require lock_verified=True. Arrays must already be finite,
normalized, real 1D values. Hashes are screening evidence only. A nonmatch cannot
prove an independent trial. Constant blocks are skipped ONLY in overlap search,
never as a reason to exclude scoring windows. A seen stride of16 with candidate
every-offset search can miss overlaps shorter than block_size+15 samples.
"""
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import struct

import numpy as np


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash file bytes in a bounded stream; do not interpret their format."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('Positive integer chunk_size required')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(chunk_size), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value):
    """Hash JSON types with string keys, stable ordering and no NaN/Infinity."""
    def check(item):
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError('JSON keys must be strings')
            for entry in item.values():
                check(entry)
        elif isinstance(item, list):
            for entry in item:
                check(entry)
        elif item is None or type(item) in (str, bool, int):
            return
        elif type(item) is float and math.isfinite(item):
            return
        else:
            raise ValueError('Finite JSON values required')
    check(value)
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def metadata_preflight(manifest, private_root, *, seen_file_hashes=(),
                       max_records=256, max_issues=64):
    """Validate declarations/hash bytes only; never return an eligible record."""
    if type(max_records) is not int or not 1 <= max_records <= 10000:
        raise ValueError('max_records must be 1..10000')
    if type(max_issues) is not int or not 1 <= max_issues <= 1000:
        raise ValueError('max_issues must be 1..1000')
    issues, files, clock_audits = [], [], []
    issue_count = 0

    def issue(field, message, index=None):
        nonlocal issue_count
        issue_count += 1
        if len(issues) < max_issues:
            issues.append(dict(record_index=index, field=field, message=message))

    def obj(value, allowed, field, index=None):
        if type(value) is not dict:
            issue(field, 'Object required', index)
            return {}
        if set(value) - allowed:
            issue(field, 'Unknown fields (including future/target fields) forbidden', index)
        return {key: entry for key, entry in value.items() if key in allowed}

    def text(value):
        return type(value) is str and 0 < len(value.strip()) <= 4096 and value.strip().lower() not in {
            'unknown', 'unspecified', 'not known', 'none', 'n/a', 'tbd', '?'}

    def number(value):
        try:
            return type(value) in (float, int) and math.isfinite(value)
        except OverflowError:
            return False

    def clock(value, field, index, synchronized=False):
        item = obj(value, {'time_column', 'clock_id', 'unit', 'evidence',
                          'synchronized', 'synchronization_evidence',
                          'clock_relation_evidence', 'nonangle_availability_evidence',
                          'nonangle_time_columns', 'angle_time_columns'}, field, index)
        for key in ['time_column', 'clock_id', 'evidence']:
            if not text(item.get(key)):
                issue(field + '.' + key, 'Explicit declaration/evidence required', index)
        if type(item.get('unit')) is not str or item['unit'] != 's':
            issue(field + '.unit', 'Normalized timestamps in seconds required', index)
        if synchronized and (item.get('synchronized') is not True or not text(item.get('synchronization_evidence'))):
            issue(field, 'Certified common acquisition grid evidence required', index)
        return item

    result = dict(status='blocked', metadata_declarations_complete=False,
                  numerical_schema_verified=False, eligible=False,
                  fresh_confirmed=False, independent_trial_proven=False,
                  never_independent_certificate=True,
                  raw_release_permitted=False, source_files=files,
                  clock_scopes=clock_audits, issues=issues,
                  requires_postlock_schema_and_overlap_screen=True,
                  evidence_scope='declarations_and_file_bytes_only')
    try:
        result['canonical_manifest_sha256'] = canonical_json_hash(manifest)
    except (TypeError, ValueError, RecursionError):
        result['canonical_manifest_sha256'] = None
        issue('manifest', 'Finite canonical JSON required')
    root = Path(private_root).resolve()
    if not root.is_dir():
        issue('private_root', 'Existing private directory required')
    top = obj(manifest, {'schema_version', 'source', 'records'}, 'manifest')
    if type(top.get('schema_version')) is not int or top.get('schema_version') != 1:
        issue('schema_version', 'Version1 required')
    source = obj(top.get('source'), {'source_id', 'rig_id', 'physical_recordings',
                 'provenance_evidence', 'research_use_permitted',
                 'research_use_evidence', 'raw_release_permitted'}, 'source')
    for key in ['source_id', 'rig_id', 'provenance_evidence', 'research_use_evidence']:
        if not text(source.get(key)):
            issue('source.' + key, 'Explicit physical-source/use evidence required')
    for key in ['physical_recordings', 'research_use_permitted']:
        if source.get(key) is not True:
            issue('source.' + key, 'Explicit true declaration required')
    release = source.get('raw_release_permitted', False)
    if type(release) is not bool:
        issue('source.raw_release_permitted', 'Explicit boolean required; omitted defaults false')
    else:
        result['raw_release_permitted'] = release
    records = top.get('records')
    if type(records) is not list or not records:
        issue('records', 'Nonempty physical-trial list required')
        records = []
    if len(records) > max_records:
        issue('records', 'Record bound exceeded; no successful partial preflight')
    ids, trials, digests = set(), set(), set()
    seen = set(seen_file_hashes)
    record_keys = {'record_id', 'session_id', 'trial_id', 'condition_id', 'independent_cluster_id', 'file',
                   'sha256', 'family', 'axes', 'sample_rate_hz', 'clock_scope',
                   'angle_age_samples', 'history_s', 'target_tail_s',
                   'genuine_history', 'channels', 'encoder_quantum_rad',
                   'filters', 'acquisition', 'arrival'}
    family_coordinates = {'trolley': ['x'], 'jib': ['J'], 'simult': ['x', 'J']}
    ref_units = {'m', 'rad', 'm/s', 'rad/s', 'm/s^2', 'rad/s^2', 'N', 'N*m', 'V', 'A', '1', 'native'}
    for index, value in enumerate(records[:max_records]):
        record = obj(value, record_keys, 'record', index)
        for key in ['record_id', 'session_id', 'trial_id', 'condition_id', 'independent_cluster_id']:
            if not text(record.get(key)):
                issue(key, 'Explicit physical-trial identifier required', index)
        rid = record.get('record_id')
        if type(rid) is str:
            if rid in ids:
                issue('record_id', 'Duplicate record identifier', index)
            ids.add(rid)
        identity = tuple(record.get(k) for k in ['session_id', 'trial_id'])
        if all(type(x) is str for x in identity):
            if identity in trials:
                issue('trial_id', 'Duplicate session/trial identity; condition changes do not create an independent trial', index)
            trials.add(identity)
        family = record.get('family')
        if type(family) is not str or family not in family_coordinates:
            issue('family', 'Declared supported crane family required', index)
            coords = []
        else:
            coords = family_coordinates[family]
        axes = record.get('axes')
        if type(axes) is not list or not axes or any(type(x) is not str for x in axes):
            issue('axes', 'Explicit nonempty target-axis list required', index)
            axes = []
        elif len(set(axes)) != len(axes) or set(axes) != ({'theta'} if family == 'trolley' else {'theta', 'beta'}):
            issue('axes', 'All family target axes exactly once per physical trial required', index)
        required = set(axes + ['L'] + coords + ['xr' if c == 'x' else 'Jr' for c in coords])
        if not number(record.get('sample_rate_hz')) or record['sample_rate_hz'] != 100:
            issue('sample_rate_hz', 'Declared100Hz acquisition grid required', index)
        age = record.get('angle_age_samples')
        valid_age = type(age) is int and 0 <= age <= 25
        if not valid_age:
            issue('angle_age_samples', 'Explicit known integer0..25 required, never default zero', index)
        scope = record.get('clock_scope')
        if scope == 'synchronous_acquisition_offline':
            if not valid_age or age != 0:
                issue('angle_age_samples', 'Offline common acquisition alignment must explicitly be zero', index)
            clock(record.get('acquisition'), 'acquisition', index, synchronized=True)
            if 'arrival' not in record:
                issue('arrival', 'Explicit None or clock metadata required; no silent latency assumption', index)
            elif record['arrival'] is not None:
                clock(record['arrival'], 'arrival', index)
            clock_audits.append(dict(record_index=index, scope=scope,
                angle_age_interpretation='acquisition_alignment_only',
                physical_arrival_age_declared_known=False, physical_arrival_age_verified=False,
                claims='offline_forecast_at_acquisition_origin_only'))
        elif scope == 'known_arrival_age':
            acq = clock(record.get('acquisition'), 'acquisition', index)
            arrival = clock(record.get('arrival'), 'arrival', index)
            if not text(arrival.get('nonangle_availability_evidence')):
                issue('arrival', 'Current availability evidence for nonangle channels required', index)
            available_columns = arrival.get('nonangle_time_columns')
            if type(available_columns) is not dict or set(available_columns) != required - set(axes) or any(not text(name) for name in available_columns.values()):
                issue('arrival.nonangle_time_columns', 'Explicit availability column for every required nonangle channel required', index)
            if 'angle_time_columns' in arrival:
                angle_columns = arrival['angle_time_columns']
                if type(angle_columns) is not dict or set(angle_columns) != set(axes) or any(not text(name) for name in angle_columns.values()):
                    issue('arrival.angle_time_columns', 'If supplied, map every selected angle to an explicit arrival column', index)
            if acq.get('clock_id') != arrival.get('clock_id') and not text(arrival.get('clock_relation_evidence')):
                issue('arrival', 'Different clocks require an explicit relation evidence', index)
            clock_audits.append(dict(record_index=index, scope=scope,
                angle_age_interpretation='declared_actual_arrival_age',
                physical_arrival_age_declared_known=valid_age, physical_arrival_age_verified=False,
                claims='availability_still_requires_postlock_verification'))
        else:
            issue('clock_scope', 'Choose offline synchronized acquisition or actual known arrival age explicitly', index)
        need = 4 + age * .01 if valid_age else math.inf
        if record.get('genuine_history') is not True or not number(record.get('history_s')) or record['history_s'] + 1e-12 < need:
            issue('history_s', 'Genuine4s+age prehistory required', index)
        if not number(record.get('target_tail_s')) or record['target_tail_s'] + 1e-12 < 1.5:
            issue('target_tail_s', 'Declared1.5s target tail required', index)
        quantum = record.get('encoder_quantum_rad')
        valid_quantum = (set(quantum) == set(axes) and all(number(q) and q > 0 for q in quantum.values())) if type(quantum) is dict else number(quantum) and quantum > 0
        if not valid_quantum:
            issue('encoder_quantum_rad', 'Explicit positive normalized angle resolution required', index)
        filt = obj(record.get('filters'), {'kind', 'description', 'evidence', 'delay_model'}, 'filters', index)
        if type(filt.get('kind')) is not str or filt['kind'] not in ['none', 'causal'] or not (text(filt.get('description')) or (filt.get('kind') == 'none' and filt.get('description') == 'none')) or not text(filt.get('evidence')):
            issue('filters', 'Known raw/causal filtering description and evidence required', index)
        if filt.get('kind') == 'causal' and not text(filt.get('delay_model')):
            issue('filters.delay_model', 'Explicit causal filter delay treatment required', index)
        channels = obj(record.get('channels'), {'theta', 'beta', 'L', 'x', 'xr', 'J', 'Jr'}, 'channels', index)
        if not required <= set(channels):
            issue('channels', 'Required selected angles/length/support/reference schema missing', index)
        columns = []
        for key, channel in channels.items():
            ch = obj(channel, {'column', 'unit', 'semantics', 'training_semantics',
                              'training_unit', 'compatibility_evidence',
                              'training_contract_sha256'}, 'channels.' + key, index)
            if not text(ch.get('column')):
                issue('channels.' + key, 'Explicit source column required', index)
            else:
                columns.append(ch['column'])
            if key in ['xr', 'Jr']:
                contract = ch.get('training_contract_sha256')
                if 'training_contract_sha256' in ch and (type(contract) is not str or len(contract) != 64 or any(c not in '0123456789abcdef' for c in contract)):
                    issue('channels.' + key, 'training_contract_sha256 must be a lowercase64hex commitment when supplied', index)
                if type(ch.get('unit')) is not str or ch['unit'] not in ref_units or type(ch.get('training_unit')) is not str or ch['training_unit'] != ch['unit']:
                    issue('channels.' + key, 'Explicit matching normalized source/training reference units required', index)
                if not text(ch.get('semantics')) or not text(ch.get('training_semantics')) or ch['training_semantics'] != ch['semantics'] or not text(ch.get('compatibility_evidence')):
                    issue('channels.' + key, 'Actual reference meaning must match training contract with evidence; names alone are insufficient', index)
            elif type(ch.get('unit')) is not str or ch['unit'] != ('m' if key in ['L', 'x'] else 'rad'):
                issue('channels.' + key, 'NormalizedSI angle/length/position units required', index)
        if len(columns) != len(set(columns)):
            issue('channels', 'Distinct channels cannot silently alias a source column', index)
        rel, expected = record.get('file'), record.get('sha256')
        try:
            if type(rel) is not str or not rel or '\\' in rel or ':' in rel or PurePosixPath(rel).is_absolute() or any(p in ['.', '..'] for p in rel.split('/')) or '' in rel.split('/'):
                raise ValueError('Relative file path without traversal required')
            path = (root / rel).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Source file must stay within private root')
            if type(expected) is not str or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError('Explicit lowercase SHA256 required')
            actual = sha256_file(path)
            if actual != expected:
                issue('sha256', 'Declared source-file bytes mismatch', index)
            if actual in seen:
                issue('sha256', 'Exact source bytes already in known-seen registry', index)
            if actual in digests:
                issue('sha256', 'Repeated raw bytes, possibly renamed duplicate trial', index)
            digests.add(actual)
            files.append(dict(record_index=index, relative_path=rel, sha256=actual,
                              bytes=path.stat().st_size, declared_hash_matches=actual == expected))
        except (OSError, ValueError, RuntimeError) as error:
            issue('file', str(error), index)
    result.update(metadata_declarations_complete=issue_count == 0,
                  status='metadata_declared_only' if issue_count == 0 else 'blocked',
                  issue_count=issue_count, issues_truncated=issue_count > len(issues))
    return result


def _quantized_relative(values, rounding):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        q = np.rint((values - values[0]) / rounding)
    if not np.isfinite(q).all() or np.max(np.abs(q), initial=0) >= 2**62:
        raise ValueError('Normalized values exceed declared quantization range')
    q = q.astype('<i8')
    nonzero = np.flatnonzero(q)
    if len(nonzero) and q[nonzero[0]] < 0:
        q = np.asarray(-q, dtype='<i8')
    return q


def fingerprint_sequence(values, *, rounding=1e-8, block_size=256,
                         seen_stride=16, candidate=False, lock_verified=False,
                         max_blocks=100000):
    """Post-lock numerical hashes; no targets, forecast errors or exclusions."""
    if lock_verified is not True:
        raise ValueError('Verify an immutable intake lock before numerical inspection')
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype.kind not in 'iuf' or not len(values):
        raise ValueError('Nonempty normalized real1D NumPy array required')
    if type(rounding) not in (int, float) or not math.isfinite(rounding) or rounding <= 0:
        raise ValueError('Positive finite declared rounding required')
    for name, value in [('block_size', block_size), ('seen_stride', seen_stride), ('max_blocks', max_blocks)]:
        if type(value) is not int or value <= 0:
            raise ValueError('Positive integer ' + name + ' required')
    if block_size < 2 or type(candidate) is not bool:
        raise ValueError('Block size>=2 and explicit candidate boolean required')
    x = np.array(values, dtype='<f8', copy=True)
    if values.dtype.kind in 'iu' and (np.any(values > 2**53) or (values.dtype.kind == 'i' and np.any(values < -2**53))):
        raise ValueError('Normalized integers must be exactly representable as float64')
    if not np.isfinite(x).all():
        raise ValueError('Finite normalized values required')
    x[x == 0] = 0.  # canonicalize negative zero without modifying the caller
    step = 1 if candidate else seen_stride
    starts = range(0, max(len(x) - block_size + 1, 0), step)
    if len(starts) > max_blocks:
        raise ValueError('Explicit block bound exceeded; no successful partial fingerprint')
    full_q = _quantized_relative(x, rounding)
    header = b'CPSEQ1\0' + struct.pack('<Q', len(x))
    quantum = json.dumps(float(rounding), allow_nan=False).encode('ascii')
    block_header = b'CPBLOCK1\0' + struct.pack('<Q', block_size) + quantum + b'\0'
    blocks, skipped = [], 0
    for start in starts:
        q = _quantized_relative(x[start:start + block_size], rounding)
        if not q.any():
            skipped += 1
            continue
        blocks.append(dict(start=start, sha256=hashlib.sha256(block_header + q.tobytes()).hexdigest()))
    return dict(version=1, length=len(x), rounding=float(rounding), block_size=block_size,
                block_step=step, candidate_every_offset=candidate,
                exact_sha256=hashlib.sha256(header + x.tobytes()).hexdigest(),
                offset_sign_sha256=hashlib.sha256(header + quantum + b'\0' + full_q.tobytes()).hexdigest(),
                informative=bool(full_q.any()), blocks=blocks,
                constant_blocks_skipped_for_search=skipped,
                independent_trial_proven=False,
                evidence_scope='potential_overlap_screen_only_not_scoring_eligibility')


def potential_overlap(candidate, seen, *, max_hits=64):
    """Compare to {source_id: fingerprint}; cap evidence and never prove freshness."""
    if type(max_hits) is not int or not 1 <= max_hits <= 10000:
        raise ValueError('max_hits must be1..10000')
    if not isinstance(seen, Mapping) or candidate.get('candidate_every_offset') is not True:
        raise ValueError('Seen mapping and every-offset candidate fingerprint required')
    hits, truncated = [], False
    block_index = {}

    def add(hit):
        nonlocal truncated
        if len(hits) < max_hits:
            hits.append(hit)
            return True
        truncated = True
        return False

    for source, fp in sorted(seen.items()):
        if type(source) is not str or fp.get('version') != 1 or candidate.get('version') != 1 or fp.get('rounding') != candidate.get('rounding') or fp.get('block_size') != candidate.get('block_size'):
            raise ValueError('Matching declared fingerprint versions/rounding/block sizes required')
        if candidate['informative'] and fp['informative'] and candidate['length'] == fp['length']:
            for kind, key in [('full_exact', 'exact_sha256'), ('full_offset_sign', 'offset_sign_sha256')]:
                if candidate[key] == fp[key] and not add(dict(source_id=source, kind=kind, candidate_start=0, seen_start=0, length=candidate['length'])):
                    break
        for block in fp['blocks']:
            block_index.setdefault(block['sha256'], []).append((source, block['start']))
    if not truncated:
        for block in candidate['blocks']:
            for source, start in block_index.get(block['sha256'], []):
                if not add(dict(source_id=source, kind='block_offset_sign',
                                candidate_start=block['start'], seen_start=start,
                                length=candidate['block_size'])):
                    break
            if truncated:
                break
    return dict(status='potential_overlap_found' if hits else 'no_potential_overlap_found',
                hits=hits, hits_truncated=truncated, max_hits=max_hits,
                independent_trial_proven=False, fresh_confirmed=False,
                numerical_nonmatch_is_not_independence=True,
                constant_blocks_skipped_only_for_duplicate_search=True,
                limitations='Rounded hashes are a potential-duplicate screen. Seen stride can miss short crops; a nonmatch never establishes an independent physical trial.')
