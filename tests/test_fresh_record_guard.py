import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.fresh_record_guard import (sha256_file, canonical_json_hash,
    metadata_preflight, fingerprint_sequence, potential_overlap)


class FreshRecordGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / 'trial.bin'
        self.file.write_bytes(b'arbitrary bytes: numerical schema remains unverified')

    def manifest(self, scope='synchronous_acquisition_offline'):
        acq = dict(time_column='t', clock_id='daq', unit='s', evidence='acquisition specification')
        if scope == 'synchronous_acquisition_offline':
            acq.update(synchronized=True, synchronization_evidence='common acquisition grid certificate')
            age, arrival = 0, None
        else:
            age = 15
            arrival = dict(time_column='arrival_t', clock_id='daq', unit='s',
                evidence='actual arrival logger specification',
                nonangle_availability_evidence='nonangle acquisition and receipt certified at origin',
                nonangle_time_columns=dict(L='L_arrival',x='x_arrival',xr='xr_arrival'))
        record = dict(record_id='record1', session_id='session1', trial_id='trial1',
            condition_id='condition1', independent_cluster_id='cluster1', file='trial.bin', sha256=sha256_file(self.file),
            family='trolley', axes=['theta'], sample_rate_hz=100,
            clock_scope=scope, angle_age_samples=age, history_s=4 + age*.01,
            target_tail_s=1.5, genuine_history=True, encoder_quantum_rad=.0001,
            filters=dict(kind='none', description='unfiltered source', evidence='logger specification'),
            acquisition=acq, arrival=arrival,
            channels=dict(theta=dict(column='theta', unit='rad'),
                L=dict(column='L', unit='m'), x=dict(column='x', unit='m'),
                xr=dict(column='reference', unit='V', training_unit='V',
                    semantics='documented_motor_voltage_reference',
                    training_semantics='documented_motor_voltage_reference',
                    compatibility_evidence='explicit training and source acquisition contract certificate')))
        return dict(schema_version=1, source=dict(source_id='source1', rig_id='rig1',
            physical_recordings=True, provenance_evidence='physical recording archive specification',
            research_use_permitted=True, research_use_evidence='research-use permission declaration'),
            records=[record])

    def test_stream_hash_and_canonical_json(self):
        self.assertEqual(sha256_file(self.file, 3), hashlib.sha256(self.file.read_bytes()).hexdigest())
        self.assertEqual(canonical_json_hash(dict(b=2, a=[1, 'unit'])), canonical_json_hash(dict(a=[1, 'unit'], b=2)))
        for obj in [dict(x=float('nan')), {1: 'nonstring'}, dict(x=np.array([1.]))]:
            with self.assertRaises(ValueError):
                canonical_json_hash(obj)

    def test_preflight_never_parses_numeric_data_or_claims_eligibility(self):
        manifest = self.manifest()
        with patch('numpy.load', side_effect=AssertionError('numeric loader forbidden')), patch('scripts.fresh_record_guard.fingerprint_sequence', side_effect=AssertionError('postlock only')):
            result = metadata_preflight(manifest, self.root)
        self.assertEqual(result['status'], 'metadata_declared_only')
        self.assertTrue(result['metadata_declarations_complete'])
        self.assertFalse(result['numerical_schema_verified'])
        self.assertFalse(result['eligible'])
        self.assertFalse(result['fresh_confirmed'])
        self.assertFalse(result['raw_release_permitted'])
        self.assertTrue(result['never_independent_certificate'])
        self.assertFalse(result['clock_scopes'][0]['physical_arrival_age_declared_known'])
        self.assertFalse(result['clock_scopes'][0]['physical_arrival_age_verified'])
        self.assertEqual(result['clock_scopes'][0]['angle_age_interpretation'], 'acquisition_alignment_only')

    def test_known_actual_arrival_mode_requires_evidence(self):
        manifest = self.manifest('known_arrival_age')
        self.assertTrue(metadata_preflight(manifest, self.root)['metadata_declarations_complete'])
        manifest['records'][0]['arrival']['clock_id'] = 'other_clock'
        self.assertEqual(metadata_preflight(manifest, self.root)['status'], 'blocked')
        manifest['records'][0]['arrival']['clock_relation_evidence'] = 'clock synchronization certificate'
        self.assertTrue(metadata_preflight(manifest, self.root)['metadata_declarations_complete'])

    def test_physical_provenance_and_research_permission_are_explicit(self):
        for key in ['physical_recordings','provenance_evidence','research_use_permitted','research_use_evidence','rig_id']:
            manifest=self.manifest();manifest['source'].pop(key)
            self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        manifest=self.manifest();manifest['source']['physical_recordings']=1
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        self.assertFalse(metadata_preflight(self.manifest(),self.root)['raw_release_permitted'])

    def test_known_arrival_requires_complete_nonnumeric_availability_mapping(self):
        manifest=self.manifest('known_arrival_age');arrival=manifest['records'][0]['arrival']
        self.assertTrue(metadata_preflight(manifest,self.root)['metadata_declarations_complete'])
        arrival['angle_time_columns']={'theta':'theta_arrival'}
        self.assertTrue(metadata_preflight(manifest,self.root)['metadata_declarations_complete'])
        arrival['angle_time_columns']={'beta':'beta_arrival'}
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        arrival.pop('angle_time_columns');arrival['nonangle_time_columns'].pop('L')
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        arrival.pop('nonangle_time_columns')
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')

    def test_optional_reference_contract_commitment_checks_syntax_only(self):
        manifest=self.manifest();ref=manifest['records'][0]['channels']['xr']
        for digest in ['notahash','f'*63,'G'*64,None]:
            ref['training_contract_sha256']=digest
            self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        ref['training_contract_sha256']='f'*64
        result=metadata_preflight(manifest,self.root)
        self.assertTrue(result['metadata_declarations_complete'])
        self.assertTrue(result['never_independent_certificate'])

    def test_unknown_age_scope_and_missing_arrival_never_default_to_zero(self):
        for age in [None, True, .5, -1, 26]:
            manifest = self.manifest();manifest['records'][0]['angle_age_samples'] = age
            self.assertEqual(metadata_preflight(manifest, self.root)['status'], 'blocked')
        for field in ['clock_scope', 'angle_age_samples', 'arrival']:
            manifest = self.manifest();manifest['records'][0].pop(field)
            self.assertEqual(metadata_preflight(manifest, self.root)['status'], 'blocked')
        manifest = self.manifest('known_arrival_age');manifest['records'][0]['arrival'] = None
        self.assertEqual(metadata_preflight(manifest, self.root)['status'], 'blocked')

    def test_renamed_raw_bytes_and_known_seen_hash_are_blocked(self):
        renamed = self.root / 'renamed.bin';renamed.write_bytes(self.file.read_bytes())
        manifest = self.manifest();other = copy.deepcopy(manifest['records'][0]);other.update(record_id='r2', trial_id='t2', file='renamed.bin');manifest['records'].append(other)
        r = metadata_preflight(manifest, self.root)
        self.assertEqual(r['status'], 'blocked')
        self.assertTrue(any('Repeated raw bytes' in x['message'] for x in r['issues']))
        r = metadata_preflight(self.manifest(), self.root, seen_file_hashes=[sha256_file(self.file)])
        self.assertEqual(r['status'], 'blocked')

    def test_record_identity_duplicates_and_repeated_conditions(self):
        manifest = self.manifest();(self.root/'second.bin').write_bytes(b'other bytes')
        other = copy.deepcopy(manifest['records'][0]);other.update(record_id='r2', file='second.bin', sha256=sha256_file(self.root/'second.bin'));manifest['records'].append(other)
        self.assertEqual(metadata_preflight(manifest, self.root)['status'], 'blocked')
        other['trial_id'] = 'different_trial'
        self.assertTrue(metadata_preflight(manifest, self.root)['metadata_declarations_complete'])
        other['trial_id'] = 'trial1';other['condition_id'] = 'different_condition'
        self.assertEqual(metadata_preflight(manifest, self.root)['status'], 'blocked')

    def test_family_axes_once_per_trial_and_per_axis_quantization(self):
        manifest=self.manifest();record=manifest['records'][0]
        record.update(family='jib',axes=['theta','beta'],encoder_quantum_rad=dict(theta=.0001,beta=.0002))
        record['channels'].pop('x');ref=record['channels'].pop('xr')
        record['channels'].update(beta=dict(column='beta',unit='rad'),J=dict(column='J',unit='rad'),Jr=ref)
        self.assertTrue(metadata_preflight(manifest,self.root)['metadata_declarations_complete'])
        record['axes']=['theta']
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        record['axes']=['theta','beta'];record['encoder_quantum_rad']={'theta':.0001}
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')

    def test_opaque_native_reference_is_not_relabelled_as_physical_si(self):
        manifest=self.manifest();ref=manifest['records'][0]['channels']['xr']
        ref.update(unit='native',training_unit='native',semantics='same_Inteco_export_xr_contract',training_semantics='same_Inteco_export_xr_contract',compatibility_evidence='exact exporter/rig/firmware/schema certificate')
        result=metadata_preflight(manifest,self.root)
        self.assertTrue(result['metadata_declarations_complete'])
        self.assertFalse(result['eligible']);self.assertTrue(result['never_independent_certificate'])
        ref['training_semantics']='different_gantry_position_setpoint'
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')

    def test_malformed_metadata_fails_without_reading_a_schema(self):
        for manifest in [None,[],{},dict(schema_version=True,source=[],records=[None])]:
            self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        for key in ['unit','training_unit','training_semantics']:
            manifest=self.manifest();manifest['records'][0]['channels']['xr'][key]=[]
            self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        manifest=self.manifest();manifest['records'][0]['sample_rate_hz']=10**1000
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')

    def test_unknown_future_fields_units_and_reference_mismatch_block(self):
        changes = [lambda r:r.update(future_angles='forbidden'),
            lambda r:r['channels']['theta'].update(unit='deg'),
            lambda r:r['channels'].pop('L'),
            lambda r:r['channels']['xr'].update(semantics='position_setpoint'),
            lambda r:r['channels']['xr'].update(training_unit='m'),
            lambda r:r['channels']['xr'].pop('compatibility_evidence'),
            lambda r:r.update(history_s=3.99), lambda r:r.update(target_tail_s=1.49),
            lambda r:r.update(genuine_history=False), lambda r:r.update(encoder_quantum_rad=None),
            lambda r:r['filters'].update(kind='noncausal'),
            lambda r:r['acquisition'].update(synchronized=False)]
        for change in changes:
            manifest=self.manifest();change(manifest['records'][0])
            self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')

    def test_paths_symlink_escape_and_wrong_bytes_are_blocked(self):
        for path in ['../outside.bin', '/outside.bin', 'sub/../trial.bin', 'C:\\trial.bin', './trial.bin']:
            manifest=self.manifest();manifest['records'][0]['file']=path
            self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        outside=tempfile.NamedTemporaryFile();self.addCleanup(outside.close)
        (self.root/'escape').symlink_to(outside.name)
        manifest=self.manifest();manifest['records'][0]['file']='escape'
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')
        manifest=self.manifest();manifest['records'][0]['sha256']='0'*64
        self.assertEqual(metadata_preflight(manifest,self.root)['status'],'blocked')

    def test_error_output_is_bounded(self):
        manifest=self.manifest();manifest['records']=[{} for _ in range(20)]
        r=metadata_preflight(manifest,self.root,max_records=3,max_issues=4)
        self.assertEqual(len(r['issues']),4);self.assertTrue(r['issues_truncated'])
        self.assertFalse(r['metadata_declarations_complete'])

    def fp(self, array, candidate=False, **kwargs):
        return fingerprint_sequence(np.asarray(array), candidate=candidate, lock_verified=True, **kwargs)

    def test_postlock_required_before_array_access(self):
        class Poison:
            def __array__(self):raise AssertionError('array inspected before lock')
        with self.assertRaisesRegex(ValueError,'lock'):
            fingerprint_sequence(Poison())
        for array in [np.array([[1.]]),np.array([True]),np.array(['1']),np.array([np.inf]),np.array([])]:
            with self.assertRaises(ValueError):fingerprint_sequence(array,lock_verified=True)

    def test_numerically_equal_formats_have_exact_full_hash(self):
        a=np.array([.1,.2,.3,.4]);parsed=np.asarray(json.loads(json.dumps(a.tolist())))
        self.assertEqual(self.fp(a)['exact_sha256'],self.fp(parsed)['exact_sha256'])
        self.assertEqual(self.fp(np.array([-0.,1.]))['exact_sha256'],self.fp(np.array([0.,1.]))['exact_sha256'])

    def test_crop_offset_and_sign_overlap_without_response_scoring(self):
        rng=np.random.default_rng(8);source=rng.normal(size=1200);original=source.copy();seen=self.fp(source)
        for candidate in [source.copy(),source+2.5,-source,3-source[37:701]]:
            fp=self.fp(candidate,candidate=True);hits=potential_overlap(fp,{'seen':seen})
            self.assertEqual(hits['status'],'potential_overlap_found')
            self.assertFalse(hits['independent_trial_proven'])
            self.assertFalse(hits['fresh_confirmed'])
        np.testing.assert_array_equal(source,original)
        self.assertEqual(self.fp(source)['offset_sign_sha256'],self.fp(2.5-source)['offset_sign_sha256'])

    def test_nonmatch_is_not_independence_and_constants_not_eligibility(self):
        rng=np.random.default_rng(9);r=potential_overlap(self.fp(rng.normal(size=1000),True),{'seen':self.fp(rng.normal(size=1000))})
        self.assertEqual(r['status'],'no_potential_overlap_found')
        self.assertTrue(r['numerical_nonmatch_is_not_independence']);self.assertFalse(r['independent_trial_proven'])
        constant=self.fp(np.ones(1000),True);self.assertFalse(constant['informative']);self.assertEqual(constant['blocks'],[])
        self.assertGreater(constant['constant_blocks_skipped_for_search'],0)
        self.assertEqual(potential_overlap(constant,{'seen':self.fp(np.ones(1000))})['hits'],[])

    def test_hits_bounds_grid_compatibility_and_sequence_bounds(self):
        x=np.tile(np.arange(40,dtype=float),40);seen=self.fp(x,block_size=32)
        r=potential_overlap(self.fp(x,True,block_size=32),{'a':seen},max_hits=2)
        self.assertEqual(len(r['hits']),2);self.assertTrue(r['hits_truncated'])
        with self.assertRaises(ValueError):potential_overlap(self.fp(x,True,rounding=1e-7),{'a':self.fp(x)})
        with self.assertRaises(ValueError):self.fp(x,True,max_blocks=2)
        with self.assertRaises(ValueError):self.fp(np.array([0.,1e300]))

    def test_short_unaligned_crop_limitation_is_not_a_freshness_certificate(self):
        source=np.random.default_rng(10).normal(size=600)
        result=potential_overlap(self.fp(source[1:257],True),{'seen':self.fp(source)})
        self.assertEqual(result['status'],'no_potential_overlap_found')
        self.assertFalse(result['independent_trial_proven'])
        self.assertTrue(result['numerical_nonmatch_is_not_independence'])


if __name__=='__main__':unittest.main()
