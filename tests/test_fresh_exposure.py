"""Synthetic-only cross-directory protection against reusing performance cohorts."""
import copy
import json
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts import fresh_confirmation as fc
from scripts.fresh_record_guard import canonical_json_hash, sha256_file
from tests import test_fresh_score_e2e as e2e_fixtures


class FreshPerformanceExposureTests(unittest.TestCase):
    def setUp(self):
        self.subject = self.new_fixture()
        self.subject.configure_record()
        self.ledger = self.subject.fixture.repo / 'data/private_confirmation_20261108/performance_exposures'

    def new_fixture(self):
        subject = e2e_fixtures.FreshScoreEndToEndTests(methodName='runTest')
        subject.setUp()
        self.addCleanup(subject.doCleanups)
        return subject

    def second_cohort(self, *, change_ids, reserialize, angle_offset=0.):
        """Different private root, but the same repository-wide exposure ledger."""
        subject = self.new_fixture(); subject.configure_record()
        first = self.subject.fixture; second = subject.fixture
        second.repo = first.repo
        second.policy = first.policy
        second.contract = first.contract
        second.p = copy.deepcopy(first.p)
        second.card = first.repo / 'docs' / 'second_card.json'
        index = fc.read(second.index); index['policy_sha256'] = sha256_file(first.policy)
        second.index.write_text(json.dumps(index))
        subject.data = {key: value.copy() for key, value in self.subject.data.items()}
        if angle_offset:subject.data['theta'] += angle_offset
        if reserialize:
            # Stored rather than compressed members and reversed member order change
            # NPZ bytes without creating another physical observation.
            np.savez(second.root / 'trial.npz', **dict(reversed(list(subject.data.items()))))
        else:
            shutil.copyfile(first.root / 'trial.npz', second.root / 'trial.npz')
        second.r['sha256'] = sha256_file(second.root / 'trial.npz')
        second.r['record_id'] = 'second-record'
        if change_ids:
            second.m['source'].update(source_id='renamed-source', rig_id='renamed-rig')
            second.r.update(session_id='renamed-session', trial_id='renamed-trial',
                            condition_id='renamed-condition', independent_cluster_id='renamed-cluster')
        return subject

    def metadata_lock(self, subject):
        """Real lock, reusing the existing fixture's exact structured-review schema."""
        f = subject.fixture
        contract_sha = sha256_file(f.contract)
        for record in f.m['records']:
            for reference in ['xr', 'Jr']:
                if reference in record['channels']:
                    record['channels'][reference]['training_contract_sha256'] = contract_sha
        f.manifest.write_text(json.dumps(f.m))
        review = copy.deepcopy(fc.read(self.subject.fixture.review))
        review.update(source_id=f.m['source']['source_id'], rig_id=f.m['source']['rig_id'],
            metadata_sha256=canonical_json_hash(f.m), training_contract_sha256=contract_sha,
            original_source_files={record['file']: record['sha256'] for record in f.m['records']})
        f.review.write_text(json.dumps(review))
        commitments = {'policy.json': sha256_file(f.policy), f.p['training_contract']: contract_sha}
        with f.patches(), patch.object(fc, 'committed'), \
             patch.object(fc, 'git_head', return_value='synthetic-exposure-test-head'), \
             patch.object(fc, 'policy', return_value=f.p), \
             patch.object(fc, 'bundle_commitments', return_value=commitments):
            return fc.lock(f.manifest, f.root, f.index, f.review, f.card)

    def successful_exposure(self):
        lock = self.subject.screened_lock()
        result = self.subject.run_score(lock)
        self.assertEqual(result['status'], 'complete')
        files = list(self.ledger.glob('*.json'))
        self.assertEqual(len(files), 1)
        entry = fc.read(files[0])
        self.assertEqual(entry['lock_sha256'], sha256_file(lock))
        self.assertEqual(entry['access'], 'performance_attempt')
        self.assertEqual(entry['records'][0]['sha256'], self.subject.fixture.r['sha256'])
        self.assertEqual((entry['records'][0]['session_id'], entry['records'][0]['trial_id']), ('s', 't'))
        self.assertTrue(entry['records'][0]['fingerprints']['theta']['informative'])
        return lock

    def test_identical_bytes_under_another_directory_and_new_ids_cannot_lock_as_fresh(self):
        original = self.successful_exposure()
        other = self.second_cohort(change_ids=True, reserialize=False)
        self.assertNotEqual(other.fixture.root, self.subject.fixture.root)
        self.assertEqual(other.fixture.r['sha256'], self.subject.fixture.r['sha256'])
        with patch('numpy.load', side_effect=AssertionError('Metadata must not parse numerical observations')):
            result = self.metadata_lock(other)
            with self.subject.fixture.patches(), patch.object(fc, 'committed'):
                # Verification of the original frozen experiment excludes only its
                # own reservation; it is not a second fresh confirmation.
                verified = fc.verify_lock(original, self.subject.fixture.card)
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['metadata_audit']['metadata_declarations_complete'])
        self.assertNotIn('private_lock', result)
        self.assertEqual(verified['_verified_lock_sha256'], sha256_file(original))

    def test_new_serialization_cannot_hide_a_previously_used_physical_trial_identity(self):
        self.successful_exposure()
        other = self.second_cohort(change_ids=False, reserialize=True)
        self.assertNotEqual(other.fixture.r['sha256'], self.subject.fixture.r['sha256'])
        with patch('numpy.load', side_effect=AssertionError('Metadata must not parse numerical observations')):
            result = self.metadata_lock(other)
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('Physical trial', result['reason'])
        self.assertNotIn('private_lock', result)

    def test_renamed_and_reserialized_waveform_is_quarantined_by_numeric_screen(self):
        self.successful_exposure()
        other = self.second_cohort(change_ids=True, reserialize=True, angle_offset=.2)
        self.assertNotEqual(other.fixture.r['sha256'], self.subject.fixture.r['sha256'])
        with patch('numpy.load', side_effect=AssertionError('No numerical observation parsing during lock')):
            locked = self.metadata_lock(other)
        self.assertEqual(locked['status'], 'locked_not_scored')
        f = other.fixture
        with f.patches(), patch.object(fc, 'committed'), \
             patch('cranepilot.dated_pilot.FrozenDatedPilot') as primary_constructor, \
             patch('cranepilot.matched_dated_pilot.FrozenDenseReference') as reference_constructor:
            report = fc.screen(locked['private_lock'], f.card)
            self.assertEqual(report['status'], 'blocked')
            self.assertTrue(report['records'][0]['potential_overlaps'])
            self.assertFalse(report['candidate_scores_computed'])
            with self.assertRaises(ValueError):fc.score(locked['private_lock'], f.card)
        primary_constructor.assert_not_called()
        reference_constructor.assert_not_called()
        self.assertEqual(len(list(self.ledger.glob('*.json'))), 1)

    def test_failed_first_forecast_reserves_the_full_cohort_before_predictor_entry(self):
        f = self.subject.fixture
        # Two declared trials exercise cohort reservation before either has returned
        # a forecast. These are explicitly synthetic fixture declarations.
        data = dict(self.subject.data, theta=np.random.default_rng(909).normal(size=800)*.01,
                    software_fixture_marker=np.array([2]))
        np.savez(f.root / 'second.npz', **data)
        second = copy.deepcopy(f.r)
        second.update(record_id='second', session_id='s2', trial_id='t2',
                      file='second.npz', sha256=sha256_file(f.root / 'second.npz'))
        f.m['records'].append(second)
        lock = self.subject.screened_lock()
        calls = []
        def forecast(history, **kwargs):
            files = list(self.ledger.glob('*.json'))
            self.assertEqual(len(files), 1)
            event = fc.read(files[0])
            self.assertEqual({r['sha256'] for r in event['records']}, {f.r['sha256'], second['sha256']})
            self.assertEqual({(r['session_id'], r['trial_id']) for r in event['records']}, {('s', 't'), ('s2', 't2')})
            self.assertTrue(all(r['fingerprints'] for r in event['records']))
            calls.append(1)
            if len(calls)==1:raise RuntimeError('SYNTHETIC forced first forecast failure')
            return dict(forecast_rad=np.zeros((len(history['theta']),7)),horizons_s=np.array(f.p['horizons_raw'])*.01)
        reference=SimpleNamespace(forecast=lambda history,**kwargs:dict(forecast_rad=np.zeros((len(history['theta']),7)),horizons_s=np.array(f.p['horizons_raw'])*.01))
        result = self.subject.run_score(lock, candidate=SimpleNamespace(forecast=Mock(side_effect=forecast)),reference=reference)
        self.assertEqual(result['status'], 'failed_coverage')
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(len(result['failures']), 1)
        self.assertFalse(result.get('statistics', {}).get('statistical_gate_passed', False))
        other = self.second_cohort(change_ids=True, reserialize=False)
        with patch('numpy.load', side_effect=AssertionError('No numerical parse on reused metadata')):
            blocked = self.metadata_lock(other)
        self.assertEqual(blocked['status'], 'blocked')

    def test_constructor_failure_before_any_forecast_does_not_create_exposure(self):
        lock = self.subject.screened_lock(); f = self.subject.fixture
        candidate = SimpleNamespace(forecast=Mock(side_effect=AssertionError('No forecast should be called')))
        with f.patches(), patch.object(fc, 'committed'), \
             patch('cranepilot.dated_pilot.FrozenDatedPilot', return_value=candidate), \
             patch('cranepilot.matched_dated_pilot.FrozenDenseReference',
                   side_effect=RuntimeError('SYNTHETIC constructor failure')):
            with self.assertRaisesRegex(RuntimeError, 'constructor failure'):fc.score(lock, f.card)
        candidate.forecast.assert_not_called()
        self.assertEqual(list(self.ledger.glob('*.json')), [])


if __name__ == '__main__':unittest.main()
