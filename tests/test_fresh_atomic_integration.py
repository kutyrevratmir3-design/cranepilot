"""Synthetic-only integration of atomic reservation and prior-event commitments."""
from contextlib import contextmanager
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import fresh_confirmation as fc
from scripts.fresh_exposure_guard import LEASE_NAME, ledger_transaction
from scripts.fresh_record_guard import sha256_file
from tests import test_fresh_score_e2e as fixture_module


class FreshAtomicIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Compose the software fixture, without inheriting/re-discovering its tests.
        self.subject = fixture_module.FreshScoreEndToEndTests(methodName='runTest')
        self.subject.setUp()
        self.addCleanup(self.subject.doCleanups)
        self.subject.configure_record()
        self.f = self.subject.fixture
        self.ledger = self.f.repo / 'data/private_confirmation_20261108/performance_exposures'

    def prior_event(self, *, fingerprints=None):
        return dict(lock_sha256='a' * 64, source_id='synthetic-prior-source',
                    rig_id='synthetic-prior-rig', access='performance_attempt',
                    records=[dict(record_id='prior-record', session_id='prior-session',
                                  trial_id='prior-trial', sha256='b' * 64,
                                  fingerprints=fingerprints or {})])

    def locked_with_prior_event(self):
        event = self.prior_event()
        path = self.ledger / (event['lock_sha256'] + '.json')
        fc.write_new(path, event)
        lock = self.subject.screened_lock()
        self.assertEqual(fc.read(lock)['prior_performance_exposure_commitments'],
                         {path.name: sha256_file(path)})
        return lock, path, event

    def assert_integrity_blocked_before_models(self, lock, message):
        with self.f.patches(), patch.object(fc, 'committed'), \
             patch('numpy.load', side_effect=AssertionError('Prior-event check must precede numeric parsing')), \
             patch('cranepilot.dated_pilot.FrozenDatedPilot') as primary_constructor, \
             patch('cranepilot.matched_dated_pilot.FrozenDenseReference') as reference_constructor:
            with self.assertRaisesRegex(ValueError, message):
                fc.verify_lock(lock, self.f.card)
            with self.assertRaisesRegex(ValueError, message):
                fc.score(lock, self.f.card)
        primary_constructor.assert_not_called()
        reference_constructor.assert_not_called()
        self.assertFalse((lock.parent / 'scoring').exists())

    def test_foreign_overlap_arriving_after_initial_checks_blocks_under_real_lease(self):
        lock = self.subject.screened_lock()
        screened = fc.read(lock.parent / 'screen.json')
        # A different serialization/label can evade SHA and identity checks. Its
        # previously exposed response still must fail the final numerical screen.
        event = self.prior_event(fingerprints=copy.deepcopy(
            screened['records'][0]['numeric_fingerprints']))
        foreign = self.ledger / (event['lock_sha256'] + '.json')
        self.assertNotEqual(event['records'][0]['sha256'], self.f.r['sha256'])
        primary = SimpleNamespace(forecast=Mock(side_effect=AssertionError('No primary forecast permitted')))
        reference = SimpleNamespace(forecast=Mock(side_effect=AssertionError('No reference forecast permitted')))
        checks_under_lease = []
        original_verify = fc.verify_lock

        @contextmanager
        def reserve_intervening_cohort(root):
            self.assertEqual(root, self.ledger)
            # Simulate another process winning and releasing the actual atomic
            # lease after this score's initial verification, before its reservation.
            with ledger_transaction(root):
                fc.write_new(foreign, event)
            with ledger_transaction(root) as held:
                yield held

        def observe_verification(*args, **kwargs):
            checks_under_lease.append((self.ledger / LEASE_NAME).is_dir())
            return original_verify(*args, **kwargs)

        with self.f.patches(), patch.object(fc, 'committed'), \
             patch.object(fc, 'ledger_transaction', side_effect=reserve_intervening_cohort), \
             patch.object(fc, 'verify_lock', side_effect=observe_verification), \
             patch('cranepilot.dated_pilot.FrozenDatedPilot', return_value=primary), \
             patch('cranepilot.matched_dated_pilot.FrozenDenseReference', return_value=reference):
            result = fc.score(lock, self.f.card)
        self.assertEqual(checks_under_lease, [False, True])
        primary.forecast.assert_not_called()
        reference.forecast.assert_not_called()
        self.assertEqual(result['status'], 'failed_coverage')
        self.assertFalse(result['coverage_complete'])
        self.assertFalse(result['final_goal_achieved'])
        self.assertFalse(result.get('statistics', {}).get('statistical_gate_passed', False))
        self.assertEqual(len(result['failures']), 1)
        self.assertIn('freshness changed', result['failures'][0]['failure'])
        # The foreign event remains; this rejected attempt never reserved the
        # cohort and the real context manager released its own empty lease.
        self.assertEqual(list(self.ledger.glob('*.json')), [foreign])
        self.assertFalse((self.ledger / LEASE_NAME).exists())
        self.assertFalse((self.ledger / (sha256_file(lock) + '.json')).exists())
        self.assertEqual(fc.read(lock.parent / 'scoring/score.json'), result)

    def test_deleted_previously_locked_event_fails_actual_verification_before_models(self):
        lock, event_path, _ = self.locked_with_prior_event()
        event_path.unlink()
        self.assert_integrity_blocked_before_models(lock, 'exposure is missing')

    def test_modified_previously_locked_event_fails_actual_verification_before_models(self):
        lock, event_path, event = self.locked_with_prior_event()
        event['access'] = 'SYNTHETIC altered event'
        event_path.write_text(json.dumps(event))
        self.assert_integrity_blocked_before_models(lock, 'exposure changed')


if __name__ == '__main__':
    unittest.main()
