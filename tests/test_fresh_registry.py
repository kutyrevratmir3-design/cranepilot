"""Synthetic-only complete-registry and within-cohort reuse regression checks."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from scripts import fresh_confirmation as fc
from scripts.fresh_record_guard import sha256_file
from tests import test_fresh_confirmation as fixtures


class FreshRegistryTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.FreshConfirmationTests(methodName='runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def test_pruned_registry_cannot_pass_using_correct_policy_hash(self):
        f=self.f
        index=fc.read(f.index);index['sequences']={'seen-response':{'informative':True}}
        f.index.write_text(json.dumps(index));f.seal_index()
        index['sequences']={};f.index.write_text(json.dumps(index))
        with f.patches(),patch('numpy.load',side_effect=AssertionError('Registry verification cannot parse candidates')):
            with self.assertRaisesRegex(ValueError,'complete-registry seal'):fc.verify_seen_index(f.index)

    def test_changed_old_source_blocks_even_with_an_unchanged_registry(self):
        f=self.f;source=f.repo/'old-response.bin';source.write_bytes(b'SYNTHETIC-OLD-RESPONSE')
        index=fc.read(f.index);index['source_hashes']={'old-response.bin':sha256_file(source)}
        f.index.write_text(json.dumps(index));f.seal_index()
        with f.patches():fc.verify_seen_index(f.index)
        source.write_bytes(b'SYNTHETIC-ALTERED-RESPONSE')
        with f.patches(),patch('numpy.load',side_effect=AssertionError('No new response parse')):
            with self.assertRaisesRegex(ValueError,'source changed'):fc.verify_seen_index(f.index)

    def test_registry_source_escape_is_rejected(self):
        f=self.f;outside=f.root/'outside.bin';outside.write_bytes(b'SYNTHETIC-OUTSIDE')
        index=fc.read(f.index);index['source_hashes']={str(outside):sha256_file(outside)}
        f.index.write_text(json.dumps(index));f.seal_index()
        with f.patches():
            with self.assertRaisesRegex(ValueError,'source changed'):fc.verify_seen_index(f.index)

    def test_reserialized_offset_copy_inside_one_cohort_is_quarantined(self):
        f=self.f
        with np.load(f.root/'trial.npz') as z:data={k:z[k].copy() for k in z.files}
        data['theta']+=.2
        np.savez(f.root/'second.npz',**dict(reversed(list(data.items()))))
        second=copy.deepcopy(f.r);second.update(record_id='second',session_id='s2',trial_id='t2',
            condition_id='c2',independent_cluster_id='g2',file='second.npz',sha256=sha256_file(f.root/'second.npz'))
        self.assertNotEqual(second['sha256'],f.r['sha256']);f.m['records'].append(second)
        lock=f.create_lock()
        with f.patches(),patch.object(fc,'committed'),patch('cranepilot.dated_pilot.FrozenDatedPilot') as primary:
            result=fc.screen(lock,f.card)
            with self.assertRaises(ValueError):fc.score(lock,f.card)
        self.assertEqual(result['status'],'blocked');self.assertFalse(result['records'][1]['passed'])
        self.assertTrue(result['records'][1]['potential_overlaps']);primary.assert_not_called()

    def test_distinct_synthetic_trial_signals_are_not_confused_with_shared_support(self):
        f=self.f
        with np.load(f.root/'trial.npz') as z:data={k:z[k].copy() for k in z.files}
        data['theta']=np.random.default_rng(888).normal(size=800)*.01
        np.savez(f.root/'second.npz',**data)
        second=copy.deepcopy(f.r);second.update(record_id='second',session_id='s2',trial_id='t2',
            condition_id='c2',independent_cluster_id='g2',file='second.npz',sha256=sha256_file(f.root/'second.npz'))
        f.m['records'].append(second);lock=f.create_lock()
        with f.patches(),patch.object(fc,'committed'):result=fc.screen(lock,f.card)
        self.assertEqual(result['status'],'schema_screened_no_detected_overlap')
        self.assertTrue(result['nonmatch_is_not_independence'])


if __name__=='__main__':unittest.main()
