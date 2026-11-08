"""Synthetic scorer integration only: no physical logs, real model fits or claims."""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts import fresh_confirmation as fc
from scripts.fresh_record_guard import sha256_file
from tests import test_fresh_confirmation as fixtures


class FreshScoreEndToEndTests(unittest.TestCase):
    def setUp(self):
        # Compose the existing fixture; do not inherit/re-discover its unit tests.
        self.fixture = fixtures.FreshConfirmationTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        # macOS exposes /var and /private/var aliases; the production gate resolves paths.
        for key, value in vars(self.fixture).copy().items():
            if isinstance(value, Path):setattr(self.fixture, key, value.resolve())
        self.calls = {'candidate': [], 'reference': []}
        self.data = None

    def configure_record(self, *, age=0, axes=('theta',), family='trolley'):
        f = self.fixture
        samples = np.arange(800, dtype=float)
        self.data = dict(t=samples * .01, theta=samples * .0006,
            beta=samples * -.0002, L=.4 + samples * 1e-6,
            x=samples * .0001, xr=samples * .00002,
            J=samples * .0003, Jr=samples * .00004)
        old = copy.deepcopy(f.r['channels'])
        coordinates = ['x'] if family == 'trolley' else ['J'] if family == 'jib' else ['x', 'J']
        channels = {axis: dict(column=axis, unit='rad') for axis in axes}
        channels['L'] = copy.deepcopy(old['L'])
        for coordinate in coordinates:
            channels[coordinate] = dict(column=coordinate, unit='m' if coordinate == 'x' else 'rad')
            key = 'xr' if coordinate == 'x' else 'Jr'
            channels[key] = copy.deepcopy(old.get(key, old['xr']))
            channels[key]['column'] = key
        f.r.update(family=family, axes=list(axes), channels=channels,
                   angle_age_samples=age, history_s=4 + age * .01)
        if age:
            nonangle = ['L'] + coordinates + ['xr' if key == 'x' else 'Jr' for key in coordinates]
            availability = {key: key + '_available' for key in nonangle}
            self.data['angle_arrival'] = self.data['t'] + age * .01
            for column in availability.values():self.data[column] = self.data['t'].copy()
            f.r.update(clock_scope='known_arrival_age', arrival=dict(time_column='angle_arrival',
                clock_id=f.r['acquisition']['clock_id'], unit='s',
                evidence='SYNTHETIC actual-arrival fixture, not hardware evidence',
                nonangle_availability_evidence='SYNTHETIC current-channel receipt fixture',
                nonangle_time_columns=availability))
        else:
            f.r.update(clock_scope='synchronous_acquisition_offline', arrival=None)
        np.savez_compressed(f.root / 'trial.npz', **self.data)
        f.r['sha256'] = sha256_file(f.root / 'trial.npz')

    def screened_lock(self):
        if self.data is None:self.configure_record()
        f = self.fixture
        lock = f.create_lock()
        with f.patches(), patch.object(fc, 'committed'):
            screening = fc.screen(lock, f.card)
        self.assertEqual(screening['status'], 'schema_screened_no_detected_overlap', screening)
        return lock

    def predictor(self, name, *, mutate=False, fail_first=False, wrong_horizons=False):
        f = self.fixture
        def forecast(history, *, family, axis, angle_age_samples):
            k = np.arange(f.p['first_origin_raw'], len(self.data['t']) - max(f.p['horizons_raw']),
                          f.p['origin_stride_raw'])
            coordinates = ['x'] if family == 'trolley' else ['J'] if family == 'jib' else ['x', 'J']
            keys = {axis, 'L'} | set(coordinates) | {'xr' if key == 'x' else 'Jr' for key in coordinates}
            self.assertEqual(set(history), keys)
            self.assertFalse({'target', 'future', 'future_reference'} & set(history))
            indices = k[:, None] - np.arange(401)
            for key in keys:
                expected = self.data[key][indices - angle_age_samples] if key == axis else self.data[key][indices]
                np.testing.assert_array_equal(history[key], expected)
            self.calls[name].append(dict(family=family, axis=axis, age=angle_age_samples,
                                         history={key: value.copy() for key, value in history.items()}))
            if fail_first and len(self.calls[name]) == 1:
                raise RuntimeError('SYNTHETIC forced prediction failure')
            horizons = np.asarray(f.p['horizons_raw'])
            if name == 'candidate':
                slope = .0006 if axis == 'theta' else -.0002
                # Exact extrapolation of a declared synthetic ramp, using past only.
                value = history[axis][:, :1] + slope * (angle_age_samples + horizons)[None]
            else:
                value = np.zeros((len(k), len(horizons)))
            if mutate:
                for array in history.values():array[:] = -999.
            return dict(forecast_rad=value, horizons_s=horizons * .01 + (.01 if wrong_horizons else 0.))
        return SimpleNamespace(forecast=Mock(side_effect=forecast))

    def run_score(self, lock, candidate=None, reference=None):
        f = self.fixture
        candidate = self.predictor('candidate') if candidate is None else candidate
        reference = self.predictor('reference') if reference is None else reference
        with f.patches(), patch.object(fc, 'committed'), \
             patch('cranepilot.dated_pilot.FrozenDatedPilot', return_value=candidate) as primary_constructor, \
             patch('cranepilot.matched_dated_pilot.FrozenDenseReference', return_value=reference) as reference_constructor:
            result = fc.score(lock, f.card)
        primary_constructor.assert_called_once_with(f.repo / f.p['primary_bundle'])
        reference_constructor.assert_called_once_with(f.repo / f.p['reference_bundle'],
                                                      primary_bundle=f.repo / f.p['primary_bundle'])
        return result

    def assert_blocked_before_predictors(self, lock):
        f = self.fixture
        with f.patches(), patch.object(fc, 'committed'), \
             patch('cranepilot.dated_pilot.FrozenDatedPilot') as primary_constructor, \
             patch('cranepilot.matched_dated_pilot.FrozenDenseReference') as reference_constructor:
            with self.assertRaises((ValueError, FileExistsError)):
                fc.score(lock, f.card)
        primary_constructor.assert_not_called()
        reference_constructor.assert_not_called()

    def test_real_lock_screen_score_uses_dated_angles_current_support_and_one_trial_for_axes(self):
        self.configure_record(age=15, family='jib', axes=('theta', 'beta'))
        lock = self.screened_lock()
        result = self.run_score(lock)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(len(result['trials']), 1)
        self.assertEqual(result['trials'][0]['axes'], 2)
        self.assertEqual(result['statistics']['physical_trials'], 1)
        self.assertTrue(result['statistics']['statistical_gate_passed'])
        self.assertFalse(result['final_goal_achieved'])
        self.assertFalse(result['statistics']['final_goal_achieved'])
        self.assertFalse(result['raw_or_target_publication'])
        self.assertEqual(len(self.calls['candidate']), 2)
        self.assertEqual(len(self.calls['reference']), 2)
        for one, two in zip(self.calls['candidate'], self.calls['reference']):
            self.assertEqual((one['family'], one['axis'], one['age']), (two['family'], two['axis'], two['age']))
            for field in one['history']:np.testing.assert_array_equal(one['history'][field], two['history'][field])
        stored = list((lock.parent / 'scoring').glob('*.npz'))
        self.assertEqual(len(stored), 1)
        with np.load(stored[0], allow_pickle=False) as z:
            k = z['k']; np.testing.assert_array_equal(k, [425, 525, 625])
            for axis in ['theta', 'beta']:
                expected = self.data[axis][k[:, None] + np.array(self.fixture.p['horizons_raw'])]
                np.testing.assert_array_equal(z[axis + '_target'], expected)
                np.testing.assert_allclose(z[axis + '_candidate'], expected, rtol=0, atol=1e-12)
        self.assertEqual(list((self.fixture.repo / 'docs').rglob('*.npz')), [])
        self.assertFalse(fc.read(self.fixture.card)['raw_data_or_identifiers_published'])

    def test_primary_mutation_does_not_reach_reference_or_private_targets(self):
        self.configure_record(age=15)
        lock = self.screened_lock()
        result = self.run_score(lock, candidate=self.predictor('candidate', mutate=True))
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(len(self.calls['reference']), 1)
        one = self.calls['candidate'][0]['history']; two = self.calls['reference'][0]['history']
        for key in one:np.testing.assert_array_equal(one[key], two[key])
        with np.load(next((lock.parent / 'scoring').glob('*.npz')), allow_pickle=False) as z:
            expected = self.data['theta'][z['k'][:, None] + np.array(self.fixture.p['horizons_raw'])]
            np.testing.assert_array_equal(z['theta_target'], expected)

    def test_second_score_attempt_rejected_before_constructing_or_calling_models(self):
        lock = self.screened_lock()
        self.run_score(lock)
        self.assert_blocked_before_predictors(lock)

    def test_forged_saved_screen_cannot_construct_predictors(self):
        lock = self.screened_lock()
        path = lock.parent / 'screen.json'; report = fc.read(path)
        report['records'][0]['origins'] += 1
        path.write_text(json.dumps(report))
        self.assert_blocked_before_predictors(lock)
        self.assertFalse((lock.parent / 'scoring').exists())

    def test_changed_locked_npz_cannot_construct_predictors(self):
        lock = self.screened_lock()
        changed = {key: value.copy() for key, value in self.data.items()}
        changed['theta'][700] += .3
        np.savez_compressed(self.fixture.root / 'trial.npz', **changed)
        self.assert_blocked_before_predictors(lock)
        self.assertFalse((lock.parent / 'scoring').exists())

    def test_changed_policy_cannot_construct_predictors(self):
        lock = self.screened_lock()
        changed = fc.read(self.fixture.policy); changed['minimum_relative_gain'] = 0.
        self.fixture.policy.write_text(json.dumps(changed))
        self.assert_blocked_before_predictors(lock)
        self.assertFalse((lock.parent / 'scoring').exists())

    def test_prediction_failure_is_saved_as_failed_coverage_without_statistical_pass(self):
        lock = self.screened_lock()
        result = self.run_score(lock, candidate=self.predictor('candidate', fail_first=True))
        self.assertEqual(result['status'], 'failed_coverage')
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(result['trials'], [])
        self.assertEqual(len(result['failures']), 1)
        self.assertIn('SYNTHETIC forced prediction failure', result['failures'][0]['failure'])
        self.assertFalse(result.get('statistics', {}).get('statistical_gate_passed', False))
        self.assertFalse(result['final_goal_achieved'])
        self.assertEqual(fc.read(lock.parent / 'scoring' / 'score.json'), result)
        self.assert_blocked_before_predictors(lock)

    def test_wrong_declared_horizons_cannot_pass_a_shape_only_comparison(self):
        lock = self.screened_lock()
        result = self.run_score(lock, reference=self.predictor('reference', wrong_horizons=True))
        self.assertEqual(result['status'], 'failed_coverage')
        self.assertFalse(result['coverage_complete'])
        self.assertFalse(result.get('statistics', {}).get('statistical_gate_passed', False))
        self.assertFalse(result['final_goal_achieved'])


if __name__ == '__main__':unittest.main()
