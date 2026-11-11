import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cranepilot.data import sha256
from cranepilot.matched_dated_pilot import FrozenDenseReference, member_path
from cranepilot.physics import LengthCalibration
from scripts import dated_forecast as dated
from scripts import dense_reference as reference
from scripts import drive_conditioned as drive


class DenseReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.grids, cls.primary, cls.task = reference.setup()

    def history(self, family='jib', axis='theta', age=5):
        name = family + '_' + axis
        cal = LengthCalibration(**self.primary.base.pilot.cases[name]['calibration'])
        t = -np.arange(401) * .01
        raw = {axis: (.05 * np.cos(cal.omega(.4) * (t - age * .01) + .2))[None],
               'L': np.full((1, 401), .4)}
        for coordinate in (['x'] if family == 'trolley' else ['J'] if family == 'jib' else ['x', 'J']):
            raw[coordinate] = (.003 * np.sin(t))[None]
            raw['xr' if coordinate == 'x' else 'Jr'] = np.zeros((1, 401))
        return raw

    def fixture_bundle(self, root):
        specs = reference.pool_specs(self.cfg, 'jib_theta')
        # Parent selects "old" here. This fixture deliberately uses the previously omitted dense4s model.
        self.assertEqual(self.primary.base.models['jib_theta']['reference'], 'old')
        key = 'drive4/quadratic_0.25/arx_io_lpv_direct_1e-06.npz'
        spec = specs[key].copy()
        shutil.copyfile(spec['file'], root / 'model.npz')
        spec['weight_file'] = 'model.npz'
        manifest = dict(cases={'jib_theta': dict(reference={str(age): key for age in self.cfg['train_ages_raw']},
                                               specs={key: spec})},
            train_ages_raw=self.cfg['train_ages_raw'], files={'model.npz': sha256(root / 'model.npz')},
            source_hashes={}, primary_manifest_sha256=sha256(Path(self.cfg['candidate_bundle']) / 'manifest.json'),
            scope='Synthetic unit-test comparator fixture')
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return key, specs[key]

    def test_complete_pool_does_not_follow_selected_parent_fallback(self):
        for name in self.primary.manifest['cases']:
            specs = reference.pool_specs(self.cfg, name)
            self.assertEqual(len(specs), 304)
            self.assertEqual(sum(key.startswith('drive4/') for key in specs), 192)
            self.assertEqual(sum(key.startswith('augmented/') for key in specs), 72)
            for prefix in ['drive3/', 'legacy3/', 'legacy4/']:
                self.assertEqual(sum(key.startswith(prefix) for key in specs), 8)
            self.assertEqual(sum(key.startswith('previous/') for key in specs), 16)
            self.assertTrue(all(Path(spec['file']).is_file() for spec in specs.values()
                                if 'file' in spec and spec['source'] != 'dense'))

    def test_dense_history_origin_support_is_exact_original_subset(self):
        previous = reference.origin_indices(9901, 4., 25, self.task)
        dense = reference.origin_indices(9901, 4.25, 25, self.task)
        evaluation = reference.origin_indices(9901, 4.25, 100, self.task)
        np.testing.assert_array_equal(dense, previous[1:])
        self.assertEqual((len(previous), len(dense), len(evaluation)), (375, 374, 94))
        self.assertEqual(dense.min() - 400 - max(self.cfg['evaluation_ages_raw']), 0)
        horizons = np.asarray(self.task['horizons_raw'])
        self.assertTrue(np.isin(dense[:, None] + horizons, previous[:, None] + horizons).all())

    def test_support_audit_checks_roles_and_original_prior_counts(self):
        ids = self.task['families']['trolley']['train']
        records = {sid: SimpleNamespace(time=np.arange(9901) * .01,
            metadata=dict(file_name='synthetic schema fixture', sha256='0' * 64)) for sid in ids}
        rows = reference.support_audit(records, 'trolley_theta', self.task, self.cfg)
        self.assertEqual(sum(row['original_windows'] for row in rows), 3750)
        self.assertEqual(sum(row['augmented_base_windows'] for row in rows), 3740)
        self.assertTrue(all(row['original_only_origins'] == [400] for row in rows))
        with self.assertRaises(ValueError):reference.support_audit({sid: records[sid] for sid in ids[:-1]},
                                                                 'trolley_theta', self.task, self.cfg)

    def test_reference_choice_averages_both_noises_and_has_fixed_tie_rule(self):
        scores = {0: {'b': [1., 1.], 'a': [0., 2.]}, 5: {'a': [4., 0.], 'b': [1., 1.]}}
        choices = reference.choose_reference(scores)
        self.assertEqual(choices['0']['key'], 'a')
        self.assertEqual(choices['5']['key'], 'b')
        for bad in [{0: {'a': [1.]}}, {0: {'a': [1., np.nan]}}]:
            with self.assertRaises(ValueError):reference.choose_reference(bad)
        self.assertEqual(reference.nearest(self.cfg['train_ages_raw'], 1), 0)
        self.assertEqual(reference.nearest(self.cfg['train_ages_raw'], 3), 2)
        self.assertEqual(reference.nearest(self.cfg['train_ages_raw'], 25), 20)

    def test_comparator_is_portable_and_matches_omitted_dense_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); key, spec = self.fixture_bundle(root)
            source_model = drive.load(Path(spec['file']))
            raw = self.history(age=25); before = copy.deepcopy(raw)
            expected = reference.reference_prediction(spec, {spec['file']: source_model}, raw, 25,
                                                       'theta', self.primary.base, 'jib_theta')
            np_load = np.load; primary_root = Path(self.cfg['candidate_bundle']).resolve()
            def restricted(path, *args, **kwargs):
                actual = Path(path).resolve()
                if not any(actual.is_relative_to(p) for p in [root, primary_root]):
                    raise AssertionError('Comparator inference read a dataset/ancestral model')
                return np_load(path, *args, **kwargs)
            with patch('numpy.load', restricted):
                api = FrozenDenseReference(root, primary_root)
                actual = api.forecast(raw, family='jib', axis='theta', angle_age_samples=25)
            np.testing.assert_array_equal(actual['forecast_rad'], expected)
            self.assertEqual(actual['reference_key'], key)
            self.assertEqual(actual['matched_train_age_raw'], 20)
            for field in raw:np.testing.assert_array_equal(raw[field], before[field])

    def test_new_reference_cannot_change_primary_predictions(self):
        raw = self.history(age=5)
        before = self.primary.forecast(raw, family='jib', axis='theta', angle_age_samples=5)['forecast_rad']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture_bundle(root)
            FrozenDenseReference(root, self.cfg['candidate_bundle']).forecast(raw, family='jib', axis='theta', angle_age_samples=5)
        after = self.primary.forecast(raw, family='jib', axis='theta', angle_age_samples=5)['forecast_rad']
        np.testing.assert_array_equal(before, after)

    def test_known_age_required_and_future_missing_or_invalid_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture_bundle(root)
            api = FrozenDenseReference(root, self.cfg['candidate_bundle'])
            with self.assertRaises(TypeError):api.forecast(self.history(), family='jib', axis='theta')
            for age in [None, -1, 26, .5, True, np.bool_(False)]:
                with self.assertRaises(ValueError):api.forecast(self.history(), family='jib', axis='theta', angle_age_samples=age)
            with self.assertRaises(ValueError):api.forecast(dict(self.history(), future=np.ones((1, 7))),
                                                          family='jib', axis='theta', angle_age_samples=5)
            raw = self.history(); raw.pop('Jr')
            with self.assertRaises(ValueError):api.forecast(raw, family='jib', axis='theta', angle_age_samples=5)

    def test_bundle_and_primary_identity_are_checked_and_members_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture_bundle(root)
            with self.assertRaises(ValueError):member_path(root, '../outside.npz')
            (root / 'model.npz').write_bytes((root / 'model.npz').read_bytes() + b'changed')
            with self.assertRaises(ValueError):FrozenDenseReference(root, self.cfg['candidate_bundle'])

    def test_all_saved_drive_forms_accept_causal_raw_input_without_mutation(self):
        raw = self.history(age=20); before = copy.deepcopy(raw)
        specs = reference.pool_specs(self.cfg, 'jib_theta')
        keys = ['drive4/hold/arx_angle_lpv_direct_1e-06.npz',
                'drive4/hold/arx_angle_lpv_modal_residual_1e-06.npz',
                'drive3/hold', 'legacy3/hold', 'legacy4/hold']
        selected = {key: specs[key] for key in keys}; models = reference.load_pool(selected)
        values = reference.all_predictions(selected, models, raw, 20, 'theta', self.primary.base, 'jib_theta')
        self.assertEqual(set(values), set(keys))
        self.assertTrue(all(value.shape == (1, 7) and np.isfinite(value).all() for value in values.values()))
        for field in raw:np.testing.assert_array_equal(raw[field], before[field])


if __name__ == '__main__':unittest.main()
