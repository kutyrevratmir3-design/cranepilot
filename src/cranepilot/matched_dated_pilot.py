"""Comparator-only frozen dense ARX; the original dated primary stays separate."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .modal_pilot import ROOT
from .regime_pilot import FrozenRegimePilot
from scripts import dated_forecast as dated
from scripts import drive_conditioned as drive
from scripts.dense_reference import nearest, reference_prediction

DEFAULT_DENSE_REFERENCE_BUNDLE = ROOT / 'artifacts/dense_reference_bundle_20261108'
DEFAULT_DATED_BUNDLE = ROOT / 'artifacts/dated_bundle_20261108'


def member_path(base, member):
    path = (Path(base) / member).resolve()
    if not path.is_relative_to(Path(base).resolve()):raise ValueError('Bundle/source member escapes its root')
    return path


class FrozenDenseReference:
    def __init__(self, bundle=DEFAULT_DENSE_REFERENCE_BUNDLE, primary_bundle=DEFAULT_DATED_BUNDLE):
        self.root = Path(bundle)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        for section, base in [('files', self.root), ('source_hashes', ROOT)]:
            for key, digest in self.manifest[section].items():
                if hashlib.sha256(member_path(base, key).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Dense reference bundle/source changed: {key}')
        primary = Path(primary_bundle)
        if hashlib.sha256((primary / 'manifest.json').read_bytes()).hexdigest() != self.manifest['primary_manifest_sha256']:
            raise ValueError('Primary bundle manifest differs from the frozen comparison')
        parent = json.loads((primary / 'manifest.json').read_text())
        for section, base in [('files', primary), ('source_hashes', ROOT)]:
            for key, digest in parent[section].items():
                if hashlib.sha256(member_path(base, key).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Primary bundle/source changed: {key}')
        self.pilot = FrozenRegimePilot(primary / 'regime')
        self.specs = {}; self.models = {}
        for name, case in self.manifest['cases'].items():
            self.specs[name] = {}
            for key, source in case['specs'].items():
                spec = source.copy()
                if 'weight_file' in spec:
                    path = str(member_path(self.root, spec['weight_file']))
                    spec['file'] = path
                    if path not in self.models:self.models[path] = drive.load(Path(path))
                self.specs[name][key] = spec

    def forecast(self, history, *, family, axis, angle_age_samples):
        return self.forecast_reference(history, family=family, axis=axis, angle_age_samples=angle_age_samples)

    def forecast_reference(self, history, *, family, axis, angle_age_samples):
        name = f'{family}_{axis}'
        if name not in self.manifest['cases']:raise ValueError('Unsupported family/axis')
        coordinates = ['x'] if family == 'trolley' else ['J'] if family == 'jib' else ['x', 'J']
        references = ['xr' if key == 'x' else 'Jr' for key in coordinates]
        if not set([axis, 'L'] + coordinates + references) <= set(history):
            raise ValueError('Required dated angle/current support/native-reference channels missing')
        raw = dated.validate(history, angle_age_samples, axis)
        age = int(angle_age_samples)
        known = nearest(self.manifest['train_ages_raw'], age)
        key = self.manifest['cases'][name]['reference'][str(known)]
        value = reference_prediction(self.specs[name][key], self.models, raw, age, axis, self.pilot, name)
        return dict(forecast_rad=value, reference_key=key, matched_train_age_raw=known,
            angle_age_samples=age, horizons_s=np.array(self.pilot.pilot.cfg['horizons_raw']) * .01,
            scope=self.manifest['scope'], primary_manifest_sha256=self.manifest['primary_manifest_sha256'],
            independent_confirmation=False)
