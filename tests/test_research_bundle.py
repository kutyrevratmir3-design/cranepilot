import builtins
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts.predict_research_bundle import (ResearchBundle,DEFAULT_BUNDLE,FOLDS,PROFILES,
                                            bundle_file,file_sha256)


def packet():
    time=-.01*np.arange(41)
    theta=np.stack([.02*np.cos(5*time),.04*np.sin(4*time)])
    x=np.stack([.15+.01*np.sin(2*time),.2+.015*np.cos(2*time)])
    return dict(theta=theta,x=x,reference=np.full((2,41),.215),
                length_m=np.array([.3,.8]),future_reference=np.full((2,30),.215))


class ResearchBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if not (DEFAULT_BUNDLE/'manifest.json').is_file():
            raise FileNotFoundError('Tracked research bundle is missing; run export_research_bundle --verify locally')

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.copy=Path(self.temporary.name)/'moved_bundle'
        shutil.copytree(DEFAULT_BUNDLE,self.copy)

    def test_actual_bundle_all_profiles_run_when_nonbundle_data_reads_are_forbidden(self):
        original_open=builtins.open;original_io_open=io.open
        def checked(function):
            def call(file,*args,**kwargs):
                if not isinstance(file,int):
                    path=Path(file).resolve()
                    if not path.is_relative_to(self.copy.resolve()) and path.suffix not in ('.py','.pyc'):
                        raise AssertionError(f'Inference tried to read outside moved bundle/code: {path}')
                return function(file,*args,**kwargs)
            return call
        with mock.patch('builtins.open',checked(original_open)),mock.patch('io.open',checked(original_io_open)):
            for fold in FOLDS:
                for profile in PROFILES:
                    model=ResearchBundle(self.copy,fold=fold,profile=profile)
                    result=model.forecast(packet())
                    self.assertEqual(result['prediction_rad'].shape,(2,30))
                    self.assertTrue(np.isfinite(result['prediction_rad']).all())
                    self.assertFalse(result['metadata']['candidate_promoted'])
                    if fold=='short_transfer':self.assertIsNone(result['seed_std_rad'])
                    else:self.assertEqual(result['seed_std_rad'].shape,(2,30))

    def test_default_retained_and_explicit_candidate_warning(self):
        retained=ResearchBundle(self.copy,fold='primary')
        candidate=ResearchBundle(self.copy,fold='primary',profile='candidate')
        self.assertEqual(retained.profile,'retained')
        self.assertEqual(retained.spec['method'],'phase_space')
        self.assertEqual(candidate.spec['method'],'adapter_direct8')
        result=candidate.forecast(packet())
        self.assertFalse(result['metadata']['gate_warning']['passed'])
        self.assertGreater(result['metadata']['gate_warning']['ratio'],1.01)
        with self.assertRaises(TypeError):ResearchBundle(self.copy)
        with self.assertRaises(ValueError):ResearchBundle(self.copy,fold='automatic')

    def test_weight_corruption_is_detected_before_deserialization(self):
        manifest=json.loads((self.copy/'manifest.json').read_text())
        path=self.copy/manifest['folds']['primary']['actor']
        path.write_bytes(path.read_bytes()+b'corrupted')
        with mock.patch('torch.load',side_effect=AssertionError('Corrupt weights must not be loaded')):
            with self.assertRaisesRegex(ValueError,'Bundled file changed'):
                ResearchBundle(self.copy,fold='primary')

    def test_targets_are_not_read_and_history_contract_remains_enforced(self):
        model=ResearchBundle(self.copy,fold='short_transfer')
        clean=packet();poisoned=copy.deepcopy(clean)
        poisoned['target_q']=np.full((2,30),np.nan);poisoned['target_x']=np.full((2,30),np.inf)
        np.testing.assert_array_equal(model.forecast(clean)['prediction_rad'],model.forecast(poisoned)['prediction_rad'])
        clean['future_reference'][0,0]+=.1
        with self.assertRaisesRegex(ValueError,'origin command'):model.forecast(clean)

    def test_source_checksums_allow_crlf_but_detect_content_changes_and_path_escape(self):
        first=Path(self.temporary.name)/'first.py';second=Path(self.temporary.name)/'second.py'
        first.write_bytes(b'a = 1\nb = 2\n');second.write_bytes(b'a = 1\r\nb = 2\r\n')
        self.assertEqual(file_sha256(first,canonical_source=True),file_sha256(second,canonical_source=True))
        second.write_bytes(b'a = 3\r\nb = 2\r\n')
        self.assertNotEqual(file_sha256(first,canonical_source=True),file_sha256(second,canonical_source=True))
        with self.assertRaises(ValueError):bundle_file(self.copy,'../outside.pt')


if __name__=='__main__':unittest.main()
