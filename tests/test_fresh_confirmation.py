"""Synthetic software fixtures only; these are never physical confirmation."""
import copy,json,tempfile,unittest,subprocess,sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scripts import fresh_confirmation as fc
from scripts.fresh_record_guard import sha256_file,canonical_json_hash

class FreshConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name).resolve();self.root=self.base/'private';self.root.mkdir();self.repo=self.base/'repo';(self.repo/'docs').mkdir(parents=True);self.p=fc.policy();self.p.update(bootstrap_draws=128,minimum_conditions=1,minimum_independent_clusters=1)
        self.policy=self.repo/'policy.json';self.policy.write_text(json.dumps(self.p));self.index=self.root/'seen.json';self.index.write_text(json.dumps(dict(policy_sha256=sha256_file(self.policy),known_file_sha256={},sequences={})));self.review=self.root/'review.json';self.card=self.repo/'docs'/'card.json';self.make_file()
        self.contract=self.repo/self.p['training_contract'];self.contract.parent.mkdir(parents=True);self.contract.write_text(json.dumps(dict(references={key:dict(unit='native',semantics='synthetic_fixture') for key in ['xr','Jr']})))
        self.r=dict(record_id='fixture',session_id='s',trial_id='t',condition_id='c',independent_cluster_id='g',file='trial.npz',sha256=sha256_file(self.root/'trial.npz'),family='trolley',axes=['theta'],sample_rate_hz=100,clock_scope='synchronous_acquisition_offline',angle_age_samples=0,history_s=4.,target_tail_s=1.5,genuine_history=True,encoder_quantum_rad=.001,filters=dict(kind='none',description='fixture',evidence='synthetic fixture'),acquisition=dict(time_column='t',clock_id='daq',unit='s',evidence='synthetic fixture',synchronized=True,synchronization_evidence='synthetic common grid'),arrival=None,channels=dict(theta=dict(column='theta',unit='rad'),L=dict(column='L',unit='m'),x=dict(column='x',unit='m'),xr=dict(column='xr',unit='native',training_unit='native',semantics='synthetic_fixture',training_semantics='synthetic_fixture',compatibility_evidence='fixture only, not an Inteco semantic assertion')))
        self.m=dict(schema_version=1,source=dict(source_id='fixture',rig_id='fixture',physical_recordings=True,provenance_evidence='SYNTHETIC DECLARATION TEST ONLY',research_use_permitted=True,research_use_evidence='software fixture'),records=[self.r]);self.manifest=self.root/'manifest.json';self.manifest.write_text(json.dumps(self.m))
    def make_file(self,**changes):
        data=dict(t=np.arange(800)*.01,theta=np.random.default_rng(7).normal(size=800)*.01,L=np.full(800,.4),x=np.zeros(800),xr=np.zeros(800));data.update(changes);np.savez_compressed(self.root/'trial.npz',**data)
        if hasattr(self,'r'):self.r['sha256']=sha256_file(self.root/'trial.npz')
    def patches(self):
        return patch.multiple(fc,ROOT=self.repo,POLICY=self.policy)
    def create_lock(self):
        for r in self.m['records']:
            for ref in ['xr','Jr']:
                if ref in r['channels']:r['channels'][ref]['training_contract_sha256']=sha256_file(self.contract)
        self.manifest.write_text(json.dumps(self.m))
        self.review.write_text(json.dumps(dict(source_id=self.m['source']['source_id'],rig_id=self.m['source']['rig_id'],metadata_sha256=canonical_json_hash(self.m),training_contract_sha256=sha256_file(self.contract),evidence=['SYNTHETIC SOFTWARE TEST FIXTURE ONLY'],normalization_kind='author_supplied_si',normalization_evidence='Synthetic normalized arrays, not a physical source',original_source_files={r['file']:r['sha256'] for r in self.m['records']},adapter_hashes={})));self.review_original=self.review.read_text()
        hashes={'policy.json':sha256_file(self.policy),self.p['training_contract']:sha256_file(self.contract)}
        with self.patches(),patch.object(fc,'committed'),patch.object(fc,'git_head',return_value='synthetic-test-head'),patch.object(fc,'policy',return_value=self.p),patch.object(fc,'bundle_commitments',return_value=hashes):
            result=fc.lock(self.manifest,self.root,self.index,self.review,self.card)
        self.assertEqual(result['status'],'locked_not_scored');return Path(result['private_lock'])
    def test_lock_hashes_bytes_without_loading_numerical_candidate(self):
        with patch('numpy.load',side_effect=AssertionError('No numerical parse before lock')):lock=self.create_lock()
        value=fc.read(lock);self.assertFalse(value['scores_computed']);self.assertFalse(value['numerical_candidate_data_read']);card=fc.read(self.card);self.assertNotIn('fixture',json.dumps(card));self.assertFalse(card['raw_data_or_identifiers_published'])
    def test_raw_or_review_changes_block_before_numeric_loading(self):
        lock=self.create_lock();self.review.write_text('changed')
        with self.patches(),patch.object(fc,'committed'),patch('numpy.load',side_effect=AssertionError('No numerical parse')):
            with self.assertRaisesRegex(ValueError,'review changed'):fc.verify_lock(lock,self.card)
        self.review.write_text(self.review_original);self.make_file(theta=np.ones(800))
        with self.patches(),patch.object(fc,'committed'),patch('numpy.load',side_effect=AssertionError('No numerical parse')):
            with self.assertRaisesRegex(ValueError,'lineage bytes changed'):fc.verify_lock(lock,self.card)
    def test_public_card_must_be_committed_before_screen(self):
        lock=self.create_lock()
        with self.patches(),patch.object(fc,'committed',side_effect=ValueError('uncommitted')),patch('numpy.load',side_effect=AssertionError('No numerical parse')):
            with self.assertRaisesRegex(ValueError,'uncommitted'):fc.screen(lock,self.card)
    def test_misleading_public_policy_is_rejected(self):
        lock=self.create_lock();card=fc.read(self.card);card['policy']['minimum_relative_gain']=0;self.card.write_text(json.dumps(card))
        with self.patches(),patch.object(fc,'committed'):
            with self.assertRaisesRegex(ValueError,'misstates'):fc.verify_lock(lock,self.card)
    def test_declared_npz_schema_checks_actual_grid_and_missing_inputs(self):
        t,a,k=fc.load_record(self.r,self.root,self.p);np.testing.assert_array_equal(k,[425,525,625]);self.assertEqual(set(a),{'theta','L','x','xr'})
        self.make_file(t=np.arange(800)*.02)
        with self.assertRaisesRegex(ValueError,'100Hz'):fc.load_record(self.r,self.root,self.p)
        self.make_file(L=np.zeros(800))
        with self.assertRaisesRegex(ValueError,'physical signals'):fc.load_record(self.r,self.root,self.p)
    def test_actual_known_age_not_derived_from_prediction_errors(self):
        r=copy.deepcopy(self.r);r.update(clock_scope='known_arrival_age',angle_age_samples=15,arrival=dict(time_column='arrival',clock_id='daq',nonangle_time_columns={key:'t' for key in ['L','x','xr']}));self.make_file(arrival=np.arange(800)*.01+.15);r['sha256']=self.r['sha256'];fc.load_record(r,self.root,self.p);self.make_file(arrival=np.arange(800)*.01+.10);r['sha256']=self.r['sha256']
        with self.assertRaisesRegex(ValueError,'arrival timestamps'):fc.load_record(r,self.root,self.p)
    def test_numeric_overlap_quarantines_whole_trial_without_forecasts(self):
        with np.load(self.root/'trial.npz') as z:known=fc.fingerprint(z['theta'])
        index=fc.read(self.index);index['sequences']={'previous':known};self.index.write_text(json.dumps(index));lock=self.create_lock()
        with self.patches(),patch.object(fc,'committed'):
            result=fc.screen(lock,self.card)
        self.assertEqual(result['status'],'blocked');self.assertTrue(result['records'][0]['potential_overlaps']);self.assertFalse(result['candidate_scores_computed'])
    def test_complete_locked_schema_screen_still_does_not_certify_independence(self):
        lock=self.create_lock()
        with self.patches(),patch.object(fc,'committed'):r=fc.screen(lock,self.card)
        self.assertEqual(r['status'],'schema_screened_no_detected_overlap');self.assertTrue(r['nonmatch_is_not_independence'])
    def rows(self):
        return [dict(session_id=f's{i}',trial_id=f't{i}',condition_id=f'c{i}',independent_cluster_id=f'g{i}',candidate_mae_deg=1.,reference_mae_deg=2.) for i in range(5)]
    def test_bootstrap_clusters_and_conditions_not_window_count(self):
        r=self.rows();p=dict(self.p,minimum_conditions=5,minimum_independent_clusters=5);one=fc.paired_statistics(r,p,'conditions_then_clusters');two=fc.paired_statistics([dict(x,origins=10000000) for x in r],p,'conditions_then_clusters');self.assertEqual(one,two);self.assertTrue(one['statistical_gate_passed']);self.assertEqual(one['upper95_delta_deg'],-1.);self.assertFalse(one['final_goal_achieved'])
    def test_losses_zero_reference_and_duplicate_trials_cannot_pass(self):
        r=self.rows();r[0]['candidate_mae_deg']=20;self.assertFalse(fc.paired_statistics(r,self.p,'conditions_then_clusters')['statistical_gate_passed'])
        r=self.rows();r[0]['reference_mae_deg']=0;result=fc.paired_statistics(r,self.p,'conditions_then_clusters');self.assertGreater(result['upper95_delta_deg'],-1)
        with self.assertRaisesRegex(ValueError,'Duplicated'):fc.paired_statistics(self.rows()*2,self.p,'conditions_then_clusters')
    def test_shared_cluster_scheme_and_missing_strata_remain_adverse(self):
        r=self.rows();r.append(dict(r[0],trial_id='other',condition_id='c1'));self.assertEqual(fc.hierarchy(r),'shared_clusters_fixed_conditions')
        with self.assertRaisesRegex(ValueError,'Shared clusters'):fc.paired_statistics(r,self.p,'conditions_then_clusters')
        result=fc.paired_statistics(r,self.p,'shared_clusters_fixed_conditions');self.assertGreater(result['missing_condition_draws'],0);self.assertIsNone(result['upper95_delta_deg']);self.assertFalse(result['statistical_gate_passed'])
    def test_finite_segments_only_index_old_values_and_never_interpolate(self):
        x=np.r_[np.arange(300.),np.nan,np.arange(300.)];segments=fc.finite_segments(x);self.assertEqual([a for a,_ in segments],[0,301]);self.assertEqual([len(v) for _,v in segments],[300,300]);self.assertTrue(np.isnan(x[300]))
    def test_actual_bytes_are_bound_even_if_file_changes_after_preflight(self):
        r=copy.deepcopy(self.r);self.make_file(theta=np.ones(800))
        with self.assertRaisesRegex(ValueError,'Parsed source bytes'):fc.load_record(r,self.root,self.p)
    def test_complex_and_time_channel_aliases_are_rejected_without_conversion(self):
        self.make_file(theta=np.ones(800,dtype=complex)*(1+2j))
        with self.assertRaisesRegex(ValueError,'real numeric'):fc.load_record(self.r,self.root,self.p)
        self.make_file();self.r['acquisition']['time_column']='theta'
        with self.assertRaisesRegex(ValueError,'alias'):fc.load_record(self.r,self.root,self.p)
    def test_equal_conditions_not_pooled_trial_counts(self):
        rows=[dict(session_id='s0',trial_id='t0',condition_id='a',independent_cluster_id='g0',candidate_mae_deg=0.,reference_mae_deg=10.)]+[dict(session_id=f's{i}',trial_id=f't{i}',condition_id='b',independent_cluster_id=f'g{i}',candidate_mae_deg=9.,reference_mae_deg=10.) for i in range(1,10)];r=fc.paired_statistics(rows,self.p,'conditions_then_clusters');self.assertEqual(r['candidate_mae_deg'],4.5);self.assertEqual(r['reference_mae_deg'],10.)
    def test_cli_error_does_not_disclose_private_paths_or_record_identifiers(self):
        secret=self.root/'private-secret-owner-record.json';result=subprocess.run([sys.executable,'-m','scripts.fresh_confirmation','lock','--manifest',str(secret),'--private-root',str(self.root)],cwd=fc.ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.assertNotEqual(result.returncode,0);self.assertEqual(json.loads(result.stdout)['status'],'blocked');self.assertNotIn(str(self.root),result.stdout+result.stderr);self.assertNotIn(secret.name,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
