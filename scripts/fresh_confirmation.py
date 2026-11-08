"""Private, preregistered paired confirmation; no fitting or target-based selection."""
import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fresh_record_guard import (sha256_file, canonical_json_hash,
    metadata_preflight, fingerprint_sequence, potential_overlap)

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'configs/fresh_confirmation_20261108.json'
LOCKED=['configs/fresh_confirmation_20261108.json','configs/inteco_native_contract_20261108.json','data/.gitignore','docs/FRESH_CONFIRMATION_20261108.md',
        'scripts/fresh_record_guard.py','scripts/fresh_confirmation.py',
        'tests/test_fresh_record_guard.py','tests/test_fresh_confirmation.py','tests/test_fresh_score_e2e.py']

def read(path):return json.loads(Path(path).read_text())
def git_head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def committed(paths):
    for relative in paths:
        p=ROOT/relative
        try:value=subprocess.check_output(['git','show',f'HEAD:{relative}'],cwd=ROOT,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:raise ValueError('Commit protocol/source first: '+str(relative)) from e
        if value!=p.read_bytes():raise ValueError('Uncommitted protocol/source: '+str(relative))

def private_output(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if root.is_relative_to(ROOT):
        if subprocess.run(['git','check-ignore','-q',str(root.relative_to(ROOT))],cwd=ROOT).returncode:
            raise ValueError('Private intake/output directory must be Git-ignored')
    return root

def policy():return read(POLICY)
def fingerprint(values,*,candidate=False):
    p=policy();return fingerprint_sequence(np.asarray(values),rounding=p['rounding_rad'],
        block_size=p['overlap_block_samples'],seen_stride=p['seen_block_stride'],
        candidate=candidate,lock_verified=True,max_blocks=p['maximum_samples_per_record'])

def finite_segments(values):
    values=np.asarray(values);good=np.isfinite(values);cuts=np.flatnonzero(np.diff(np.r_[False,good,False]))
    return [(int(a),values[a:b]) for a,b in cuts.reshape(-1,2) if b-a>=policy()['overlap_block_samples']]

def build_seen_index(destination):
    """Only the already opened corpora; no candidate files are loaded here."""
    committed(LOCKED);destination=Path(destination);private_output(destination.parent)
    from scripts.run_external_benchmark import verify_hashes
    from scripts.verify_finetune_campaign import original_files
    if not original_files()['passed']:raise ValueError('Protected originals changed')
    prior=read(ROOT/'results/dated_forecast_20261108/frozen_development/development.json')
    verify_hashes(prior['guarded_hashes'])
    files={};auxiliary={};sequences={};sources={}
    def source(path):
        p=ROOT/path;sources[str(path)]=sha256_file(p);return p
    def register(digest,label,response=True):(files if response else auxiliary).setdefault(digest,[]).append(label)
    def sequence(label,values):
        for start,v in finite_segments(values):sequences[f'{label}/segment{start}']=fingerprint(v)
    m=read(source(Path('data/processed/manifest.json')))
    register(m['time_file']['sha256'],'Inteco33/time_xlsx',False)
    for record in m['scenarios']:register(record['sha256'],f'Inteco33/s{record["sid"]}/xlsx')
    data=source(Path('data/processed/dataset.npz'));register(sha256_file(data),'Inteco33/processed_npz')
    with np.load(data,allow_pickle=False) as z:
        for key in z.files:
            if key.endswith(('_theta','_beta')):sequence('Inteco33/'+key,z[key])
    archive=source(Path('data/raw/aalto_6907569/crane_dataset.zip'))
    if sha256_file(archive)!=read(ROOT/'results/latest/aalto_20261106_audit.json')['sha256']:raise ValueError('Known Aalto archive changed')
    register(sha256_file(archive),'Aalto6907569/archive')
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if not member.endswith('/'):register(hashlib.sha256(z.read(member)).hexdigest(),'Aalto6907569/member/'+member,member.lower().endswith('.csv'))
    obs=source(Path('data/processed/aalto_6907569/observations.npz'))
    if sha256_file(obs)!=read(ROOT/'data/processed/aalto_6907569/provenance.json')['output_sha256']:raise ValueError('Known Aalto reconstruction changed')
    register(sha256_file(obs),'Aalto6907569/causal_npz')
    with np.load(obs,allow_pickle=False) as z:
        for key in z.files:
            if key.endswith(('trolley_angle_rad','bridge_angle_rad')):sequence('Aalto6907569/'+key,z[key])
    dm=read(source(Path('data/raw/behavioral_crane_1aa26b5/download_manifest.json')))
    for f in dm['files']:register(f['sha256'],'Behavioral20/'+f['path'],'results_seq' in f['path'])
    from scripts.external_crane_data import config,load_sequence
    cfg=config();source(Path('configs/external_benchmark_20261107.json'));source(Path('scripts/external_crane_data.py'))
    ids=cfg['train_ids']+cfg['validation_ids']+cfg['test_ids']
    for sid in ids:
        s=load_sequence(cfg,sid,ids)
        for axis in ['theta','radial']:sequence(f'Behavioral20/s{sid}/{axis}',s[axis])
    value=dict(schema_version=1,policy_sha256=sha256_file(POLICY),source_hashes=sources,
        known_file_sha256=files,auxiliary_file_sha256=auxiliary,sequences=sequences,previously_opened_corpora=['Inteco33','Aalto6907569','Behavioral20'],
        file_reuse_rule='Only response-bearing assets block reuse; static clock/calibration/license/planned-profile assets are auxiliary, never physical-trial evidence.',
        includes_metadata_only_gantry_candidate=False,independent_trial_proven=False,
        warning='Nonmatching fingerprints do not prove independent physical trials. Rounded/resampled/noisy copies can evade this exact screen.')
    write_new(destination,value);return dict(index_sha256=sha256_file(destination),known_file_hashes=len(files),auxiliary_file_hashes=len(auxiliary),
        sequence_segments=len(sequences),informative_blocks=sum(len(x['blocks']) for x in sequences.values()),
        scope='old-seen registry only, no new physical records acquired',independent_confirmation=False)

def bundle_commitments():
    p=policy();out={}
    if read(ROOT/p['primary_bundle']/'manifest.json')['selected']!=p['primary_selected']:raise ValueError('Preregistered primary was replaced')
    for base in [ROOT/p['primary_bundle'],ROOT/p['reference_bundle']]:
        if not (base/'manifest.json').is_file():raise ValueError('Frozen primary/reference bundle is missing')
        for f in sorted(base.rglob('*')):
            if f.is_file():out[str(f.relative_to(ROOT))]=sha256_file(f)
            if f.name=='manifest.json':
                for section in ['source_hashes']:
                    for relative,digest in read(f).get(section,{}).items():
                        target=(ROOT/relative).resolve()
                        if not target.is_relative_to(ROOT) or sha256_file(target)!=digest:raise ValueError('Bundle source guard failed')
                        out[relative]=digest
    for f in LOCKED:out[f]=sha256_file(ROOT/f)
    contract=read(ROOT/p['training_contract'])
    if sha256_file(ROOT/contract['training_processed_manifest'])!=contract['training_processed_manifest_sha256']:raise ValueError('Original TRAIN export manifest changed')
    for f in p['training_evidence']:out[f]=sha256_file(ROOT/f)
    return out

def hierarchy(records):
    by_cluster={}
    for r in records:by_cluster.setdefault(r['independent_cluster_id'],set()).add(r['condition_id'])
    return 'conditions_then_clusters' if all(len(c)==1 for c in by_cluster.values()) else 'shared_clusters_fixed_conditions'

def lock(manifest_path,private_root,index_path,review_path,public_card):
    committed(LOCKED);p=policy();root=private_output(private_root);manifest=read(manifest_path);index=read(index_path)
    if index['policy_sha256']!=sha256_file(POLICY):raise ValueError('Seen index uses a different frozen screening policy')
    audit=metadata_preflight(manifest,root,seen_file_hashes=index['known_file_sha256'])
    audit_name='preflight_'+str(audit['canonical_manifest_sha256'])[:20]+'_'+canonical_json_hash(audit)[:12]+'.json'
    audit_path=root/'audits'/audit_name
    if not audit_path.exists():write_new(audit_path,audit)
    if not audit['metadata_declarations_complete']:return dict(status='blocked',metadata_audit=audit,final_goal_achieved=False)
    records=manifest['records'];conditions={r['condition_id'] for r in records};clusters={r['independent_cluster_id'] for r in records}
    if len(conditions)<p['minimum_conditions'] or len(clusters)<p['minimum_independent_clusters']:
        return dict(status='blocked',reason='Insufficient declared independent conditions/clusters for this fixed confirmation policy',final_goal_achieved=False)
    contract_path=ROOT/p['training_contract'];contract_sha=sha256_file(contract_path);contract=read(contract_path)
    for r in records:
        for ref in ['xr','Jr']:
            if ref not in required_channels(r):continue
            ch=r['channels'][ref];expected=contract['references'][ref]
            if ch.get('training_contract_sha256')!=contract_sha or ch['unit']!=expected['unit'] or ch['semantics']!=expected['semantics']:raise ValueError('Incoming reference is not bound to the frozen TRAIN export contract')
    review=Path(review_path).resolve()
    if not review.is_file():raise ValueError('Freeze a concrete source/units/provenance review first')
    review_value=read(review)
    if (review_value.get('metadata_sha256')!=audit['canonical_manifest_sha256'] or
        review_value.get('training_contract_sha256')!=contract_sha or
        review_value.get('source_id')!=manifest['source']['source_id'] or
        review_value.get('rig_id')!=manifest['source']['rig_id'] or
        not isinstance(review_value.get('evidence'),list) or not review_value['evidence'] or
        any(not isinstance(x,str) or not x.strip() for x in review_value['evidence'])):raise ValueError('Source review must bind this exact metadata, rig and TRAIN contract to concrete evidence')
    if review_value.get('normalization_kind') not in ['author_supplied_si','frozen_causal_adapter'] or not isinstance(review_value.get('normalization_evidence'),str) or not review_value['normalization_evidence'].strip():raise ValueError('Raw-to-SI source lineage and causal normalization evidence required')
    originals=review_value.get('original_source_files');adapters=review_value.get('adapter_hashes',{})
    if not isinstance(originals,dict) or not originals or not isinstance(adapters,dict):raise ValueError('Original source bytes and adapter hashes must be explicit')
    if review_value['normalization_kind']=='frozen_causal_adapter' and not adapters:raise ValueError('Freeze the actual conversion adapter before new scores')
    source_files={}
    for relative,digest in originals.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or sha256_file(path)!=digest or digest in index['known_file_sha256']:raise ValueError('Raw source lineage changed, escaped privacy root, or was previously opened')
        source_files[str(path)]=digest
    committed(list(adapters));hashes=bundle_commitments()
    for relative,digest in adapters.items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path)!=digest:raise ValueError('Normalization adapter changed')
        hashes[relative]=digest
    run=root/'runs'/audit['canonical_manifest_sha256'][:20]
    value=dict(schema_version=1,manifest=manifest,metadata_audit=audit,private_root=str(root),
        seen_index=str(Path(index_path).resolve()),seen_index_sha256=sha256_file(index_path),
        source_review_path=str(review),source_review_sha256=sha256_file(review),
        original_source_hashes=source_files,
        guarded_hashes=hashes,policy=p,bootstrap_scheme=hierarchy(records),
        prelock_git_head=git_head(),
        numerical_candidate_data_read=False,scores_computed=False,independent_trial_proven=False)
    lockfile=run/'lock.json';write_new(lockfile,value)
    card=Path(public_card).resolve()
    if not card.is_relative_to(ROOT/'docs') or card.is_relative_to(root):raise ValueError('Public commitment belongs in repo docs, not raw/private storage')
    write_new(card,dict(protocol='fresh_confirmation_20261108',private_lock_sha256=sha256_file(lockfile),
        policy=p,policy_sha256=sha256_file(POLICY),frozen_model_code_hashes=hashes,
        source_review_commitment=canonical_json_hash({'sha256':sha256_file(review),'lock':sha256_file(lockfile)}),
        records=len(records),conditions=len(conditions),independent_clusters=len(clusters),
        bootstrap_scheme=value['bootstrap_scheme'],raw_data_or_identifiers_published=False,
        physical_independence_not_certified_by_software=True,scored=False))
    return dict(status='locked_not_scored',private_lock=str(lockfile),public_card=str(card.relative_to(ROOT)),run_id=audit['canonical_manifest_sha256'][:20],
        next_step='Commit the public card before screen/score. No numeric candidate values have been parsed.',final_goal_achieved=False)

def verify_lock(lockfile,cardfile):
    locked_bytes=Path(lockfile).read_bytes();v=json.loads(locked_bytes);lock_sha=hashlib.sha256(locked_bytes).hexdigest();cardfile=Path(cardfile).resolve()
    if not cardfile.is_relative_to(ROOT):raise ValueError('Public card must be committed in this repository')
    committed([str(cardfile.relative_to(ROOT))]);card=read(cardfile)
    if card['private_lock_sha256']!=lock_sha:raise ValueError('Public/private lock commitment differs')
    if card['policy']!=v['policy'] or card['frozen_model_code_hashes']!=v['guarded_hashes'] or card['bootstrap_scheme']!=v['bootstrap_scheme']:raise ValueError('Public card misstates the private locked protocol')
    if sha256_file(v['seen_index'])!=v['seen_index_sha256'] or sha256_file(v['source_review_path'])!=v['source_review_sha256']:raise ValueError('Intake source/index review changed')
    for path,digest in v['original_source_hashes'].items():
        if sha256_file(path)!=digest:raise ValueError('Original source lineage bytes changed')
    for rel,digest in v['guarded_hashes'].items():
        if sha256_file(ROOT/rel)!=digest:raise ValueError('Frozen source/model/policy changed: '+rel)
    audit=metadata_preflight(v['manifest'],v['private_root'],seen_file_hashes=read(v['seen_index'])['known_file_sha256'])
    if not audit['metadata_declarations_complete'] or audit['canonical_manifest_sha256']!=v['metadata_audit']['canonical_manifest_sha256']:raise ValueError('Private raw/declarations changed')
    v['_verified_lock_sha256']=lock_sha
    return v

def required_channels(record):
    coords={'trolley':['x'],'jib':['J'],'simult':['x','J']}[record['family']]
    return record['axes']+['L']+coords+['xr' if c=='x' else 'Jr' for c in coords]

def load_record(record,root,p):
    """Called only after verify_lock. No interpolation, centering or target selection."""
    path=Path(root)/record['file']
    if path.suffix.lower()!='.npz':raise ValueError('A frozen metadata-defined adapter must first provide normalized NPZ')
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=record['sha256']:raise ValueError('Parsed source bytes differ from the immutable lock')
    def real(z,key):
        value=z[key]
        if value.dtype.kind not in 'iuf':raise ValueError('Only real numeric acquisition channels permitted')
        return np.asarray(value,float)
    with np.load(io.BytesIO(content),allow_pickle=False) as z:
        if record['acquisition']['time_column'] in [ch['column'] for ch in record['channels'].values()]:raise ValueError('Time cannot alias a physical channel')
        t=real(z,record['acquisition']['time_column'])
        arrays={k:real(z,record['channels'][k]['column']) for k in required_channels(record)}
        if t.ndim!=1 or len(t)>p['maximum_samples_per_record'] or len(t)<=p['first_origin_raw']+max(p['horizons_raw']):raise ValueError('Insufficient or oversized actual time grid')
        if not np.isfinite(t).all() or not np.all(np.abs(np.diff(t)-.01)<=p['clock_grid_tolerance_s']):raise ValueError('Actual acquisition grid is not the fixed100Hz contract')
        if any(a.shape!=t.shape or not np.isfinite(a).all() for a in arrays.values()) or (arrays['L']<=0).any():raise ValueError('Invalid/missing aligned physical signals; no imputation')
        if record['clock_scope']=='known_arrival_age':
            if record['arrival']['clock_id']!=record['acquisition']['clock_id']:raise ValueError('Different actual clocks need a separately frozen conversion adapter')
            arrival=record['arrival'];axis_columns=arrival.get('angle_time_columns',{axis:arrival['time_column'] for axis in record['axes']})
            for column in axis_columns.values():
                available=real(z,column)
                if available.shape!=t.shape or not np.isfinite(available).all() or not np.all(np.abs(available-t-record['angle_age_samples']*.01)<=p['clock_grid_tolerance_s']):raise ValueError('Actual arrival timestamps disagree with fixed declared age')
            for channel,column in arrival['nonangle_time_columns'].items():
                available=real(z,column)
                if available.shape!=t.shape or not np.isfinite(available).all() or not np.all(np.abs(available-t)<=p['clock_grid_tolerance_s']):raise ValueError('Actual nonangle availability violates the current-clock input contract')
    k=np.arange(p['first_origin_raw'],len(t)-max(p['horizons_raw']),p['origin_stride_raw'],dtype=int)
    if not len(k) or (k[0]-400-record['angle_age_samples'])<0:raise ValueError('Genuine dated prehistory is missing')
    return t,arrays,k

def screen_records(v,lock_sha):
    index=read(v['seen_index']);report=dict(status='blocked',records=[],
        private_lock_sha256=lock_sha,candidate_scores_computed=False,
        nonmatch_is_not_independence=True,source_provenance_requires_evidence_audit=True)
    for record in v['manifest']['records']:
        item=dict(record_id=record['record_id'],passed=False)
        try:
            t,arrays,k=load_record(record,v['private_root'],v['policy']);hits=[]
            for axis in record['axes']:
                overlap=potential_overlap(fingerprint(arrays[axis],candidate=True),index['sequences'])
                if overlap['status']=='potential_overlap_found':hits.append(dict(axis=axis,result=overlap))
            item.update(passed=not hits,potential_overlaps=hits,origins=len(k),samples=len(t),
                        axes=record['axes'],origin_index_sha256=hashlib.sha256(k.astype('<i8').tobytes()).hexdigest())
        except Exception as error:item['failure']=f'{type(error).__name__}: {error}'
        report['records'].append(item)
    report['status']='schema_screened_no_detected_overlap' if all(r['passed'] for r in report['records']) else 'blocked'
    return report

def screen(lockfile,cardfile):
    v=verify_lock(lockfile,cardfile);report=screen_records(v,v['_verified_lock_sha256'])
    write_new(Path(lockfile).parent/'screen.json',report);return report

def paired_statistics(rows,p,scheme):
    if not rows:raise ValueError('No physical trial summaries')
    if len({(r['session_id'],r['trial_id']) for r in rows})!=len(rows):raise ValueError('Duplicated physical-trial summaries')
    conditions=sorted({r['condition_id'] for r in rows});clusters=sorted({r['independent_cluster_id'] for r in rows})
    y=np.array([[r['candidate_mae_deg'],r['reference_mae_deg']] for r in rows],float)
    if not np.isfinite(y).all() or (y<0).any():raise ValueError('Finite nonnegative paired trial errors required')
    cc=np.array([conditions.index(r['condition_id']) for r in rows]);gg=np.array([clusters.index(r['independent_cluster_id']) for r in rows]);per=np.array([y[cc==i].mean(0) for i in range(len(conditions))]);point=per.mean(0);rng=np.random.default_rng(p['bootstrap_seed']);draws=[];invalid=0
    for _ in range(p['bootstrap_draws']):
        means=[]
        if scheme=='conditions_then_clusters':
            if any(len(set(cc[gg==g]))>1 for g in range(len(clusters))):raise ValueError('Shared clusters invalidate disjoint-condition bootstrap')
            for c in rng.integers(0,len(conditions),len(conditions)):
                ids=np.flatnonzero(cc==c);groups=np.unique(gg[ids]);chosen=rng.choice(groups,len(groups),replace=True);w=np.array([(chosen==g).sum() for g in gg[ids]]);means.append(np.average(y[ids],axis=0,weights=w))
        elif scheme=='shared_clusters_fixed_conditions':
            chosen=rng.integers(0,len(clusters),len(clusters));w=np.bincount(chosen,minlength=len(clusters))[gg]
            for c in range(len(conditions)):
                ids=np.flatnonzero(cc==c)
                if not w[ids].sum():break
                means.append(np.average(y[ids],axis=0,weights=w[ids]))
        else:raise ValueError('Undeclared bootstrap hierarchy')
        if len(means)!=len(conditions):draws.append(np.inf);invalid+=1
        else:draws.append(float(np.mean(means,axis=0)@[1.,-1.]))
    upper=float(np.quantile(draws,p['bootstrap_upper_quantile'],method='higher'));gain=float(1-point[0]/point[1]) if point[1]>0 else None;wins=int((per[:,0]<per[:,1]).sum());enough=len(conditions)>=p['minimum_conditions'] and len(clusters)>=p['minimum_independent_clusters'];passed=enough and gain is not None and gain>=p['minimum_relative_gain'] and wins/len(conditions)>p['condition_win_fraction_exclusive'] and np.isfinite(upper) and upper<0
    return dict(candidate_mae_deg=float(point[0]),reference_mae_deg=float(point[1]),relative_gain=gain,
        condition_wins=wins,conditions=len(conditions),independent_clusters=len(clusters),physical_trials=len(rows),
        bootstrap_scheme=scheme,bootstrap_draws=p['bootstrap_draws'],missing_condition_draws=invalid,
        upper95_delta_deg=upper if np.isfinite(upper) else None,statistical_gate_passed=bool(passed),
        physical_independence_not_certified_by_software=True,final_goal_achieved=False)

def score(lockfile,cardfile):
    v=verify_lock(lockfile,cardfile);folder=Path(lockfile).parent;screening=read(folder/'screen.json')
    if screening['status']!='schema_screened_no_detected_overlap' or screening['private_lock_sha256']!=v['_verified_lock_sha256']:raise ValueError('Locked full schema/overlap screen must pass first')
    if canonical_json_hash(screening)!=canonical_json_hash(screen_records(v,v['_verified_lock_sha256'])):raise ValueError('Saved numeric screen differs from current locked replay')
    score_folder=folder/'scoring';score_folder.mkdir()  # One attempt only, including failed attempts.
    # Imports/construction occur after all immutable protocol/data checks.
    from cranepilot.dated_pilot import FrozenDatedPilot
    from cranepilot.matched_dated_pilot import FrozenDenseReference
    p=v['policy'];candidate=FrozenDatedPilot(ROOT/p['primary_bundle']);reference=FrozenDenseReference(ROOT/p['reference_bundle'],primary_bundle=ROOT/p['primary_bundle']);rows=[];failures=[]
    for record in v['manifest']['records']:
        try:
            t,a,k=load_record(record,v['private_root'],p);age=record['angle_age_samples'];history=k[:,None]-np.arange(401);predictions={};errors=[]
            for axis in record['axes']:
                raw={key:values[history].copy() for key,values in a.items() if key not in ['theta','beta']};raw[axis]=a[axis][history-age].copy();target=a[axis][k[:,None]+np.array(p['horizons_raw'])]
                input_one={key:value.copy() for key,value in raw.items()};input_two={key:value.copy() for key,value in raw.items()}
                output_one=candidate.forecast(input_one,family=record['family'],axis=axis,angle_age_samples=age);output_two=reference.forecast(input_two,family=record['family'],axis=axis,angle_age_samples=age)
                expected_horizons=np.array(p['horizons_raw'])*.01
                for result in [output_one,output_two]:
                    if np.asarray(result['horizons_s']).shape!=expected_horizons.shape or not np.allclose(result['horizons_s'],expected_horizons,rtol=0,atol=1e-12):raise ValueError('Predictor horizons differ from the locked paired target')
                one=output_one['forecast_rad'];two=output_two['forecast_rad']
                if one.shape!=target.shape or two.shape!=target.shape or not np.isfinite(one).all() or not np.isfinite(two).all():raise ValueError('Forecast shape/nonfinite failure')
                errors.append([float(np.rad2deg(np.abs(one-target)).mean()),float(np.rad2deg(np.abs(two-target)).mean())]);predictions.update({axis+'_candidate':one,axis+'_reference':two,axis+'_target':target})
            # Private output contains targets; never include this folder in public packaging.
            filename=canonical_json_hash({'record_id':record['record_id']})+'.npz'
            actual_horizons=t[k[:,None]+np.array(p['horizons_raw'])]-t[k,None]
            with (score_folder/filename).open('xb') as f:np.savez_compressed(f,k=k,actual_horizons_s=actual_horizons,**predictions)
            means=np.mean(errors,axis=0);rows.append({key:record[key] for key in ['record_id','condition_id','session_id','trial_id','independent_cluster_id']}|dict(candidate_mae_deg=float(means[0]),reference_mae_deg=float(means[1]),origins=len(k),axes=len(record['axes']),clock_scope=record['clock_scope'],maximum_target_time_deviation_s=float(np.abs(actual_horizons-np.array(p['horizons_raw'])*.01).max())))
        except Exception as error:failures.append(dict(record_id=record['record_id'],failure=f'{type(error).__name__}: {error}'))
    report=dict(status='complete' if not failures else 'failed_coverage',private_lock_sha256=v['_verified_lock_sha256'],trials=rows,failures=failures,coverage_complete=not failures,raw_or_target_publication=False,physical_independence_not_certified_by_software=True,hardware_success_claimed=False,final_goal_achieved=False)
    if not failures:report['statistics']=paired_statistics(rows,p,v['bootstrap_scheme'])
    write_new(score_folder/'score.json',report);return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['registry','lock','screen','score']);parser.add_argument('--private-root',default='data/private_confirmation_20261108');parser.add_argument('--index');parser.add_argument('--manifest');parser.add_argument('--review');parser.add_argument('--public-card');parser.add_argument('--lock');args=parser.parse_args()
    code=0
    try:
        if args.command=='registry':result=build_seen_index(args.index or Path(args.private_root)/'known_seen.json')
        elif args.command=='lock':result=lock(args.manifest,args.private_root,args.index,args.review,args.public_card)
        elif args.command=='screen':result=screen(args.lock,args.public_card)
        else:result=score(args.lock,args.public_card)
    except Exception as error:
        code=1;failure=dict(command=args.command,error_type=type(error).__name__,private_error=str(error),final_goal_achieved=False)
        try:
            folder=private_output(Path(args.lock).resolve().parent if args.lock else args.private_root)
            dest=folder/'failures'/(canonical_json_hash(failure)+'.json')
            if not dest.exists():write_new(dest,failure)
        except Exception:pass
        result=dict(status='blocked',reason=type(error).__name__+': consult private audit/failure report',final_goal_achieved=False)
    # Bounded status only; full raw/provenance/targets remain in the private output.
    allowed=['status','reason','run_id','index_sha256','known_file_hashes','sequence_segments','informative_blocks','scope','next_step','coverage_complete','candidate_scores_computed','nonmatch_is_not_independence','final_goal_achieved','independent_confirmation']
    print(json.dumps({k:result[k] for k in allowed if k in result},indent=2,allow_nan=False))
    raise SystemExit(code or (2 if result.get('status') in ['blocked','failed_coverage'] else 0))
