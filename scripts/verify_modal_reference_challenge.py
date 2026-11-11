"""Replay 240 validation fits and all retrospective reference predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import load_allowed,packets,concatenate,score
from scripts.modal_reference_challenge import predict,load
from scripts.run_modal_reference_challenge import RUN,parents,prediction
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

OUT=Path('results/modal_reference_challenge_20261107')
def main():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain prior verification')
    c,cfg,parent,conf=parents();p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());selected=json.loads((RUN/'selection.json').read_text());rows=pd.read_csv(RUN/'candidate_validation.csv');metrics=pd.read_csv(RUN/'retrospective/metrics.csv')
    r=dict(passed=False,checks=[],development_sha256=sha256(RUN/'development.json'),retrospective_provenance_sha256=sha256(RUN/'retrospective/provenance.json'),verifier_sha256=sha256(Path(__file__)),independent_confirmation=False)
    try:
        assert p['status']==q['status']=='complete';verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');assert q['development_sha256']==sha256(RUN/'development.json') and q['selection_sha256']==sha256(RUN/'selection.json');r['original_files']=original_files();assert r['original_files']['passed']
        for family,fold in cfg['families'].items():
            val=load_allowed(cfg,fold['validation'],'validation');records=load_confirmation(cfg,fold['confirmation'])
            for axis in fold['axes']:
                name=f'{family}_{axis}';case=selected[name];cal=LengthCalibration(**json.loads((parent/name/'calibration.json').read_text()));va=concatenate([packets(s,axis,cfg,cal)[0] for s in val.values()]);frame=rows[rows['case']==name];assert len(frame)==48
                for _,row in frame.iterrows():np.testing.assert_allclose(score(predict(load(RUN/row.file),va,cfg),va),row.validation_mae_deg,rtol=1e-12,atol=1e-12)
                with np.load(RUN/name/'validation.npz') as f:
                    y=prediction(case,va,cfg,RUN/name,parent/name);np.testing.assert_array_equal(y,f['challenged_arx']);np.testing.assert_allclose(score(y,va),min(frame.validation_mae_deg.min(),score(f['original_arx'],va)),rtol=1e-12)
                for sid,seq in records.items():
                    part,_=packets(seq,axis,cfg,cal);y=prediction(case,part,cfg,RUN/name,parent/name)
                    with np.load(RUN/'retrospective'/f'{name}_s{sid}.npz') as f:np.testing.assert_array_equal(y,f['challenged_arx']);np.testing.assert_array_equal(part['k'],f['k'])
                    for j,h in enumerate(cfg['horizons_raw']):
                        row=metrics[(metrics['case']==name)&(metrics.scenario==sid)&(metrics.method=='challenged_arx')&(np.abs(metrics.horizon_s-h*.01)<1e-10)].iloc[0];np.testing.assert_allclose(np.abs(np.rad2deg(y[:,j]-part['target'][:,j])).mean(),row.mae_deg,rtol=1e-12,atol=1e-12)
                    small={k:v[:2].copy() for k,v in part.items()};a=prediction(case,small,cfg,RUN/name,parent/name);small['target'][:]=999.;np.testing.assert_array_equal(a,prediction(case,small,cfg,RUN/name,parent/name))
                    r['checks'].append(dict(case=name,scenario=sid,packets=len(part['k']),replay_error=0.,targets_ignored=True))
                print('VERIFIED',name,flush=True)
        old=pd.read_csv(conf/'metrics.csv');copy=metrics[metrics.method!='challenged_arx'];merged=copy.merge(old,on=['case','scenario','method','horizon_s'],suffixes=('_new','_old'),validate='one_to_one');assert len(merged)==len(copy);np.testing.assert_array_equal(merged.mae_deg_new,merged.mae_deg_old)
        r.update(passed=True,validation_fits=240,retrospective_forecasts=sum(c['packets'] for c in r['checks']),old_modal_and_references_unchanged=True,validation_only_selection=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
