"""Replay complete raw-to-gated forecasts; preserve retrospective qualification."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import load_allowed,packets,modal,score
from scripts.modal_reference_challenge import load
from scripts.modal_backtest_gate import backtest,compose
from scripts.run_modal_backtest_gate import RUN,parents,references
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import verify_hashes
from scripts.verify_finetune_campaign import original_files

OUT=Path('results/modal_backtest_gate_20261107')
def replay(part,name,case,cfg,g,cal,base,reference,prior,short):
    long,short_y=references(name,part,cfg,cal,base,reference,prior,short);yr=short_y if case['reference']=='short' else long;ym=modal(part,cfg,'two_mode')[0];evidence=backtest(part,cfg,g,cal,short);predictions=dict(long_arx=long,short_arx=short_y);weights={}
    for variant in g['variants']:
        y,w=compose(ym,yr,evidence,variant);predictions[variant]=y;weights['weight_'+variant]=w
    return predictions,weights,evidence

def main():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError('Retain prior verification')
    g,cfg,base,reference,confirmation,prior=parents();p=json.loads((RUN/'development.json').read_text());q=json.loads((RUN/'retrospective/provenance.json').read_text());s=json.loads((RUN/'selection.json').read_text());r=dict(passed=False,checks=[],development_sha256=sha256(RUN/'development.json'),retrospective_sha256=sha256(RUN/'retrospective/provenance.json'),source_sha256=sha256(Path(__file__)),independent_confirmation=False)
    try:
        assert p['status']==q['status']=='complete';verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],RUN);verify_hashes(q['result_hashes'],RUN/'retrospective');assert q['selection_sha256']==sha256(RUN/'selection.json');assert q['development_sha256']==sha256(RUN/'development.json');r['original_files']=original_files();assert r['original_files']['passed']
        for family,fold in cfg['families'].items():
            for group,records in [('validation',load_allowed(cfg,fold['validation'],'validation')),('retrospective',load_confirmation(cfg,fold['confirmation']))]:
                metrics=pd.read_csv(RUN/('validation_metrics.csv' if group=='validation' else 'retrospective/metrics.csv'))
                for axis in fold['axes']:
                    name=f'{family}_{axis}';case=s['cases'][name];cal=LengthCalibration(**json.loads((base/name/'calibration.json').read_text()));short=load(RUN/name/'short_arx.npz');assert sha256(RUN/name/'short_arx.npz')==case['short_model_sha256']
                    for sid,seq in records.items():
                        part,_=packets(seq,axis,cfg,cal);predictions,weights,evidence=replay(part,name,case,cfg,g,cal,base,reference,prior,short)
                        if group=='retrospective':predictions['selected']=predictions[s['selected']].copy()
                        file=RUN/name/'validation.npz' if group=='validation' else RUN/'retrospective'/f'{name}_s{sid}.npz'
                        with np.load(file) as f:
                            np.testing.assert_array_equal(f['k'],part['k']);maximum=0.
                            for key,value in dict(**predictions,**weights,**evidence).items():
                                difference=float(np.abs(value-f[key]).max());assert difference<1e-12;maximum=max(maximum,difference)
                            for method,y in predictions.items():
                                if group=='validation' and method in ['long_arx','short_arx']:continue
                                for j,h in enumerate(cfg['horizons_raw']):
                                    row=metrics[(metrics['case']==name)&(metrics.scenario==sid)&(metrics.method==method)&(np.abs(metrics.horizon_s-h*.01)<1e-10)].iloc[0];np.testing.assert_allclose(np.abs(np.rad2deg(y[:,j]-part['target'][:,j])).mean(),row.mae_deg,rtol=1e-12,atol=1e-12)
                        if group=='validation':
                            values=s['case_scores'][name]
                            for method,y in predictions.items():np.testing.assert_allclose(score(y,part),values[method],rtol=1e-12)
                            assert case['reference']==('short' if values['short_arx']<values['long_arx'] else 'long')
                        small={k:v[:2].copy() for k,v in part.items()};a,w,e=replay(small,name,case,cfg,g,cal,base,reference,prior,short);small['target'][:]=999.;b,x,z=replay(small,name,case,cfg,g,cal,base,reference,prior,short)
                        for left,right in [(a,b),(w,x),(e,z)]:
                            for key in left:np.testing.assert_array_equal(left[key],right[key])
                        r['checks'].append(dict(group=group,case=name,scenario=sid,packets=len(part['k']),methods=len(predictions),maximum_replay_error=maximum,future_targets_ignored=True));print('VERIFIED',group,name,sid,maximum,flush=True)
        means={variant:float(np.mean([s['case_scores'][name][variant] for name in s['cases']])) for variant in g['variants']};assert min(means,key=means.get)==s['selected']
        for variant,value in means.items():np.testing.assert_allclose(value,s['validation_means'][variant],rtol=1e-12)
        r.update(passed=True,short_checkpoints=5,global_selection_validation_only=True,forecast_comparisons=sum(c['packets']*c['methods'] for c in r['checks']),backtest_targets_within_observed_history=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
