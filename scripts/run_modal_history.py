"""Development only: restricted loader cannot open the seven conditional holdouts."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from cranepilot.benchmark import environment_record
from scripts.modal_history import load_allowed,packets,concatenate,modal,arx_statistics,fit_arx,predict_arx,save_arx,load_arx,local_ar,score
from scripts.run_external_benchmark import write

CONFIG=Path('configs/modal_history_20261107.json');RUN=Path('artifacts/runs/modal_history_20261107')
LOCKED=[CONFIG,Path('docs/MODAL_HISTORY_DESIGN_20261107.md'),Path('scripts/modal_history.py'),Path('scripts/run_modal_history.py'),Path('tests/test_modal_history.py')]

def main():
    cfg=json.loads(CONFIG.read_text());audit=Path('results/inteco_usage_audit_20261107/audit.json');a=json.loads(audit.read_text());assert a['passed'] and not a['signal_arrays_opened'] and a['candidate_ids']==cfg['sealed_confirmation_ids']
    for path in LOCKED:
        if subprocess.check_output(['git','show',f'HEAD:{path}'])!=path.read_bytes():raise ValueError(f'Commit first: {path}')
    if RUN.exists():raise FileExistsError('Retain all prior development')
    RUN.mkdir(parents=True);paths=LOCKED+[audit,Path('scripts/audit_inteco_holdouts.py'),Path('src/cranepilot/data.py'),Path('src/cranepilot/physics.py'),Path(cfg['data'])/'dataset.npz',Path(cfg['data'])/'manifest.json']
    prov=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,guarded_hashes={str(p):sha256(p) for p in paths},environment=environment_record(),records_read=[],confirmation_records_read=False,independent_confirmation=False)
    write(RUN/'provenance.json',prov);selection={};coverage=[];rows=[];metrics=[]
    try:
        for family,fold in cfg['families'].items():
            train=load_allowed(cfg,fold['train'],'train');val=load_allowed(cfg,fold['validation'],'validation');prov['records_read']+=fold['train']+fold['validation']
            assert not(set(prov['records_read'])&set(cfg['sealed_confirmation_ids']))
            for axis in fold['axes']:
                name=f'{family}_{axis}';out=RUN/name;out.mkdir();cal=LengthCalibration.fit(train,fold['train'],axis)
                write(out/'calibration.json',cal.__dict__);parts=[]
                for s in train.values():
                    part,c=packets(s,axis,cfg,cal);parts.append(part);coverage.append(dict(case=name,role='train',**c))
                tr=concatenate(parts);parts=[]
                for s in val.values():
                    part,c=packets(s,axis,cfg,cal);parts.append(part);coverage.append(dict(case=name,role='validation',**c))
                va=concatenate(parts);best=(np.inf,None);candidate=out/'candidates';candidate.mkdir()
                for form in cfg['arx_forms']:
                    stats=arx_statistics(tr,cfg,form)
                    for alpha in cfg['ridge_grid']:
                        m=fit_arx(stats,alpha);file=candidate/f'{form}_{alpha:g}.npz';save_arx(file,m);y=predict_arx(m,va,cfg);value=score(y,va)
                        np.testing.assert_array_equal(y,predict_arx(load_arx(file),va,cfg));rows.append(dict(case=name,family='arx',variant=form,alpha=alpha,validation_mae_deg=value,file=str(file.relative_to(RUN))))
                        if value<best[0]:best=value,file
                model=load_arx(best[1]);save_arx(out/'strong_arx.npz',model);forecasts=dict(strong_arx=predict_arx(model,va,cfg));modal_best=(np.inf,None)
                for variant in cfg['modal_variants']:
                    y,diag=modal(va,cfg,variant);np.savez_compressed(out/f'{variant}_diagnostics.npz',**diag);value=score(y,va);forecasts[variant]=y;rows.append(dict(case=name,family='modal',variant=variant,alpha=None,validation_mae_deg=value,file=None))
                    if value<modal_best[0]:modal_best=value,variant
                    print('MODAL',name,variant,value,flush=True)
                local_best=(np.inf,None)
                settings=[dict(order=o,alpha=a,tls=False) for o in cfg['local_ar_orders'] for a in cfg['local_ar_ridge']]+[dict(order=2,alpha=0.,tls=True)]
                for setting in settings:
                    y=local_ar(va,cfg,**setting);value=score(y,va);label=('tls' if setting['tls'] else 'ridge')+f"_ar{setting['order']}_{setting['alpha']:g}";rows.append(dict(case=name,family='local_ar',variant=label,alpha=setting['alpha'],validation_mae_deg=value,file=None))
                    if value<local_best[0]:local_best=value,setting
                forecasts['selected_modal']=forecasts[modal_best[1]].copy();forecasts['local_ar']=local_ar(va,cfg,**local_best[1]);forecasts['persistence']=np.repeat(va['theta'][:,:1],len(cfg['horizons_raw']),axis=1)
                selection[name]=dict(family=family,axis=axis,modal=modal_best[1],local_ar=local_best[1],arx_form=model['form'],arx_alpha=model['alpha'],validation_scores={method:score(y,va) for method,y in forecasts.items()},arx_checkpoint_sha256=sha256(out/'strong_arx.npz'),calibration_sha256=sha256(out/'calibration.json'))
                np.savez_compressed(out/'validation_forecasts.npz',k=va['k'],scenario=va['scenario'],**forecasts)
                for method,y in forecasts.items():
                    for sid in fold['validation']:
                        mask=va['scenario']==sid
                        for j,h in enumerate(cfg['horizons_raw']):
                            err=np.rad2deg(y[mask,j]-va['target'][mask,j]);metrics.append(dict(case=name,scenario=sid,method=method,horizon_s=h*.01,mae_deg=float(np.abs(err).mean()),rmse_deg=float(np.sqrt((err**2).mean()))))
                print('CASE',name,json.dumps(selection[name]),flush=True)
        pd.DataFrame(rows).to_csv(RUN/'candidate_validation.csv',index=False);pd.DataFrame(metrics).to_csv(RUN/'validation_metrics.csv',index=False);pd.DataFrame(coverage).to_csv(RUN/'coverage.csv',index=False)
        arx=np.array([s['validation_scores']['strong_arx'] for s in selection.values()]);new=np.array([s['validation_scores']['selected_modal'] for s in selection.values()]);local=np.array([s['validation_scores']['local_ar'] for s in selection.values()]);gate=bool(new.mean()/arx.mean()<=cfg['development_gate']['maximum_mae_ratio'] and (new<arx).sum()>=cfg['development_gate']['minimum_case_wins'] and new.mean()<local.mean())
        chosen=dict(cases=selection,development_gate_passed=gate,validation_mean_arx_deg=float(arx.mean()),validation_mean_selected_deg=float(new.mean()),validation_mean_local_ar_deg=float(local.mean()),change_percent=float(100*(new.mean()/arx.mean()-1)),case_wins=int((new<arx).sum()),confirmation_records_read=False,independent_confirmation=False,final_goal_achieved=False)
        write(RUN/'selection.json',chosen);prov.update(status='complete',records_read=sorted(set(prov['records_read'])),selection_sha256=sha256(RUN/'selection.json'),result_hashes={str(p.relative_to(RUN)):sha256(p) for p in RUN.rglob('*') if p.is_file() and p.name!='provenance.json'})
        print('DEVELOPMENT',json.dumps({k:v for k,v in chosen.items() if k!='cases'}),flush=True)
    except Exception as e:prov.update(status='failed',failure=f'{type(e).__name__}: {e}');raise
    finally:write(RUN/'provenance.json',prov)

if __name__=='__main__':main()
