"""Post-hoc failure diagnosis on opened records; future signals are diagnostics only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import packets
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import verify_hashes

OUT=Path('results/modal_diagnostics_20261107')
def main():
    if OUT.exists():raise FileExistsError('Preserve earlier diagnosis')
    OUT.mkdir(parents=True);base=Path('artifacts/runs/modal_history_20261107');conf=Path('artifacts/runs/modal_confirmation_20261107');ref=Path('artifacts/runs/modal_reference_challenge_20261107');p=json.loads((conf/'provenance.json').read_text());cfg=p['config'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],conf);q=json.loads((ref/'retrospective/provenance.json').read_text());verify_hashes(q['result_hashes'],ref/'retrospective');rows=[]
    for family,fold in cfg['families'].items():
        records=load_confirmation(cfg,fold['confirmation'])
        for axis in fold['axes']:
            name=f'{family}_{axis}';cal=LengthCalibration(**json.loads((base/name/'calibration.json').read_text()))
            for sid,s in records.items():
                part,_=packets(s,axis,cfg,cal);k=part['k'];past=k[:,None]-np.arange(401);future=k[:,None]+np.arange(151)
                with np.load(conf/f'{name}_s{sid}.npz') as f:
                    np.testing.assert_array_equal(f['k'],k);modal_error=np.abs(np.rad2deg(f['selected_modal']-part['target'])).mean(1);history_rms=np.rad2deg(np.sqrt(f['diagnostic_history_sse']/81));ratio=f['diagnostic_omega']/part['omega'];second=f['diagnostic_second_omega']/part['omega'];modes=f['diagnostic_modes']
                with np.load(ref/'retrospective'/f'{name}_s{sid}.npz') as f:np.testing.assert_array_equal(f['k'],k);arx_error=np.abs(np.rad2deg(f['challenged_arx']-part['target'])).mean(1)
                item=dict(case=name,scenario=sid,k=k,time_s=s.time[k],modal_mae_deg=modal_error,arx_mae_deg=arx_error,history_fit_rms_deg=history_rms,primary_frequency_ratio=ratio,secondary_frequency_ratio=second,modes=modes,past_length_range_m=np.ptp(s.signals['L'][past],axis=1),future_measured_length_range_m_DIAGNOSTIC_ONLY=np.ptp(s.signals['L'][future],axis=1))
                for coordinate in ['x','J']:
                    if coordinate in s.signals:item[f'future_measured_{coordinate}_range_DIAGNOSTIC_ONLY']=np.ptp(s.signals[coordinate][future],axis=1)
                frame=pd.DataFrame(item);frame.to_csv(OUT/f'{name}_s{sid}.csv',index=False)
                for segment,ids in [('all',np.arange(len(k)))]+[(f'chronological_quartile_{j+1}',a) for j,a in enumerate(np.array_split(np.arange(len(k)),4))]:
                    v=frame.iloc[ids];rows.append(dict(case=name,scenario=sid,segment=segment,windows=len(v),start_s=float(v.time_s.min()),end_s=float(v.time_s.max()),modal_mae_deg=float(v.modal_mae_deg.mean()),arx_mae_deg=float(v.arx_mae_deg.mean()),history_fit_rms_deg=float(v.history_fit_rms_deg.mean()),primary_boundary_fraction=float(((v.primary_frequency_ratio<=.505)|(v.primary_frequency_ratio>=1.595)).mean()),secondary_upper_boundary_fraction=float(((v.modes==2)&(v.secondary_frequency_ratio>=2.99)).mean()),future_length_change_over2mm_fraction=float((v.future_measured_length_range_m_DIAGNOSTIC_ONLY>.002).mean())))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);report=dict(source_sha256=sha256(Path(__file__)),confirmation_provenance_sha256=sha256(conf/'provenance.json'),reference_provenance_sha256=sha256(ref/'retrospective/provenance.json'),post_hoc=True,independent_confirmation=False,source_models_changed=False,future_measured_channels_used_for_diagnosis_only=True,rows_sha256=sha256(OUT/'summary.csv'),scope='All12 previously scored case/records and fixed chronological quartiles. No fitting, exclusions, new predictions or operating threshold changes; future diagnostics cannot be used as an online gate.')
    (OUT/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(pd.DataFrame(rows).query("scenario==21").to_string(index=False))
if __name__=='__main__':main()
