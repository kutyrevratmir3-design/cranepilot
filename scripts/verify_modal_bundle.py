"""Check the portable raw-history API against all frozen development forecasts."""
import json
from pathlib import Path
import numpy as np
from cranepilot.data import sha256
from cranepilot.modal_pilot import FrozenModalPilot,DEFAULT_BUNDLE
from scripts.modal_history import load_allowed
from scripts.run_external_benchmark import verify_hashes

OUT=Path('results/modal_campaign_20261107')
def main():
    OUT.mkdir(exist_ok=True);dest=OUT/'bundle_verification.json'
    if dest.exists():raise FileExistsError('Retain earlier verification')
    pilot=FrozenModalPilot();dev=Path('artifacts/runs/modal_history_20261107');ref=Path('artifacts/runs/modal_reference_challenge_20261107');p=json.loads((dev/'provenance.json').read_text());cfg=p['config'];verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],dev);r=dict(passed=False,source_sha256=sha256(Path(__file__)),bundle_manifest_sha256=sha256(DEFAULT_BUNDLE/'manifest.json'),checks=[])
    try:
        for family,fold in cfg['families'].items():
            records=load_allowed(cfg,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';pred=[];references=[];eligible=[];origin=[]
                for sid,s in records.items():
                    k=np.arange(int(cfg['origin_start_s']/.01),len(s.time)-max(cfg['horizons_raw']),cfg['origin_stride_raw']);coordinates=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if c=='x' else 'Jr' for c in coordinates];keys=['L',axis]+coordinates+refs;history={key:s.signals[key][k[:,None]-np.arange(401)] for key in keys};future={key:s.signals[key][k[:,None]+np.arange(151)] for key in refs};y=pilot.forecast(history,future,family=family,axis=axis,known_plan=True);mask=y['eligible'];pred.append(y['forecast_rad'][mask]);references.append(y['reference_arx_rad'][mask]);origin.append(k[mask]);eligible.extend(mask.tolist());assert np.isnan(y['forecast_rad'][~mask]).all()
                with np.load(dev/name/'validation_forecasts.npz') as f:
                    np.testing.assert_array_equal(np.concatenate(origin),f['k']);difference=float(np.abs(np.rad2deg(np.concatenate(pred)-f['selected_modal'])).max());assert difference<1e-7
                with np.load(ref/name/'validation.npz') as f:
                    np.testing.assert_array_equal(np.concatenate(origin),f['k']);reference_difference=float(np.abs(np.rad2deg(np.concatenate(references)-f['challenged_arx'])).max());assert reference_difference<1e-7
                r['checks'].append(dict(case=name,origins_considered=len(eligible),eligible=int(sum(eligible)),maximum_modal_replay_error_deg=difference,maximum_reference_replay_error_deg=reference_difference));print('BUNDLE VERIFIED',name,difference,reference_difference,flush=True)
        r.update(passed=True,eligible_packets=sum(c['eligible'] for c in r['checks']),raw_data_in_bundle=False,inference_requires_no_training_data=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
