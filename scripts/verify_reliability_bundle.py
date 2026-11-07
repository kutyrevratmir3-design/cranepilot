"""Compare the portable causal API with frozen gate outputs on every scored window."""
import json
from pathlib import Path
import numpy as np
from cranepilot.data import sha256
from cranepilot.reliability_pilot import FrozenReliabilityPilot,DEFAULT_RELIABILITY_BUNDLE
from scripts.modal_history import load_allowed
from scripts.run_modal_confirmation import load_confirmation
from scripts.run_external_benchmark import verify_hashes

OUT=Path('results/modal_backtest_gate_20261107')
def main():
    dest=OUT/'bundle_verification.json'
    if dest.exists():raise FileExistsError('Retain prior verification')
    pilot=FrozenReliabilityPilot();run=Path('artifacts/runs/modal_backtest_gate_20261107');p=json.loads((run/'development.json').read_text());cfg=p['task'];s=json.loads((run/'selection.json').read_text());verify_hashes(p['guarded_hashes']);verify_hashes(p['result_hashes'],run);r=dict(passed=False,source_sha256=sha256(Path(__file__)),bundle_manifest_sha256=sha256(DEFAULT_RELIABILITY_BUNDLE/'manifest.json'),checks=[],independent_confirmation=False)
    try:
        for family,fold in cfg['families'].items():
            for group,records in [('validation',load_allowed(cfg,fold['validation'],'validation')),('retrospective',load_confirmation(cfg,fold['confirmation']))]:
                for axis in fold['axes']:
                    name=f'{family}_{axis}'
                    for sid,seq in records.items():
                        k=np.arange(int(cfg['origin_start_s']/.01),len(seq.time)-max(cfg['horizons_raw']),cfg['origin_stride_raw']);coords=['x'] if family=='trolley' else ['J'] if family=='jib' else ['x','J'];refs=['xr' if c=='x' else 'Jr' for c in coords];keys=['L',axis]+coords+refs;history={key:seq.signals[key][k[:,None]-np.arange(401)] for key in keys};future={key:seq.signals[key][k[:,None]+np.arange(151)] for key in refs};result=pilot.forecast(history,future,family=family,axis=axis,known_plan=True);mask=result['eligible'];file=run/name/'validation.npz' if group=='validation' else run/'retrospective'/f'{name}_s{sid}.npz'
                        with np.load(file) as f:
                            np.testing.assert_array_equal(k[mask],f['k']);expected=f[s['selected']] if group=='validation' else f['selected'];maximum=float(np.abs(np.rad2deg(result['forecast_rad'][mask]-expected)).max());reference_max=float(np.abs(np.rad2deg(result['reference_arx_rad'][mask]-f['reference'])).max());assert maximum<1e-7 and reference_max<1e-7;np.testing.assert_allclose(result['modal_weight'][mask],f['weight_'+s['selected']],rtol=1e-12,atol=1e-12)
                        assert np.isnan(result['forecast_rad'][~mask]).all();r['checks'].append(dict(group=group,case=name,scenario=sid,eligible=int(mask.sum()),maximum_forecast_replay_deg=maximum,maximum_reference_replay_deg=reference_max));print('BUNDLE VERIFIED',group,name,sid,maximum,reference_max,flush=True)
        r.update(passed=True,case_records=len(r['checks']),packets=sum(c['eligible'] for c in r['checks']),no_dataset_or_ancestral_run_required_by_api=True)
    except Exception as e:r['failure']=f'{type(e).__name__}: {e}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
