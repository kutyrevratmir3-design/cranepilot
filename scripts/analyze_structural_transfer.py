"""Posthoc distribution audit; no selection, tuning or controller modification."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cranepilot.data import sha256

def main():
    out=Path('results/structural_transfer_20261107');run=Path('artifacts/runs/structural_transfer_20261107');p=json.loads((run/'provenance.json').read_text());cfg=p['config']
    dest=out/'independent_metrics.json'
    if dest.exists():raise FileExistsError('Preserve prior audit')
    frame=pd.read_csv(run/'evaluate/cases.csv');expected=pd.read_csv(run/'evaluate/metrics.csv');checks=[];distribution=[];family=[]
    for method in cfg['methods']:
        with np.load(run/'evaluate'/f'{method}.npz') as z:
            latent=z['raw_states'][:,40::5];s=z['states'];grid=np.arange(s.shape[1])*.05
            # Independently recover net acceleration, without the plant/metric helpers.
            active=frame.family.isin(['friction','combined']).to_numpy()[:,None]
            acceleration=latent[:,:,4]-active*(frame.coulomb.to_numpy()[:,None]*np.tanh(latent[:,:,1]/.015)+.08*latent[:,:,1])
            np.testing.assert_array_equal(acceleration,s[:,:,4])
            target=np.where(grid<4.,.3,.15);error=latent[:,:,0]-target
            cost=np.trapezoid((error/.05)**2+(latent[:,:,2]/.035)**2+.02*(latent[:,:,1]/.2)**2+.05*(acceleration/.5)**2,grid,axis=1)+5*(error[:,-1]/.05)**2+5*(latent[:,-1,2]/.035)**2+(latent[:,-1,1]/.1)**2
            saved=expected[expected.method==method].set_index('case');difference=float(np.abs(cost-saved.realized_cost.to_numpy()).max());assert difference<1e-10
            checks.append(dict(method=method,cases=len(frame),cost_max_abs_difference=difference,net_acceleration_exact=True))
            distribution.append(dict(method=method,cost_mean=cost.mean(),cost_median=np.median(cost),cost_p95=np.quantile(cost,.95),cost_max=cost.max()))
    a=expected[expected.method=='adaptive_force_observer'].set_index('case');b=expected[expected.method=='force_actuator'].set_index('case')
    for f in cfg['families']:
        ca=a[a.family==f];cb=b[b.family==f]
        family.append(dict(family=f,cost_change_percent=100*(ca.realized_cost.mean()/cb.realized_cost.mean()-1),cost_wins=int((ca.realized_cost<cb.realized_cost).sum()),cases=len(ca),tracking_change_percent=100*(ca.tracking_ise_m2_s.mean()/cb.tracking_ise_m2_s.mean()-1)))
    pd.DataFrame(family).to_csv(out/'posthoc_adaptive_vs_static_families.csv',index=False)
    pd.DataFrame(distribution).to_csv(out/'posthoc_cost_distribution.csv',index=False)
    with np.load(run/'evaluate/adaptive_force_observer.npz') as a,np.load(run/'evaluate/frozen_adaptive_force_observer.npz') as b:
        exact_keys=[k for k in a.files if k!='batch_seconds']
        for key in exact_keys:np.testing.assert_array_equal(a[key],b[key])
    info=dict(passed=True,posthoc=True,source_sha256=sha256(Path(__file__)),checks=checks,new_and_frozen_observer_all_nontiming_arrays_identical=True,no_selection_or_tuning=True)
    dest.write_text(json.dumps(info,indent=2)+'\n');print(pd.DataFrame(family).to_string(index=False))

if __name__=='__main__':main()
