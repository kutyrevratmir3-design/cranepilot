"""Poison unreceived angle samples and future channels; inference must be identical."""
import copy,json
from pathlib import Path
import numpy as np
from cranepilot.data import sha256
from cranepilot.physics import LengthCalibration
from scripts.modal_history import load_allowed
from scripts.run_latency_observer import OUT,setup,current_packet,raw_history
from scripts import latency_observer as lo

def main():
    dest=OUT/'clock_causality.json'
    if dest.exists():raise FileExistsError('Retain prior clock audit')
    cfg,qc,pilot,task=setup();selection=json.loads(Path('artifacts/runs/latency_observer_20261108/selection.json').read_text());r=dict(passed=False,source_sha256=sha256(Path(__file__)),checks=[],scope='Validation-only metadata-fixed input causality audit; no new model selection',independent_confirmation=False);index=0
    try:
        for family,fold in task['families'].items():
            records=load_allowed(task,fold['validation'],'validation')
            for axis in fold['axes']:
                name=f'{family}_{axis}';cal=LengthCalibration(**pilot.pilot.cases[name]['calibration'])
                for sid,s in records.items():
                    packet,_,_=current_packet(s,axis,task,qc,cal);k=packet['k'][:1];origin=int(k[0])
                    for age in [0,3,15]:
                        altered=copy.deepcopy(s)
                        for key,values in altered.signals.items():values[origin+1:]=1e6
                        altered.signals[axis][origin-age+1:]=np.linspace(1e5,2e5,len(s.time)-(origin-age+1));a=raw_history(s,k,axis,family,age,0,cfg,index);b=raw_history(altered,k,axis,family,age,0,cfg,index)
                        for key in a:np.testing.assert_array_equal(a[key],b[key])
                        ya=lo.predict(a,age,axis,selection['selected'],pilot,name)[0];yb=lo.predict(b,age,axis,selection['selected'],pilot,name)[0];np.testing.assert_array_equal(ya,yb);r['checks'].append(dict(case=name,scenario=sid,age_raw=age,origin_raw=origin,unreceived_angle_and_future_poison_had_no_effect=True));print('CLOCK_VERIFIED',name,age,flush=True)
                index+=1
        r.update(passed=True,probes=len(r['checks']))
    except Exception as error:r['failure']=f'{type(error).__name__}: {error}';raise
    finally:dest.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
