"""Research-only evaluation outside API eligibility; never changes the frozen API."""
import numpy as np
from cranepilot.physics import LengthCalibration
from scripts.modal_history import modal
from scripts.modal_reference_challenge import predict
from scripts.modal_backtest_gate import sliced,backtest,compose

def qualification_packets(s,axis,task,cfg,cal):
    history=task['history_samples'];spacing=task['history_spacing_raw'];h=np.array(task['horizons_raw']);lo=max((history-1)*spacing,int(round(cfg['origin_start_s']/.01)));k=np.arange(lo,len(s.time)-h.max(),cfg['origin_stride_raw']);raw=k[:,None]-np.arange((history-1)*spacing+1);sampled=k[:,None]-np.arange(history)*spacing
    p=dict(theta=s.signals[axis][sampled],length=s.signals['L'][sampled],omega=cal.omega(s.signals['L'][k]),k=k,scenario=np.full(len(k),s.sid,dtype=int))
    for key in ['x','xr','J','Jr']:
        if key in s.signals:p[key]=s.signals[key][sampled]
    drive=np.zeros(len(k),dtype=bool)
    for key in ['x','J']:
        if key in s.signals:drive |= np.ptp(s.signals[key][raw],axis=1)>task['past_position_range']
    length=np.ptp(s.signals['L'][raw],axis=1)>task['past_length_range_m'];regime=drive.astype(int)+2*length.astype(int);old=(s.time[k]>=task['origin_start_s'])&~drive&~length
    for key in ['xr','Jr']:
        if key in s.signals:old &= np.ptp(s.signals[key][k[:,None]+np.arange(h.max()+1)],axis=1)<=task['future_reference_range']
    labels=dict(regime=regime,drive_active=drive,length_active=length,old_numerical_scope=old,time_s=s.time[k])
    target=s.signals[axis][k[:,None]+h]
    return p,target,labels

def frozen_predictions(p,pilot,case_name):
    if 'target' in p:raise ValueError('Targets must be kept separate from inference')
    task=pilot.cfg;case=pilot.reliability_manifest['cases'][case_name];cal=LengthCalibration(**pilot.cases[case_name]['calibration']);short=pilot.short_models[case_name];ym,diag=modal(p,task,'two_mode');long=predict(pilot.cases[case_name]['reference'],p,task);short_y=predict(short,sliced(p,0,61,cal),task);reference=short_y if case['reference']=='short' else long;evidence=backtest(p,task,pilot.gate_cfg,cal,short);blend,w=compose(ym,reference,evidence,pilot.reliability_manifest['selected']);out=dict(strong_arx=reference,frozen_modal=ym,frozen_blend=blend,fixed_frequency=modal(p,task,'fixed')[0],persistence=np.repeat(p['theta'][:,:1],len(task['horizons_raw']),axis=1))
    if not all(np.isfinite(y).all() for y in out.values()):raise ValueError('Nonfinite forecast: retain failed run')
    return out,dict(modal_weight=w,history_sse=diag['history_sse'],primary_omega=diag['omega'],modal_backtest_mae=evidence['modal_backtest_mae'],arx_backtest_mae=evidence['arx_backtest_mae'])
