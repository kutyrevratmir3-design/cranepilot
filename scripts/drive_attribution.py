"""Late, fixed attribution controls. These never enter model selection."""
import numpy as np
from scripts.modal_history import modal
from scripts.modal_reference_challenge import predict as old_arx
from scripts.modal_backtest_gate import sliced,compose
from scripts.run_drive_conditioned import ref_prediction
from scripts import drive_conditioned as dc

def controls(p,cfg,task,pilot,name,cal,prior,long_model,short_model,choice):
    if 'target' in p:raise ValueError('Targets must be separate')
    prefix=sliced(p,20,None,cal);free=modal(p,task,'two_mode')[0];prefix_free=modal(prefix,task,'two_mode')[0];ref=ref_prediction(p,free,choice,long_model,short_model,pilot,name,cal,task);back_ref=old_arx(pilot.short_models[name],prefix,task) if choice=='old' else dc.arx_predict(short_model,prefix,prefix_free);observed=p['theta'][:,[18,16,13,10,5,0]];er=np.abs(back_ref[:,:6]-observed).mean(1);local=dc.prepare(p,task,'hold',prior['bounds'])['unforced'];past_local=dc.prepare(prefix,task,'hold',prior['bounds'])['unforced'];result={}
    for key,y,back in [('matched_free_blend',free,prefix_free),('matched_local_zero_blend',local,past_local)]:
        em=np.abs(back[:,:6]-observed).mean(1);result[key]=compose(y,ref,dict(modal_backtest_mae=em,arx_backtest_mae=er),'soft')[0]
    return result
