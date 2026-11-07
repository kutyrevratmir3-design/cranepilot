"""Frozen conditional-history selection and guarded residual-transfer controls."""
import numpy as np
from cranepilot.forced_observer import refine_windows
from scripts.forced_adaptive_candidate import ForcedAdaptiveCandidate,compose
from scripts.history_backtest_physics import backtest_forced_forecast


class HistoryBacktestCandidate(ForcedAdaptiveCandidate):
    def forecast(self,raw,prescribed_acceleration=None):
        previous,previous_meta,acceleration=super().forecast(raw,prescribed_acceleration)
        controls=['incumbent','soft_symmetry','combined','physical','adaptive','adaptive_fine']
        pred={name:previous[name] for name in controls}
        metadata={name:previous_meta[name] for name in ['adaptive','adaptive_fine']}
        theta=raw['angular_context'][:,:41]*.1;x=raw['actuator_context'][:,:41]*.1
        for name,choice in self.config['backtest_variants'].items():
            pred[name],metadata[name]=backtest_forced_forecast(theta,x,raw['omega'],acceleration,
                velocity=raw['initial_velocity'],**choice)
        rms=np.sqrt(np.mean(refine_windows(raw)['forced_residual']**2,axis=1))
        hybrid=compose(pred['soft_symmetry'],pred['physical'],pred['bt_guarded'],rms,self.config['tau_bins'])
        pred.update({'history_residual_transfer':hybrid['residual_transfer'],
                     'history_residual_gate':hybrid['residual_gate'],
                     'history_soft_gate':hybrid['soft_adaptive_gate']})
        if set(pred)!=set(self.config['comparison_models']):raise ValueError('Unexpected comparison pool')
        if not all(np.isfinite(y).all() for y in pred.values()):raise ValueError('Nonfinite prediction')
        return pred,metadata,acceleration
