import copy,json,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from cranepilot.regime_pilot import FrozenRegimePilot
from cranepilot.physics import LengthCalibration
from scripts import dated_forecast as df
from scripts import latency_observer as lo

class DatedForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot=FrozenRegimePilot();cls.cfg=json.loads(Path('configs/dated_forecast_20261108.json').read_text());cls.cal=LengthCalibration(**cls.pilot.pilot.cases['trolley_theta']['calibration'])
    def raw(self,age=0):
        t=-np.arange(401)*.01;w=self.cal.omega(.4)
        return dict(theta=(.05*np.cos(w*(t-age*.01)+.3))[None],L=np.full((1,401),.4),x=np.zeros((1,401)),xr=np.zeros((1,401)))
    def test_zero_age_all_candidates_exactly_preserve_parent(self):
        raw=self.raw();y,d=df.predictions(raw,0,'theta',self.pilot,'trolley_theta',self.cfg['methods']);parent=self.pilot.forecast(raw,family='trolley',axis='theta')['forecast_rad']
        for p in y.values():np.testing.assert_array_equal(p,parent)
        self.assertTrue(d['bank_bypassed'])
    def test_parent_extension_matches_frozen_source_at_supported_ages(self):
        for age in [0,5,15]:
            raw=self.raw(age);np.testing.assert_array_equal(df.parent_quad(raw,age,'theta',self.pilot,'trolley_theta'),lo.predict(raw,age,'theta','quadratic_0.25',self.pilot,'trolley_theta')[0])
    def test_extended_head_reindexes_real_observations_and_recovers_ramp(self):
        raw=self.raw(25);t=-np.arange(401)*.01-.25;raw['theta'][:]=.1+.03*t;before=copy.deepcopy(raw)
        aligned=df.extended_head(raw,25,'theta','quadratic_0.25',self.pilot,'trolley_theta');np.testing.assert_allclose(aligned['theta'],(.1-.03*np.arange(401)*.01)[None],atol=1e-13)
        for key in raw:np.testing.assert_array_equal(raw[key],before[key])
    def test_constant_length_preserves_analytical_free_model(self):
        raw=self.raw(25);theta=raw['theta'][:,:251:5];omega=self.cal.omega(raw['L'][:,25]);basis,H,G,c,w,d,low=lo.mode_fit(theta,omega,self.pilot.pilot.cfg,True);h=np.array([10,50,150]);free=np.einsum('nti,ni->nt',basis((h+25)*.01),c);y=df.variable_length_free(raw,25,h,self.cal,c,w,d,np.ones(1)*7,low,free);np.testing.assert_array_equal(y,free)
    def test_variable_length_matches_independent_continuous_ode(self):
        age=25;raw=self.raw(age);raw['L'][0,:age+1]=.4+.02*np.arange(age+1)*.01;c=np.array([[.01,.05,.02,.007,.003]]);w=np.array([4.]);d=np.array([.1]);second=np.array([7.]);h=np.array([10,50,150]);y=df.variable_length_free(raw,age,h,self.cal,c,w,d,second,np.array([False]),np.zeros((1,3)));L0=self.cal.a*.405+self.cal.b
        def f(t,z):
            L=self.cal.a*(.405-.02*min(t,.25))+self.cal.b;rate=-.02*self.cal.a if t<.25 else 0.;return [z[1],-(16+.1**2/4)*L0/L*z[0]-(.1+2*rate/L)*z[1],z[3],-49*L0/L*z[2]-2*rate/L*z[3]]
        sol=solve_ivp(f,[0,1.75],[.05,-.1*.05/2+4*.02,.007,7*.003],t_eval=(h+age)*.01,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(y[0],.01+sol.y[0]+sol.y[2],atol=3e-6,rtol=0)
    def test_future_driver_uses_observed_gap_then_explicit_assumption(self):
        raw=self.raw(25);raw['x'][:]=np.arange(401)*.001;uh=np.array([[[.025]]]);held=df.future_drivers(raw,25,35,['x'],uh,np.array([[0.,1.]]),'hold');np.testing.assert_array_equal(held[0,:26,0],raw['x'][0,25::-1]);np.testing.assert_array_equal(held[0,26:,0],0.)
        linear=df.future_drivers(raw,25,35,['x'],uh,np.array([[-1.,1.]]),'linear');np.testing.assert_allclose(linear[0,26:,0],-.001*np.arange(1,11),atol=1e-14)
    def test_shadow_truth_has_six_received_angles_and_no_estimated_head(self):
        raw=self.raw(25);before=copy.deepcopy(raw);pred,d=df.predictions(raw,25,'theta',self.pilot,'trolley_theta',self.cfg['methods']);np.testing.assert_array_equal(d['truth_indices'],[65,55,40,25,5,0]);self.assertEqual(d['past_errors'].shape,(1,6,6));self.assertEqual(len(pred),21)
        shadow={k:v[:,100:] for k,v in raw.items()};back=df.direct_bank(shadow,25,'theta',self.pilot,'trolley_theta',d['shadow_heads']);np.testing.assert_array_equal(d['past_errors'],back-raw['theta'][:,d['truth_indices'],None])
        for key in raw:np.testing.assert_array_equal(raw[key],before[key])
        for key in ['weight_bank_simplex_0.1','weight_bank_simplex_1','weight_bank_simplex_10']:self.assertTrue((d[key]>=0).all());np.testing.assert_allclose(d[key].sum(1),1.)
    def test_invalid_age_future_fields_and_missing_signals_rejected(self):
        for age in [None,-1,26,.5,True,np.bool_(True)]:
            with self.assertRaises(ValueError):df.validate(self.raw(),age,'theta')
        with self.assertRaises(ValueError):df.validate(dict(self.raw(),future=np.zeros((1,7))),5,'theta')
        with self.assertRaises(ValueError):df.validate(dict(L=np.ones((1,401))),5,'theta')

if __name__=='__main__':unittest.main()
