import json
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from cranepilot.data import Scenario
from cranepilot.physics import LengthCalibration
from scripts.forced_adaptive_physics import forced_fit,damped_rollout
from scripts.global_weak_physics import fit_global,forecast_global


def training_record(sid,length,calibration,parameters=(.35,1.12,.72),offset=0.):
    time=np.arange(0.,20.0001,.01)
    omega=float(calibration.omega(length));c,kscale,bscale=parameters
    kappa=omega**2*kscale;coupling=omega**2/9.81*bscale
    def x(t):return offset+.018*np.sin(2.3*t+.2)+.011*np.sin(5.1*t+.8)+.007*np.cos(.7*t)
    def acc(t):return -.018*2.3**2*np.sin(2.3*t+.2)-.011*5.1**2*np.sin(5.1*t+.8)-.007*.7**2*np.cos(.7*t)
    solved=solve_ivp(lambda t,y:[y[1],-c*y[1]-kappa*y[0]-coupling*acc(t)],
                     (time[0],time[-1]),[.07,-.1],t_eval=time,rtol=2e-11,atol=2e-13,method='DOP853')
    assert solved.success
    return Scenario(sid,time,{'theta':solved.y[0],'x':x(time),'L':np.full(len(time),length)}, {'family':'trolley'})


def independent_packet(omega,model):
    time=np.arange(41)*.01;x=.03*np.sin(4*time)+.02*time*time
    slope=np.diff(x)/.01
    kappa=model['kscale']*omega**2;coupling=model['bscale']*omega**2/9.81
    matrix=np.array([[0.,1.],[-kappa,-model['c']]])
    transition=expm(.01*matrix);state=np.array([.08,-.14]);angles=[state[0]]
    for i in range(40):
        if i:state[1]-=coupling*(slope[i]-slope[i-1])
        state=transition@state;angles.append(state[0])
    acceleration=(.13*np.cos(np.arange(30)*.2))[None]
    future=[];step=expm(.05*matrix)
    for a in acceleration[0]:
        equilibrium=np.array([-coupling/kappa*a,0.])
        state=equilibrium+step@(state-equilibrium);future.append(state[0])
    return np.asarray(angles)[None,::-1],x[None,::-1],acceleration,np.asarray(future)[None],slope[-1]


class GlobalWeakPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration=LengthCalibration(a=1.,b=.05)
        cls.records=[training_record(1,.3,cls.calibration),training_record(2,.7,cls.calibration)]

    def test_weak_identification_recovers_nonzero_damping_and_independent_coupling(self):
        model=fit_global(self.records,self.calibration)
        np.testing.assert_allclose([model['c'],model['kscale'],model['bscale']],[.35,1.12,.72],atol=3e-4,rtol=3e-4)
        self.assertTrue(model['identified_full_rank'])
        self.assertEqual(model['training_sids'],[1,2])
        self.assertEqual({r['duration_s'] for r in model['row_details']},{1.,2.,4.})
        json.dumps(model,allow_nan=False)

    def test_regularization_shrinks_toward_declared_prior(self):
        ordinary=fit_global(self.records,self.calibration)
        regularized=fit_global(self.records,self.calibration,ridge=.1)
        vector=lambda m:np.array([m['c'],m['kscale'],m['bscale']])
        prior=np.array([0.,1.,1.])
        self.assertLess(np.linalg.norm(vector(regularized)-prior),np.linalg.norm(vector(ordinary)-prior))
        self.assertGreater(regularized['scaled_data_mse'],ordinary['scaled_data_mse'])

    def test_training_position_offset_invariance_and_calibration_split_guard(self):
        base=fit_global(self.records,self.calibration)
        shifted=[Scenario(r.sid,r.time,{**r.signals,'x':r.signals['x']+11.},r.metadata) for r in self.records]
        changed=fit_global(shifted,self.calibration)
        np.testing.assert_allclose([changed[k] for k in ['c','kscale','bscale']],
                                   [base[k] for k in ['c','kscale','bscale']],atol=1e-10,rtol=1e-10)
        wrong=LengthCalibration(evidence=[{'sid':1}])
        with self.assertRaises(ValueError):fit_global(self.records,wrong)

    def test_zero_equilibrium_retains_nominal_prior_without_claiming_identification(self):
        t=np.arange(0.,8.0001,.01)
        s=Scenario(8,t,{'theta':np.zeros(len(t)),'x':np.full(len(t),7.),'L':np.full(len(t),.5)}, {'family':'trolley'})
        model=fit_global([s],self.calibration)
        np.testing.assert_array_equal([model['c'],model['kscale'],model['bscale']],[0.,1.,1.])
        self.assertFalse(model['identified_full_rank']);self.assertEqual(model['data_rank'],0)
        prediction=forecast_global(model,np.zeros((1,41)),np.full((1,41),7.),5.,np.zeros((1,30)))
        np.testing.assert_array_equal(prediction,0.)

    def test_forecast_matches_independent_forced_damped_solution(self):
        model={'c':.35,'kscale':1.12,'bscale':.72}
        theta,x,a,target,velocity=independent_packet(6.1,model)
        actual=forecast_global(model,theta,x,6.1,a,velocity=np.array([velocity]))
        np.testing.assert_allclose(actual,target,atol=3e-11,rtol=3e-10)
        shifted=forecast_global(model,theta,x+9.,6.1,a,velocity=np.array([velocity]))
        np.testing.assert_allclose(shifted,actual,atol=3e-11,rtol=3e-10)
        reflected=forecast_global(model,-theta,-x,6.1,-a,velocity=np.array([-velocity]))
        np.testing.assert_allclose(reflected,-actual,atol=3e-11,rtol=3e-10)

    def test_fitted_model_forecast_and_nominal_equivalence(self):
        model=fit_global(self.records,self.calibration)
        theta,x,a,target,velocity=independent_packet(6.1,{'c':.35,'kscale':1.12,'bscale':.72})
        actual=forecast_global(model,theta,x,6.1,a,velocity=np.array([velocity]))
        np.testing.assert_allclose(actual,target,atol=2e-5,rtol=2e-4)
        nominal={'c':0.,'kscale':1.,'bscale':1.}
        q,v,_=forced_fit(theta,x,6.1,0.)
        expected=damped_rollout(q,v,6.1,0.,a)
        np.testing.assert_array_equal(forecast_global(nominal,theta,x,6.1,a),expected)

    def test_overdamped_and_critical_fallback_remain_stable_and_correct(self):
        model={'c':2.,'kscale':1.,'bscale':.8}
        for omega in [.5,1.]:
            theta,x,a,target,velocity=independent_packet(omega,model)
            actual=forecast_global(model,theta,x,omega,a,velocity=np.array([velocity]))
            np.testing.assert_allclose(actual,target,atol=3e-10,rtol=3e-9)
            self.assertTrue(np.isfinite(actual).all())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):fit_global([],self.calibration)
        with self.assertRaises(ValueError):fit_global(self.records,self.calibration,ridge=-1.)
        with self.assertRaises(ValueError):fit_global([self.records[0],self.records[0]],self.calibration)
        with self.assertRaises(ValueError):forecast_global({'c':0.,'kscale':0.,'bscale':1.},np.zeros((1,41)),np.zeros((1,41)),5.,np.zeros((1,30)))
        with self.assertRaises(ValueError):forecast_global({'c':0.,'kscale':1.,'bscale':1.},np.zeros((1,41)),np.zeros((1,41)),5.,np.zeros((1,29)))


if __name__=='__main__':unittest.main()
