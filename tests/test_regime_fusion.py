import unittest
import numpy as np
from scipy.optimize import minimize
from cranepilot.data import QUANTIZATION_RAD
from cranepilot.regime_pilot import FrozenRegimePilot
from scripts.regime_fusion import context,simplex_fit,fuse

class RegimeFusionTests(unittest.TestCase):
    def evidence(self):
        rng=np.random.default_rng(71);n=5;F=rng.normal(size=(n,7,4));a=np.tile([.6,0,.4,0],(n,1));b=np.tile([0,.3,0,.7],(n,1));return dict(forecasts=F,errors=rng.normal(0,.01,(n,6,4)),old_prior=a,new_prior=b,legacy=np.einsum('nhu,nu->nh',F,a),active=np.einsum('nhu,nu->nh',F,b),context=np.array([0.,1.,1.1,2.,9.]))
    def test_simplex_matches_independent_constrained_optimizer(self):
        rng=np.random.default_rng(8);E=rng.normal(0,.02,(12,6,4));prior=rng.dirichlet(np.ones(4),12)
        for strength in [.01,1,10]:
            w=simplex_fit(E,prior,strength)
            for i in range(len(E)):
                lam=strength*((E[i]**2).mean()+QUANTIZATION_RAD**2/12)
                def objective(x):return np.mean((E[i]@x)**2)+lam*np.sum((x-prior[i])**2)
                result=minimize(objective,prior[i],method='SLSQP',bounds=[(0,1)]*4,constraints=[dict(type='eq',fun=lambda x:x.sum()-1)],options=dict(ftol=1e-14,maxiter=400));self.assertTrue(result.success);self.assertLessEqual(objective(w[i]),objective(result.x)+1e-12);np.testing.assert_allclose(w[i],result.x,atol=2e-5)
    def test_uninformative_errors_retain_prior(self):
        prior=np.array([[.1,.2,.3,.4],[1.,0,0,0]])
        for E in [np.zeros((2,6,4)),np.full((2,6,4),.2)]:np.testing.assert_allclose(simplex_fit(E,prior,.1),prior,atol=1e-10)
    def test_guarded_methods_preserve_quiet_bit_for_bit(self):
        e=self.evidence()
        for method in ['legacy','hard_context','ramp_context','guarded_inverse','guarded_horizon_inverse','guarded_winner','guarded_simplex_0.01','guarded_simplex_1','guarded_simplex_10']:
            y,w=fuse(e,method);np.testing.assert_array_equal(y[:2],e['legacy'][:2]);np.testing.assert_allclose(w.sum(2),1,atol=1e-12);self.assertTrue((w>=0).all());self.assertTrue((y>=e['forecasts'].min(2)-1e-12).all());self.assertTrue((y<=e['forecasts'].max(2)+1e-12).all())
    def test_hard_router_equals_selected_parents(self):
        e=self.evidence();y,_=fuse(e,'hard_context');np.testing.assert_array_equal(y[:2],e['legacy'][:2]);np.testing.assert_allclose(y[2:],e['active'][2:],atol=1e-14)
    def test_simplex_is_equivariant_to_expert_permutation(self):
        e=self.evidence();ids=[3,0,2,1];a=simplex_fit(e['errors'],e['new_prior'],.1);b=simplex_fit(e['errors'][:,:,ids],e['new_prior'][:,ids],.1);np.testing.assert_allclose(a[:,ids],b,atol=1e-11)
    def test_context_uses_full_raw_past_including_unsampled_spike(self):
        x=np.zeros((2,401));x[1,1]=.003;np.testing.assert_array_equal(context(dict(x=x),.002),[0,1.5])
        with self.assertRaises(ValueError):context(dict(x=x[:,:81]),.002)
    def test_target_and_invalid_evidence_are_rejected(self):
        e=self.evidence();e['target']=np.zeros((5,7))
        with self.assertRaises(ValueError):fuse(e,'hard_context')
        del e['target'];e['errors'][0,0,0]=np.nan
        with self.assertRaises(ValueError):fuse(e,'guarded_inverse')
    def test_api_rejects_future_fields_bad_shapes_and_lengths(self):
        pilot=object.__new__(FrozenRegimePilot);pilot.manifest=dict(cases={'trolley_theta':{}});history={k:np.zeros((2,401)) for k in ['theta','L','x','xr']};history['L'][:]=.4
        for patch in [dict(target=np.zeros((2,7))),dict(theta=np.zeros((2,81))),dict(L=np.zeros((2,401))),dict(x=np.full((2,401),np.nan))]:
            with self.assertRaises(ValueError):pilot.forecast(dict(history,**patch),family='trolley',axis='theta')

if __name__=='__main__':unittest.main()
