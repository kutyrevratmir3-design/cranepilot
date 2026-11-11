import unittest,numpy as np
from scripts.modal_reference_challenge import statistics,fit,predict,features

class ReferenceChallengeTests(unittest.TestCase):
    def test_offset_invariance_and_no_target_access(self):
        rng=np.random.default_rng(771);n=180;p=dict(theta=rng.normal(0,.02,(n,81)),omega=rng.uniform(3,8,n),length=np.full((n,81),.4),k=np.arange(n),scenario=np.arange(n)%3)
        cfg=dict(horizons_raw=[10,20,35,50,75,100,150]);p['target']=np.repeat(p['theta'][:,:1],7,axis=1)+p['theta'][:,1:8]*.1
        for representation in ['difference','difference_lpv']:
            m=fit(statistics(p,cfg,representation,'direct'),.0001);a=predict(m,p,cfg);q={k:v.copy() for k,v in p.items()};q['theta']+=.12;q['target'][:]=999.
            np.testing.assert_allclose(predict(m,q,cfg),a+.12,atol=1e-10)
    def test_absolute_drive_levels_are_not_features(self):
        p=dict(theta=np.zeros((2,81)),omega=np.ones(2)*5,length=np.ones((2,81))*.4,x=np.ones((2,81))*999)
        for representation in ['theta_only','angle_lpv','difference','difference_lpv']:
            a=features(p,representation);p['x']*=-1;np.testing.assert_array_equal(a,features(p,representation))

if __name__=='__main__':unittest.main()
