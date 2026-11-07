import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy.linalg import expm
from scripts.external_crane_data import config,pixels_to_angles,packets,train_offset,load_sequence,blob_hash,verified_path
from scripts.external_baselines import geometric,oscillator,fit_oscillator,fit_arx,predict_arx,save_arx,load_arx,ResidualLSTM,lstm_inputs,predict_lstm,FrozenAngular,radius

class ExternalBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg=config();torch.set_num_threads(1)
    def sequence(self,n=80,sid=1):
        t=np.arange(n)*.05
        return dict(sid=sid,time=t,theta=.03*np.sin(3.13*t),radial=np.zeros(n),boom=.1*np.sin(.7*t),velocity=.07*np.cos(.7*t),acceleration=-.049*np.sin(.7*t))
    def packet(self):return packets(self.sequence(),self.cfg,0,2)
    def test_camera_geometry_matches_scalar_transform(self):
        c=self.cfg;A=np.array([[.99,-.01,.02],[.02,.98,.03],[-.01,.02,1.]]);cam=np.array([1.4,.2,2.4]);u=np.array([600.,650.]);v=np.array([400.,410.]);b=np.array([.1,.4]);out=pixels_to_angles(u,v,b,A,cam,c)
        K=np.array([[779.1104169208932,0,635.4402695731219],[0,778.6639838714995,374.9313955118369],[0,0,1.]])
        for i in range(2):
            R=np.array([[np.cos(b[i]),-np.sin(b[i]),0],[np.sin(b[i]),np.cos(b[i]),0],[0,0,1.]])
            world=A@R.T@((cam[2]-1)*np.linalg.solve(K,[u[i],v[i],1]))+R@cam
            answer=1.7*(R.T@(world-radius(c)*np.array([np.cos(b[i]),np.sin(b[i]),0])))[:2]
            np.testing.assert_allclose(out[i],answer,rtol=0,atol=1e-14)
    def test_partitions_sealed_and_git_blob_integrity(self):
        with self.assertRaises(PermissionError):load_sequence(self.cfg,17,self.cfg['train_ids'])
        self.assertFalse(set(self.cfg['train_ids'])&set(self.cfg['test_ids']))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'data.mat';p.write_bytes(b'abc');cfg=dict(cache=td,files=[dict(path='data.mat',bytes=3,git_blob_sha1=blob_hash(b'abc'))])
            self.assertEqual(verified_path(cfg,'data.mat'),p);p.write_bytes(b'abd')
            with self.assertRaises(ValueError):verified_path(cfg,'data.mat')
    def test_raw_prefix_causality_and_filtered_fields_ignored(self):
        seq=self.sequence();n=len(seq['theta']);cells=np.empty(n,dtype=object)
        for j in range(n):cells[j]=np.array([str(np.datetime64('2026-01-01T00:00:00.000')+np.timedelta64(j*50,'ms')),seq['velocity'][j],seq['acceleration'][j],seq['boom'][j],600.+j,400.])
        inp=dict(velocity=seq['velocity'],acceleration=seq['acceleration'],yd=np.full((n,4),999.));result=dict(raw_vision_data=cells,vision_data='ignored');cal=dict(A_WP=np.eye(3),W_r_OP=np.array([1.,0.,2.]))
        def fake(path,**kw):return inp if path=='input' else result if path=='results' else cal
        def path(cfg,name):return 'input' if 'input_seq' in name else 'results' if 'results_seq' in name else 'camera'
        with patch('scripts.external_crane_data.verified_path',path),patch('scripts.external_crane_data.loadmat',fake):
            before=load_sequence(self.cfg,1,[1]);cells[40:]=[np.array([*cell[:4],-999.,999.],dtype=object) for cell in cells[40:]];after=load_sequence(self.cfg,1,[1])
        np.testing.assert_array_equal(before['theta'][:40],after['theta'][:40]);self.assertFalse(np.array_equal(before['theta'][40:],after['theta'][40:]))
    def test_packet_prefix_and_index_contract(self):
        s=self.sequence();a=packets(s,self.cfg,.001,2);t=copy.deepcopy(s);t['theta'][40:]=999.;b=packets(t,self.cfg,.001,2)
        np.testing.assert_array_equal(a['theta'][a['k']<40],b['theta'][b['k']<40]);self.assertEqual(a['theta'].shape[1],9);self.assertEqual(a['target'].shape[1],30)
        np.testing.assert_allclose(a['actual_horizon_s'][0],np.arange(1,31)*.05,atol=1e-14)
        self.assertAlmostEqual(train_offset([dict(theta=np.ones(3)),dict(theta=np.ones(30)*3)]),2.)
    def test_known_damped_system_profile_and_parameter_recovery(self):
        cfg=self.cfg;rng=np.random.default_rng(53);n=16;omega=3.;damping=.12;gain=1.1;A=np.array([[0,1],[-omega**2,-damping]]);T=expm(A*.05);B=np.linalg.solve(A,T-np.eye(2))@np.array([0,-gain]);state=rng.normal(0,.02,(n,2));u=rng.normal(0,.12,(n,38));ys=[state[:,0].copy()]
        for a in u.T:state=state@T.T+a[:,None]*B;ys.append(state[:,0].copy())
        y=np.stack(ys,1);p=dict(theta=y[:,:9][:,::-1].copy(),past_acceleration=np.column_stack([u[:,:8],np.zeros(n)])[:,::-1]/radius(cfg),future_acceleration=u[:,8:]/radius(cfg),target=y[:,9:],scenario=np.repeat([1,2],8))
        np.testing.assert_allclose(oscillator(p,cfg,[omega,damping,gain]),p['target'],atol=1e-8)
        parameters,_=fit_oscillator(p,cfg);np.testing.assert_allclose(parameters,[omega,damping,gain],atol=2e-5)
    def test_arx_lstm_command_prefix_and_zero_initial_fallback(self):
        p=self.packet();models=fit_arx(p,self.cfg,.01);a=predict_arx(models,p,self.cfg);changed=copy.deepcopy(p);changed['future_acceleration'][:,10:]=7;changed['future_velocity'][:,10:]=-7
        np.testing.assert_allclose(predict_arx(models,changed,self.cfg)[:,:10],a[:,:10],rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'a.npz';save_arx(path,models);np.testing.assert_array_equal(predict_arx(load_arx(path),p,self.cfg),a)
        m=ResidualLSTM(32);np.testing.assert_allclose(predict_lstm(m,p,self.cfg),geometric(p,self.cfg),atol=1e-7)
        torch.nn.init.normal_(m.output.weight,std=.01)
        before=predict_lstm(m,p,self.cfg);after=predict_lstm(m,changed,self.cfg);np.testing.assert_array_equal(before[:,:10],after[:,:10])
    def test_frozen_angular_uses_no_targets_and_preserves_weights(self):
        p=self.packet();m=FrozenAngular('retained');states=[{k:v.clone() for k,v in model.state_dict().items()} for model in m.bundle.models]
        a=m.forecast(p,self.cfg);poison=copy.deepcopy(p);poison['target'][:]=999.;poison['actual_horizon_s'][:]=999.;np.testing.assert_array_equal(a,m.forecast(poison,self.cfg))
        for state,model in zip(states,m.bundle.models):
            for key,value in state.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        lag=(np.arange(41)+4)//5;np.testing.assert_array_equal(lag[:7],[0,1,1,1,1,1,2])

if __name__=='__main__':unittest.main()
