"""Pinned raw-camera data and causal sample-index packets from a second crane.

Original MATLAB code is inspected, never executed. No author-filtered angles or
simulated input-file yd targets enter this pipeline. Input arrays are prescribed
profiles; encoder angle and raw pixels are observed quantities.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

CONFIG=Path('configs/external_benchmark_20261107.json')

def config():return json.loads(CONFIG.read_text())

def blob_hash(content):return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()

def verified_path(cfg,name):
    spec=next(x for x in cfg['files'] if x['path']==name);p=Path(cfg['cache'])/name
    b=p.read_bytes()
    if len(b)!=spec['bytes'] or blob_hash(b)!=spec['git_blob_sha1']:raise ValueError(f'Changed external data {name}')
    return p

def pixels_to_angles(u,v,boom,A,camera,cfg):
    u,v,boom=[np.asarray(x,float) for x in (u,v,boom)]
    intrinsic=np.array([[779.1104169208932,0,635.4402695731219],[0,778.6639838714995,374.9313955118369],[0,0,1.]])
    ray=np.column_stack([u,v,np.ones_like(u)])@np.linalg.inv(intrinsic).T
    ray*=camera[2]-cfg['rope_length_m'];c=np.cos(boom);s=np.sin(boom)
    rotated=np.column_stack([c*ray[:,0]+s*ray[:,1],-s*ray[:,0]+c*ray[:,1],ray[:,2]])
    world=rotated@A.T+np.column_stack([c*camera[0]-s*camera[1],s*camera[0]+c*camera[1],np.full_like(c,camera[2])])
    radius=cfg['boom_tip_height_m']*np.tan(cfg['boom_elevation_rad'])
    dx=world[:,0]-radius*c;dy=world[:,1]-radius*s
    return cfg['pixel_scale']*np.column_stack([c*dx+s*dy,-s*dx+c*dy])/cfg['rope_length_m']


def load_sequence(cfg,sid,allowed_ids):
    if sid not in allowed_ids:raise PermissionError('Partition is sealed for this stage')
    inp=loadmat(verified_path(cfg,f'data/validation_data/input_seq_sim_500_{sid}.mat'),simplify_cells=True)
    result=loadmat(verified_path(cfg,f'data/validation_data/results_seq_sim_500_{sid}.mat'),simplify_cells=True)
    cal=loadmat(verified_path(cfg,'data/camera_coordinates.mat'),simplify_cells=True)
    raw=result['raw_vision_data'];numeric=np.array([list(row[1:]) for row in raw],dtype=float)
    time=np.array([np.datetime64(str(row[0]),'ms') for row in raw]);time=(time-time[0])/np.timedelta64(1,'s')
    if np.any(np.diff(time)<=0):raise ValueError('Non-increasing camera timestamps')
    boom=np.unwrap(numeric[:,2]);angles=pixels_to_angles(numeric[:,3],numeric[:,4],boom,cal['A_WP'],cal['W_r_OP'],cfg)
    velocity=np.asarray(inp['velocity'],float);acceleration=np.asarray(inp['acceleration'],float)
    if not(len(angles)==len(velocity)==len(acceleration)):raise ValueError('Unaligned source lengths')
    # Source raw cells carry exactly the issued profiles in columns 2/3, not an independent velocity sensor.
    np.testing.assert_allclose(numeric[:,:2],np.column_stack([velocity,acceleration]),rtol=0,atol=1e-12)
    if not all(np.isfinite(a).all() for a in [angles,boom,velocity,acceleration]):raise ValueError('Nonfinite raw observations; no posthoc imputation')
    return dict(sid=sid,time=np.asarray(time,float),theta=angles[:,1],radial=angles[:,0],boom=boom,velocity=velocity,acceleration=acceleration)


def packets(sequence,cfg,offset,stride):
    n=len(sequence['theta']);k=np.arange(cfg['history_samples']-1,n-cfg['steps'],stride)
    h=k[:,None]-np.arange(cfg['history_samples']);f=k[:,None]+np.arange(cfg['steps']);target=k[:,None]+1+np.arange(cfg['steps'])
    radius=cfg['boom_tip_height_m']*np.tan(cfg['boom_elevation_rad'])
    return dict(k=k,scenario=np.full(len(k),sequence['sid']),theta=(sequence['theta'][h]-offset).copy(),
        x=(radius*sequence['boom'][h]).copy(),past_velocity=sequence['velocity'][h].copy(),past_acceleration=sequence['acceleration'][h].copy(),
        future_velocity=sequence['velocity'][f].copy(),future_acceleration=sequence['acceleration'][f].copy(),
        target=(sequence['theta'][target]-offset).copy(),actual_horizon_s=sequence['time'][target]-sequence['time'][k,None])


def concatenate(parts):return {key:np.concatenate([p[key] for p in parts],axis=0) for key in parts[0]}


def train_offset(sequences):return float(np.mean([s['theta'].mean() for s in sequences]))
