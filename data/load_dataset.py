#!/usr/bin/env python3
"""Read NN-TENO5 V2 trajectory corpus without mixing PDE physics in batches.

Example:
  from load_dataset import load_bucket, iter_batches
  bucket = load_bucket('data/nnteno_v2', 'scalar','train',128)
  for batch in iter_batches(bucket, batch_size=16, seed=1, family='burgers_smooth'):
      # batch['initial'], batch['target']: (B,N) or (B,N,3)
      # batch['dt'], batch['advection_speed']: (B,)
      # per-example time step / speed must be handled inside jax.vmap
      ...
"""
from pathlib import Path
import numpy as np
import h5py

SCALAR_ADVECTION = ('constant','square','fourier','gaussian')
SCALAR_BURGERS = ('burgers_smooth','burgers_top_hat')

def load_bucket(root, physics='scalar', split='train', cells=128):
    root=Path(root)
    if physics not in ('scalar','euler') or split not in ('train','validation','test'):
        raise ValueError('Invalid physics or split')
    with h5py.File(root/f'{physics}_{split}.h5','r') as f:
        group=f[f'N{cells:03d}']
        data={name:group[name][()] for name in ['states','times','short_target','short_time','short_dt','advection_speed','case_id','family']}
        data['short_steps']=int(group.attrs['short_steps'])
    data['initial']=data['states'][:,0]
    data['target']=data['short_target']
    data['dt']=data['short_dt']
    data['family']=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in data['family']])
    data['case_id']=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in data['case_id']])
    data['physics']=physics;data['cells']=cells;data['split']=split
    return data

def iter_batches(bucket,batch_size=16,seed=0,family=None,shuffle=True):
    if family is None and bucket['physics']=='scalar':
        raise ValueError('Specify scalar family or group, because advection and Burgers need different physical fluxes')
    if family=='advection':selected=np.isin(bucket['family'],SCALAR_ADVECTION)
    elif family=='burgers':selected=np.isin(bucket['family'],SCALAR_BURGERS)
    elif family is None:selected=np.ones(len(bucket['family']),dtype=bool)
    else:selected=bucket['family']==family
    indices=np.flatnonzero(selected)
    if shuffle:np.random.default_rng(seed).shuffle(indices)
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        yield {'initial':bucket['initial'][ix], 'target':bucket['target'][ix],
               'dt':bucket['dt'][ix], 'case_id':bucket['case_id'][ix],
               'family':bucket['family'][ix], 'advection_speed':bucket['advection_speed'][ix], 'steps':bucket['short_steps']}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/nnteno_v2');p.add_argument('--physics',default='scalar');p.add_argument('--split',default='train');p.add_argument('--cells',type=int,default=128)
    a=p.parse_args()
    b=load_bucket(a.root,a.physics,a.split,a.cells)
    print('Loaded',len(b['initial']),'cases at N=',a.cells, 'shape=',b['initial'].shape)
    print('Families:',{z:int(np.sum(b['family']==z)) for z in np.unique(b['family'])})
    print('Reference substep dt:',float(np.min(b['dt'])),float(np.max(b['dt'])))
