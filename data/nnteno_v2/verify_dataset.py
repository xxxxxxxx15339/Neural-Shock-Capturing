#!/usr/bin/env python3
"""Independently scan every released file and verify hashes, integrity and physics."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import h5py

def run(path):
    root=Path(path)
    audit=json.loads((root/'quality_report.json').read_text())
    entries=[json.loads(s) for s in (root/'cases.jsonl').read_text().splitlines()]
    assert len(entries)==sum(audit['counts'].values())
    assert audit['manifest_sha256']==hashlib.sha256((root/'cases.jsonl').read_bytes()).hexdigest()
    lookup={r['id']:r for r in entries}
    assert len(lookup)==len(entries)
    seen=set(); counted={}
    max_residual={'scalar':0.,'euler':0.};max_short_cfl=0.; total_cells=0
    for physics in ('scalar','euler'):
      for split in ('train','validation','test'):
        name=f'{physics}_{split}.h5';file=root/name
        expected=audit['files'][name]
        assert file.stat().st_size==expected['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==expected['sha256']
        count=0
        with h5py.File(file,'r') as h5:
          for grid in h5:
            group=h5[grid];n=int(grid[1:])
            u=group['states'][:];t=group['times'][:];sh=group['short_target'][:]; adv_speed=group['advection_speed'][:]
            short_t=group['short_time'][:];dt=group['short_dt'][:]
            ids=group['case_id'][:];families=group['family'][:]
            assert group.attrs['short_steps']==15
            assert u.shape==(len(ids),4,n,3) if physics=='euler' else u.shape==(len(ids),4,n)
            assert sh.shape==u[:,0].shape
            assert t.shape==(len(ids),4)
            assert np.isfinite(u).all() and np.isfinite(sh).all()
            assert np.all(np.diff(t,axis=-1)>0)
            assert np.allclose(short_t,15*dt,rtol=0,atol=1e-15)
            assert np.all(short_t>0) and np.all(short_t<t[:,-1])
            total_cells+=len(ids)*4*n
            for j,encoded in enumerate(ids):
                id=encoded.decode();family=families[j].decode()
                assert id not in seen;seen.add(id)
                m=lookup[id]
                assert m['split']==split and m['physics']==physics and m['family']==family and m['cells']==n
                assert np.isclose(adv_speed[j],m['params'].get('speed',1.0) if physics=='scalar' and family in ('constant','square','gaussian','fourier') else 0.0)
                assert abs(m['short_dt']-dt[j])<1e-15
                bound=m['characteristic_speed_bound'];cfl=dt[j]*bound*n
                max_short_cfl=max(max_short_cfl,cfl)
                assert cfl<=.35+1e-12
                if physics=='scalar':
                    residual=float(np.max(np.abs(np.mean(u[j],axis=1)-np.mean(u[j,0]))))
                    rshort=float(abs(np.mean(sh[j])-np.mean(u[j,0])))
                    residual=max(residual,rshort)
                else:
                    rho=u[j,...,0];v=u[j,...,1]/rho;p=.4*(u[j,...,2]-.5*rho*v*v)
                    sr=sh[j,:,0];sv=sh[j,:,1]/sr;sp=.4*(sh[j,:,2]-.5*sr*sv*sv)
                    assert np.min(rho)>0 and np.min(p)>0 and np.min(sr)>0 and np.min(sp)>0
                    fl=np.array(m['params']['initial_left_flux']);fr=np.array(m['params']['initial_right_flux'])
                    mass=u[j].mean(axis=1)
                    residual=float(np.max(abs(mass-(mass[0]-t[j,:,None]*(fr-fl)))))
                    rshort=float(np.max(abs(sh[j].mean(axis=0)-(mass[0]-short_t[j]*(fr-fl)))))
                    residual=max(residual,rshort)
                max_residual[physics]=max(max_residual[physics],residual)
            count+=len(ids)
        assert count==audit['counts'][f'{physics}_{split}']
        counted[f'{physics}_{split}']=count
    assert len(seen)==len(entries) and max_short_cfl<=.35000001
    assert max_residual['scalar']<1e-9 and max_residual['euler']<1e-8
    result={'status':'PASS','trajectories':len(entries),'exact_reference_snapshots':len(entries)*4,'total_snapshot_cell_locations':total_cells,
        'counts':counted,'max_scalar_conservation_residual':max_residual['scalar'],'max_euler_conservation_residual':max_residual['euler'],
        'max_estimated_short_horizon_CFL':max_short_cfl,'dataset_source':'analytic mathematical exact solutions','physical_observations_included':False}
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/nnteno_v2');a=p.parse_args()
    run(a.root)
