"""Predeclared V2 spatial/temporal and pretraining-distribution studies."""
import argparse
import json
import math
from pathlib import Path

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance

from .artifacts import create_output, finish_manifest, provenance, write_json
from .diagnostics import diagnostic_rollout
from .euler_jax import euler_characteristic_state_stencils, euler_split_flux_stencils, get_euler_flux_jax
from .evaluation import errors
from .gated_teno5_jax import gated_reconstruction_diagnostics
from .model_io import REPO_ROOT, load_checkpoint
from .numerics_jax import lax_friedrichs_flux
from .reconstruction import make_reconstruction
from .teno5_jax import teno5_selection_values, DEFAULT_CUTOFF
from .teno5_jax_toy import toy_keep_weight
from .v2_evaluate import load_config, load_validation


def smooth_state(n,system,time=0.,profile='regular'):
    nodes,weights=np.polynomial.legendre.leggauss(16)
    x=(np.arange(n)[:,None]+.5+.5*nodes)/n
    speed=.7 if system=='euler' else 1.
    q=x-speed*time
    values=np.exp(np.sin(2*np.pi*q)) if profile=='regular' else 1.+np.sin(2*np.pi*q)**4
    avg=.5*np.sum(values*weights,axis=1)
    if system!='euler':return avg
    rho=.5+.5*avg
    return np.column_stack((rho,speed*rho,2.5+.5*speed**2*rho))


def spatial(n,system,rec,profile):
    initial=smooth_state(n,system,profile=profile)
    faces=np.arange(n+1)/n
    values=np.exp(np.sin(2*np.pi*faces)) if profile=='regular' else 1.+np.sin(2*np.pi*faces)**4
    d=np.diff(values)*n
    if system=='euler':
        exact=np.column_stack((.35*d,.245*d,.25*.7**3*d))
        flux=get_euler_flux_jax(rec,'periodic')
    else:
        exact=d
        flux=lambda u:lax_friedrichs_flux(u,lambda v:v,1.,rec)
    numerical=np.asarray(jax.jit(flux)(jnp.asarray(initial)))*n
    return dict(cells=n,**errors(numerical,exact))


def add_orders(rows):
    for a,b in zip(rows,rows[1:]):
        for key in ('l1','l2','linf'):
            b['order_'+key]=float(np.log(a[key]/b[key])/np.log(b['cells']/a['cells'])) if a[key]>0 and b[key]>0 else None


def convergence(config,output):
    result=dict(spatial=[],rollout=[],settings=dict(resolutions=[40,80,160,320,640],profiles=['regular','critical-sin4'],
                final_time=.05,quadrature=16,time_step='.15*dx^(5/3)/speed_bound; dt/2 control',
                scalar='linear advection',euler='advected entropy wave: constant velocity/pressure',
                metric='cell-average derivative error for spatial; cell-average state error for rollout'))
    for physics,system in [('scalar','advection'),('euler','euler')]:
        model=load_checkpoint(**dict(path=config['models'][physics]['path'],expected_sha256=config['models'][physics]['sha256']))
        strength=.75 if physics=='scalar' else .25
        for method in ['teno5','gated-soft','gated-hard','neural-soft']:
            rec=make_reconstruction(model,method,strength)
            for profile in ['regular','critical-sin4']:
                rows=[spatial(n,system,rec,profile) for n in [40,80,160,320,640]]
                add_orders(rows)
                result['spatial'].append(dict(physics=physics,method=method,strength=strength,profile=profile,rows=rows))
                print(f'spatial {physics}/{method}/{profile}: last p={rows[-1]["order_l1"]:.5g}',flush=True)
                if method=='neural-soft':continue
                time_rows=[]
                for n in [40,80,160,320,640]:
                    speed=2.5 if physics=='euler' else 1.
                    steps=math.ceil(.05*speed/(.15*(1/n)**(5/3)))
                    # Temporal controls at every grid separate time and space errors.
                    finals=[];diags=[]
                    for factor in [1,2]:
                        count=steps*factor
                        final,diag,_=diagnostic_rollout(smooth_state(n,system,profile=profile),rec,.05/count,1/n,count,system,'periodic')
                        finals.append(final);diags.append(diag)
                    metric=errors(finals[1],smooth_state(n,system,.05,profile))
                    temporal=errors(finals[0],finals[1])
                    time_rows.append(dict(cells=n,steps=steps*2,dt=.05/(steps*2),**metric,
                                          temporal_cauchy_l1=temporal['l1'],diagnostics=diags[1]))
                add_orders(time_rows)
                result['rollout'].append(dict(physics=physics,method=method,strength=strength,profile=profile,rows=time_rows))
                print(f'rollout {physics}/{method}/{profile}: last p={time_rows[-1]["order_l1"]:.5g}',flush=True)
            jax.clear_caches()
    write_json(output/'convergence.json',result)


def normalize(x):
    lo=np.min(x,axis=-1,keepdims=True);r=np.ptp(x,axis=-1,keepdims=True)
    return (x-lo)/np.where(r>0,r,1.)


def input_mismatch(config,output):
    """Use training data only. Compare normalized shapes as used by the MLP."""
    import h5py
    model=load_checkpoint(config['models']['euler']['path'])
    groups=[];pools={k:[] for k in ['warmup-state','positive-flux','negative-flux']}
    rng=np.random.default_rng(config['seed'])
    with h5py.File(REPO_ROOT/config['data_root']/'euler_train.h5','r') as h5:
        for n in [64,128,256]:
            group=h5[f'N{n:03d}'];families=np.array([v.decode() for v in group['family'][:]])
            @jax.jit
            def extract(u):
                pos,neg=euler_split_flux_stencils(u)
                return euler_characteristic_state_stencils(u),pos,neg
            for family in sorted(set(families)):
                ix=np.flatnonzero(families==family)
                ix=rng.choice(ix,size=min(4,len(ix)),replace=False)
                states=np.asarray(group['states'][:])[ix]
                # Warm-up is on initial states; actual inputs include later states.
                initial=np.concatenate([np.asarray(extract(u)[0]).reshape(-1,5) for u in states[:,0]])
                actual=[extract(u) for u in states.reshape(-1,n,3)]
                pos=np.concatenate([np.asarray(a[1]).reshape(-1,5) for a in actual]);neg=np.concatenate([np.asarray(a[2]).reshape(-1,5) for a in actual])
                data={'warmup-state':initial,'positive-flux':pos,'negative-flux':neg}
                for source,array in data.items():
                    chi=np.asarray(teno5_selection_values(jnp.asarray(array)));labels=chi>=DEFAULT_CUTOFF
                    keep=np.asarray(toy_keep_weight(model.backbone,jnp.asarray(array)))
                    clipped=np.clip(keep,1e-12,1-1e-12)
                    troubled=~np.all(labels,axis=-1)
                    norm=normalize(array);warm=normalize(initial)
                    groups.append(dict(cells=n,family=family,source=source,count=len(array),
                            gate_fraction=float(troubled.mean()),constant_fraction=float(np.mean(np.ptp(array,axis=-1)<1e-12)),
                            raw_quantiles=np.quantile(array,[0,.01,.5,.99,1]).tolist(),
                            normalized_wasserstein_per_position=[float(wasserstein_distance(warm[:,k],norm[:,k])) for k in range(5)],
                            teacher_bce=float(np.mean(-labels.astype(float)*np.log(clipped)-(~labels)*np.log(1-clipped))),
                            hard_teacher_disagreement=float(np.mean((keep>.5)!=labels)),
                            troubled_teacher_disagreement=float(np.mean(((keep>.5)!=labels)[troubled])) if troubled.any() else None))
                    chosen=rng.choice(len(array),size=min(2000,len(array)),replace=False)
                    pools[source].append(norm[chosen])
            jax.clear_caches()
    write_json(output/'input_mismatch.json',dict(seed=config['seed'],split='train',
               sampling='four independent trajectories per family per grid; four stored times for solver fluxes; warm-up initial only',
               limitations='Snapshot inputs are exact-reference states, not evolving neural numerical states; marginal distances alone do not establish causal training benefit.',
               groups=groups))
    np.savez_compressed(output/'normalized_inputs.npz',**{k:np.concatenate(v) for k,v in pools.items()})


def repair_study(config,output):
    model=load_checkpoint(config['models']['euler']['path'])
    case_id='euler-validation-000088-1023461505243801251'
    rows=[]
    for bucket in load_validation(config,'euler'):
        if case_id not in bucket['ids']:continue
        j=np.flatnonzero(bucket['ids']==case_id)[0];n=bucket['n']
        for mode,strength in [('soft',.25),('soft',.75),('hard',1.)]:
            for policy in ['none','historical-repair']:
                final,diag,history=diagnostic_rollout(bucket['initial'][j],make_reconstruction(model,'gated-'+mode,strength),
                                    bucket['dt'][j],1/n,15,'euler','transmissive',repair_policy=policy)
                rows.append(dict(case=case_id,cells=n,mode=mode,strength=strength,repair_policy=policy,
                                 **errors(final,bucket['target'][j]),diagnostics=diag))
                np.savez_compressed(output/f'{mode}-{strength}-{policy}.npz',state=final,reference=bucket['target'][j],**history)
    write_json(output/'repair.json',dict(results=rows,interpretation='Optional historical training repair is not conservative; repaired runs cannot become admissible accuracy successes. Original training interventions were not recorded at every update.'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['convergence','inputs','repair'])
    p.add_argument('--config',type=Path,default=REPO_ROOT/'configs/v2/full.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);config=load_config(args.config);out=create_output(args.output)
    write_json(out/'configuration.json',config);write_json(out/'provenance.json',provenance(args.config))
    {'convergence':convergence,'inputs':input_mismatch,'repair':repair_study}[args.action](config,out)
    finish_manifest(out)


if __name__=='__main__':main()
