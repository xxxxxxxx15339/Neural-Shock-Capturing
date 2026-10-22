"""Independent, predeclared long-time benchmarks with explicit reference uncertainty.

No final case is used for tuning. Numerical references use a spatial ladder and
dt/2 check. A reference comparison remains unresolved unless its Cauchy changes
are small relative to the measured method margin.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from .artifacts import create_output,finish_manifest,provenance,write_json
from .diagnostics import diagnostic_rollout
from .euler_jax import euler_split_flux_stencils, spds_jax
from .evaluation import errors
from .gated_teno5_jax import gated_reconstruction_diagnostics
from .model_io import REPO_ROOT,load_checkpoint,sha256_file
from .numerics_jax import periodic_stencils
from .reconstruction import METHODS,make_reconstruction


def exact_module():
    path=REPO_ROOT/'data/nnteno_v2/build_dataset.py'
    spec=importlib.util.spec_from_file_location('v2_exact_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_manifest(manifest):
    cases=manifest['cases']
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('Case IDs must be unique')
    for case in cases:
        if case['kind'] not in ('advection','riemann','burgers-shock','burgers-interaction','shu-osher','blast'):
            raise ValueError('Unknown benchmark kind')
        if not np.isfinite(case['time']) or case['time']<=0:raise ValueError('Positive finite final time required')
        if any(not isinstance(n,int) or n<8 for n in case['grids']):raise ValueError('Invalid grids')
        if case['kind'] not in ('advection','riemann') and not case.get('stability_only') and not case.get('reference_from'):
            ladder=case['reference_grids']
            if len(ladder)<3 or ladder!=sorted(set(ladder)) or any(r%n for r in ladder for n in case['grids']):
                raise ValueError('Reference ladder must have three increasing grids divisible by every target grid')
    return manifest


def state(case,n,time=0.):
    ex=exact_module();kind=case['kind']
    if kind=='advection':
        return ex.exact_square(n,.17,.56,1.1,-.1,.83*time)
    if kind=='riemann':
        return ex.ExactEuler(case['left'],case['right'],case['jump']).averages(n,time)
    if time!=0:
        raise ValueError('This problem has no supplied analytic reference')
    if kind=='burgers-shock':
        return ex.exact_fourier(n,.45,[[.22,1,.37]],0,1.)
    if kind=='burgers-interaction':
        return ex.exact_square(n,.2,.4,1.,.1,0.)
    if kind=='shu-osher':
        # x in [0,10], shock at x=1; equivalent conventional x in [-5,5].
        def func(q):
            x=10*q-5;left=x < -4
            rho=np.where(left,3.857143,1.+.2*np.sin(5*x));vel=np.where(left,2.629369,0.);p=np.where(left,10.33333,1.)
            return np.stack((rho,rho*vel,p/.4+.5*rho*vel**2),axis=-1)
        return ex.average_samples(func,n,breaks=(.1,))
    if kind=='blast':
        rho=np.ones(n);mom=np.zeros(n)
        # Exact cell averages of the piecewise-constant pressure.
        edges=np.arange(n+1)/n
        left=np.maximum(0,np.minimum(edges[1:],.1)-edges[:-1])*n
        right=np.maximum(0,edges[1:]-np.maximum(edges[:-1],.9))*n
        pressure=.01+(1000.-.01)*left+(100.-.01)*right
        return np.column_stack((rho,mom,pressure/.4))
    raise ValueError(kind)


def settings(case,n,refinement=1):
    u=state(case,n);system=case['system'];length=case.get('length',1.)
    speed=float(np.max(np.abs(spds_jax(jnp.asarray(u))))) if system=='euler' else .83 if system=='advection' else float(np.max(np.abs(u)))
    # Fixed initial speed margin; actual stage CFL is measured, not assumed.
    count=math.ceil(case['time']*speed*case.get('speed_safety',1.5)/(case.get('cfl',.25)*length/n))*refinement
    return dict(dt=case['time']/count,dx=length/n,steps=count,system=system,
                boundary=case['boundary'],wave_speed=.83 if system=='advection' else 1.,
                alpha='dynamic' if system=='burgers' else .83)


def run_one(case,n,rec,refinement=1,states=False,repair_policy='none'):
    opts=settings(case,n,refinement)
    return diagnostic_rollout(state(case,n),rec,**opts,record_states=states,repair_policy=repair_policy)


def diagnostics_for_states(states,model,case,strength,mode):
    @jax.jit
    def observe(u):
        if case['system']=='euler':
            # Wall case diagnostics require reflected ghosts; use matching local inputs.
            if case['boundary'] in ('reflective','reflective-wall'):
                sign=jnp.array([1.,-1.,1.]);padded=jnp.concatenate((jnp.flip(u[:3],axis=0)*sign,u,jnp.flip(u[-3:],axis=0)*sign))
                p,m=euler_split_flux_stencils(padded,'transmissive');p=p[3:3+len(u)];m=m[3:3+len(u)]
            else:p,m=euler_split_flux_stencils(u,case['boundary'])
            inputs=jnp.stack((p,m),axis=1)
        else:
            plus=periodic_stencils(u)
            minus=jnp.roll(jnp.flip(periodic_stencils(jnp.flip(u)),axis=0),-1,axis=0)
            inputs=jnp.stack((plus,minus),axis=1)
        return gated_reconstruction_diagnostics(model.backbone,inputs,mode,strength)
    observations=[jax.tree.map(np.asarray,observe(jnp.asarray(u))) for u in states]
    return {key:np.stack([o[key] for o in observations]) for key in observations[0]}


def reference(case,n,output):
    if case.get('stability_only'):
        return None,dict(kind='stability-only; no accuracy reference',status='not computed after failed short-time neural smoke',
                         trustworthy_for_win_claim=False,reason=case.get('reference_limit')),{}
    if case['kind'] in ('advection','riemann'):
        return state(case,n,case['time']),dict(kind='analytic cell averages',certified_for_defined_pde=True),{}
    if case.get('reference_from'):
        source=REPO_ROOT/case['reference_from'].format(n=n)
        info=json.loads((source/'reference.json').read_text())
        info=dict(info,reused_from=str(source.relative_to(REPO_ROOT)),
                  source_metadata_sha256=sha256_file(source/'reference.json'),
                  reuse_caveat='Classical ghost-reflection references have zero wall mass/energy flux to roundoff. Explicit neural wall closure is a distinct coarse-solver variant; retain numerical reference uncertainty.')
        if not (source/'reference.npz').exists():return None,info,{}
        arrays=dict(np.load(source/'reference.npz'))
        info['source_arrays_sha256']=sha256_file(source/'reference.npz')
        return arrays.pop('state'),info,arrays
    rec=make_reconstruction(None,'teno5');ladder=case['reference_grids'];projections=[];records=[];arrays={}
    cache=output.parent/'reference-cache'/case['id'];cache.mkdir(parents=True,exist_ok=True)
    identity={k:v for k,v in case.items() if k!='grids'}
    metadata=cache/'configuration.json'
    if metadata.exists():
        if json.loads(metadata.read_text())!=identity:raise ValueError('Reference cache case configuration mismatch')
    else:write_json(metadata,identity)

    def saved_reference(cells,refinement):
        name=f'N{cells}-dt{refinement}'
        data=cache/f'{name}.npz';meta=cache/f'{name}.json'
        if data.exists() and meta.exists():
            return np.load(data)['state'],json.loads(meta.read_text())
        final,diag,_=run_one(case,cells,rec,refinement)
        np.savez_compressed(data,state=final)
        write_json(meta,diag)
        return final,diag
    for k,cells in enumerate(ladder):
        final,diag=saved_reference(cells,1)
        records.append(dict(cells=cells,refinement=1,diagnostics=diag))
        if not diag['valid_for_accuracy_claim']:
            return None,dict(kind='numerical',status='reference-failed',records=records,trustworthy_for_win_claim=False),arrays
        projection=final.reshape(n,cells//n,*final.shape[1:]).mean(axis=1)
        projections.append(projection);arrays[f'reference_{cells}']=final
        print(f'  reference N={cells} admissible',flush=True)
    fine,diag=saved_reference(ladder[-1],2)
    if not diag['valid_for_accuracy_claim']:
        return None,dict(kind='numerical',status='temporal-reference-failed',records=records,diagnostics=diag,trustworthy_for_win_claim=False),arrays
    target=fine.reshape(n,ladder[-1]//n,*fine.shape[1:]).mean(axis=1)
    spatial_changes=[errors(a,b)['l1'] for a,b in zip(projections,projections[1:])]
    temporal=errors(target,projections[-1])['l1']
    arrays['reference_temporal_refined']=fine
    return target,dict(kind='numerical cell-average projection',scheme='historical TENO5 split-flux/state LF + SSPRK3',
              status='Cauchy changes measured; no rigorous bound',grids=ladder,
              spatial_l1_changes=spatial_changes,temporal_l1_change=temporal,
              uncertainty_proxy=2*(spatial_changes[-1]+temporal),records=records,
              temporal_fine_diagnostics=diag,trustworthy_for_win_claim=False),arrays


def benchmark(config,manifest,output):
    selected=json.loads((REPO_ROOT/'configs/v2/selected.json').read_text())
    models={p:load_checkpoint(v['path'],v['sha256']) for p,v in config['models'].items()}
    p0=json.loads((REPO_ROOT/'src/differentiable/benchmarks.json').read_text())
    historical={p:load_checkpoint(v['path'],v['sha256']) for p,v in p0['models'].items()}
    override=manifest.get('method_override')
    if override:
        chosen=historical if override['checkpoint_source']=='p0' else models
        for model in chosen.values():
            for method in override['methods']:make_reconstruction(model,method,override['strength'])
    records=[]
    for case in manifest['cases']:
        for n in case['grids']:
            dest=output/f'{case["id"]}-{n}';dest.mkdir()
            print(f'{case["id"]} N={n}',flush=True)
            ref,refinfo,refarrays=reference(case,n,dest)
            write_json(dest/'reference.json',refinfo)
            if ref is not None:np.savez_compressed(dest/'reference.npz',state=ref,**refarrays)
            physics='euler' if case['system']=='euler' else 'scalar';model=models[physics]
            specs=[('teno5','teno5',0.,model),('v2-selected-soft','gated-soft',selected[physics]['strength'],model),
                   ('v2-trained-soft','gated-soft',.25 if physics=='euler' else .75,model),
                   ('v2-hard','gated-hard',.25,model),('v2-neural-only','neural-soft',1.,model),
                   ('p0-historical-learned','learned-blend',p0['fixed_lambda'],historical[physics])]
            if override:
                methods=list(dict.fromkeys(['teno5',*override['methods']]))
                chosen=historical[physics] if override['checkpoint_source']=='p0' else model
                specs=[(method,method,override['strength'],chosen) for method in methods]
            baseline=None;simulations={}
            for label,policy,strength,checkpoint in specs:
                key=(policy,strength,checkpoint.sha256)
                if key not in simulations:
                    simulations[key]=run_one(case,n,make_reconstruction(checkpoint,policy,strength),states=True)
                final,diag,history=simulations[key]
                keep=np.unique(np.linspace(0,len(history['state'])-1,min(21,len(history['state'])),dtype=int))
                snapshots=np.concatenate((state(case,n)[None],history['state'][keep]))
                times=np.concatenate(([0.],(keep+1)*settings(case,n)['dt']))
                tv=np.sum(np.abs(np.diff(final,axis=0)),axis=0) if case['boundary']!='periodic' else np.sum(np.abs(final-np.roll(final,1,axis=0)),axis=0)
                row=dict(case=case['id'],kind=case['kind'],physics=physics,cells=n,method=label,
                         policy=policy,strength=strength,checkpoint_sha256=checkpoint.sha256,
                         time=case['time'],sampling='finite-volume cell averages',diagnostics=diag,
                         total_variation=np.atleast_1d(tv).tolist(),
                         minimum=np.atleast_1d(np.min(final,axis=0)).tolist(),maximum=np.atleast_1d(np.max(final,axis=0)).tolist(),
                         reference_status=refinfo['kind'])
                arrays=dict(state=final,initial=state(case,n),snapshots=snapshots,times=times)
                if ref is not None:
                    row.update(errors(final,ref));row['overshoot']=np.maximum(np.max(final,axis=0)-np.max(ref,axis=0),0).tolist()
                    row['undershoot']=np.maximum(np.min(ref,axis=0)-np.min(final,axis=0),0).tolist()
                    if label=='teno5':baseline=row
                    margin=(baseline['l1']-row['l1']) if baseline else 0.
                    row['relative_l1']=row['l1']/baseline['l1'] if baseline and baseline['l1']>1e-10 else None
                    row['ranking_resolved']=(refinfo.get('certified_for_defined_pde',False) or refinfo.get('uncertainty_proxy',np.inf)<.25*abs(margin)) and diag['valid_for_accuracy_claim'] and baseline['diagnostics']['valid_for_accuracy_claim']
                    if case['kind'] in ('advection','riemann'):
                        arrays['error_l1_over_time']=np.array([errors(u,state(case,n,t))['l1'] for u,t in zip(snapshots,times)])
                if label=='v2-selected-soft' and diag['finite']:
                    observations=diagnostics_for_states(snapshots,model,case,strength,'soft')
                    arrays.update({'neural_'+k:v for k,v in observations.items()})
                    row['gate_fraction']=float(observations['gate'].mean())
                    row['correction_mean_absolute']=float(np.mean(np.abs(observations['correction'])))
                write_json(dest/f'{label}.json',row);np.savez_compressed(dest/f'{label}.npz',**arrays)
                records.append(row)
                print(f'  {label}: L1={row.get("l1")} valid={diag["valid_for_accuracy_claim"]}',flush=True)
            jax.clear_caches()
    write_json(output/'benchmark.json',dict(manifest=manifest,selected=selected,results=records))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=REPO_ROOT/'configs/v2/independent.json')
    p.add_argument('--config',type=Path,default=REPO_ROOT/'configs/v2/full.json');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=METHODS,action='append',help='Explicit policy override; repeat for matched comparisons (TENO5 always included)')
    p.add_argument('--strength',type=float,default=.25,help='Fixed strength for explicit --method overrides')
    p.add_argument('--checkpoint-source',choices=['v2','p0'],default='v2',help='Keep model generation explicit; P0 heads are historical, not V2-trained heads')
    args=p.parse_args(argv);config=json.loads(args.config.read_text());manifest=validate_manifest(json.loads(args.manifest.read_text()))
    if not np.isfinite(args.strength) or not 0<=args.strength<=1:p.error('Strength must be finite and in [0,1]')
    if args.method:manifest=dict(manifest,method_override=dict(methods=args.method,strength=args.strength,checkpoint_source=args.checkpoint_source))
    out=create_output(args.output);write_json(out/'configuration.json',dict(config=config,manifest=manifest))
    write_json(out/'provenance.json',provenance(args.manifest));benchmark(config,manifest,out);finish_manifest(out)


if __name__=='__main__':main()
