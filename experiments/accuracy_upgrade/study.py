"""Reproducible numerical experiments; every output path must be new."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from src.differentiable.accuracy_upgrade import (
    admissible, pressure, state_faces, flux_variant, retry_step, state_stencils)
from src.differentiable.euler_jax import flux_jax, spds_jax
from src.differentiable.teno5_jax import teno5_reconstruct_jax as teno
from src.differentiable.reconstruction import make_reconstruction
from src.differentiable.model_io import load_checkpoint
from src.differentiable.evaluation import errors
from src.differentiable.numerics_jax import rollout
from src.differentiable.v2_benchmark import exact_module, state as benchmark_state

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
VARIANTS=('historical_teno','historical_p0_learned','historical_v2_soft25',
          'fv_teno','fv_v2_soft25','safe_fv_teno','safe_fv_v2_soft25')
EXTRA_RECONSTRUCTIONS={}

def write(path,value):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    with Path(path).open('x') as f:json.dump(clean(value),f,indent=2,allow_nan=False)

@lru_cache(None)
def reconstruction(variant):
    if variant in EXTRA_RECONSTRUCTIONS:return EXTRA_RECONSTRUCTIONS[variant]
    if variant.endswith('teno'):return teno
    if 'p0' in variant:
        model=load_checkpoint('models/nn_teno5euler_unrolled_jax_params_curriculum_phase18_learnedlambda.pkl')
        return make_reconstruction(model,'learned-blend')
    return make_reconstruction(load_checkpoint('models/v2/euler_gated_dataset_v2_seed2026.pkl'),'gated-soft',.25)

def formulation(variant):return 'historical' if variant.startswith('historical') else 'fv_state_flux'

def profile(x,kind):
    rho=np.ones_like(x);v=np.full_like(x,.4);p=np.ones_like(x)
    if kind in ('density','coupled'):rho=1+.1*np.sin(2*np.pi*x)
    if kind in ('momentum','coupled'):v=.4+.1*np.cos(2*np.pi*x)
    if kind in ('pressure','coupled'):p=1+.1*np.sin(4*np.pi*x)
    if kind=='critical':
        q=np.sin(2*np.pi*x)**4;rho=1+.1*q;v=.4+.1*q;p=1+.1*q
    return np.stack((rho,rho*v,p/.4+.5*rho*v*v),axis=-1)

def averages(n,kind):
    z,w=np.polynomial.legendre.leggauss(24)
    return .5*np.einsum('j,ijk->ik',w,profile((np.arange(n)[:,None]+.5+.5*z)/n,kind))

def spatial(output):
    rows=[]
    for kind in ['constant','density','momentum','pressure','coupled','critical']:
        for variant in ['historical_teno','historical_v2_soft25','fv_teno','fv_v2_soft25','safe_fv_teno']:
            previous=None
            for n in [40,80,160,320]:
                u=jnp.asarray(averages(n,kind));rec=reconstruction(variant)
                f=flux_variant(rec,'periodic',formulation(variant),face_guard=variant.startswith('safe'))
                start=time.perf_counter();result=jax.jit(f)(u).block_until_ready()
                exact=np.diff(np.asarray(flux_jax(jnp.asarray(profile(np.arange(n+1)/n,kind)))),axis=0)*n
                e=errors(np.asarray(result)*n,exact)
                row=dict(profile=kind,variant=variant,cells=n,**e,
                    order=None if previous is None or e['l1']<1e-13 or previous<1e-13 else np.log2(previous/e['l1']),
                    summed_divergence=np.asarray(result).sum(axis=0),compile_and_execute_s=time.perf_counter()-start)
                if variant.startswith('safe'):
                    _,d=state_faces(u,rec,'periodic',True);row['face_theta']=float(d['face_theta'])
                rows.append(row);previous=e['l1']
            print('spatial',kind,variant,'last order',rows[-1]['order'],flush=True)
    write(output,rows)

@lru_cache(None)
def step_kernel(variant,boundary):
    rec=reconstruction(variant)
    if variant.startswith('safe'):
        return jax.jit(lambda u,dt,dx: retry_step(u,dt,dx,rec,boundary))
    flux=flux_variant(rec,boundary,formulation(variant),True,face_guard='faceguard' in variant)
    def step(u,dt,dx):
        def stage(v,base,weight):
            net,(l,r)=flux(v)
            candidate=base+weight*v-weight*dt/dx*net
            ok=jnp.all(admissible(candidate))
            return jnp.where(ok,candidate,v),dict(ok=ok,low_ok=ok,
                theta=jnp.array(1.),raw_invalid=jnp.sum(~admissible(candidate)),
                min_density=jnp.min(candidate[:,0]),min_pressure=jnp.min(pressure(candidate)),
                exchange=dt*(r-l),cfl=dt/dx*jnp.max(jnp.abs(spds_jax(v))),
                face_theta=jnp.array(1.),raw_faces_valid=jnp.array(True),raw_faces_finite=jnp.array(True))
        a,da=stage(u,jnp.zeros_like(u),1.)
        b,db=stage(a,.75*u,.25)
        v,dc=stage(b,u/3.,2./3.)
        diag=jax.tree.map(lambda *x:jnp.stack(x),da,db,dc)
        ok=jnp.all(diag['ok'])&jnp.all(admissible(v))
        return jnp.where(ok,v,u),dict(success=ok,stages=diag,
            exchange=(da['exchange']+db['exchange'])/6.+2*dc['exchange']/3.,
            accepted_dt=jnp.where(ok,dt,0.),retries=jnp.array(0))
    return jax.jit(step)

def evolve(initial,variant,boundary,final_time,dx,dt,snapshots=10):
    kernel=step_kernel(variant,boundary);u=jnp.asarray(initial);t=0.;exchange=np.zeros(3)
    history=[];sample=[np.asarray(u)];times=[0.];steps=0;retries=0;limit=0;face_limit=0;invalid=0
    minrho=float(np.min(initial[:,0]));minp=float(np.min(pressure(u)));success=True
    compile_start=time.perf_counter()
    compiled=kernel.lower(u,jnp.array(dt),jnp.array(dx)).compile()
    compilation=time.perf_counter()-compile_start
    # Separate first execution and synchronized warmed single-step timings.
    start=time.perf_counter();v,d=compiled(u,jnp.array(dt),jnp.array(dx));v.block_until_ready()
    first=time.perf_counter()-start
    repeats=[]
    for _ in range(3):
        start=time.perf_counter();v,d=compiled(u,jnp.array(dt),jnp.array(dx));v.block_until_ready();repeats.append(time.perf_counter()-start)
    start=time.perf_counter()
    while t<final_time-1e-14 and steps<100000:
        attempt=min(dt,final_time-t)
        v,d=compiled(u,jnp.array(attempt),jnp.array(dx));v.block_until_ready()
        d=jax.tree.map(np.asarray,d);steps+=1
        retries+=int(d['retries']);invalid+=int(np.sum(d['stages']['raw_invalid']))
        if not d['success']:
            success=False
            history.append(dict(time=t,success=False,diagnostics=d));break
        u=v;t+=float(d['accepted_dt']);exchange+=d['exchange']
        limit+=int(np.sum(d['stages']['theta']<1));face_limit+=int(np.sum(d['stages']['face_theta']<1))
        minrho=min(minrho,float(d['stages']['min_density'].min()));minp=min(minp,float(d['stages']['min_pressure'].min()))
        history.append(dict(time=t,min_density=minrho,min_pressure=minp,
            theta_min=float(d['stages']['theta'].min()),face_theta_min=float(d['stages']['face_theta'].min()),retries=int(d['retries'])))
        if t>=len(times)*final_time/snapshots-1e-14:sample.append(np.asarray(u));times.append(t)
    elapsed=time.perf_counter()-start
    success=success and abs(t-final_time)<1e-12
    final=np.asarray(u)
    diag=dict(success=success,final_time=t,steps=steps,retries=retries,limited_fe_stages=limit,
        limited_face_stages=face_limit,raw_invalid_fe_rows=invalid,minimum_density=minrho,minimum_pressure=minp,
        limiter_frequency=limit/max(3*steps,1),face_limiter_frequency=face_limit/max(3*steps,1),
        conservation_residual=dx*np.sum(final-initial,axis=0)+exchange,boundary_exchange=exchange,
        compile_s=compilation,first_execution_s=first,warm_step_median_s=float(np.median(repeats)),
        warm_step_repeats_s=repeats,simulation_s=elapsed,host_instrumentation_included=True,
        first_failed_step=None if success else steps)
    return final,diag,history,np.array(times),np.stack(sample)

def temporal(output):
    # Entropy wave is an exact Euler solution; special linear-in-density flux.
    rows=[];ex=exact_module();end=.02
    for variant in ['historical_teno','fv_teno','fv_v2_soft25','safe_fv_teno']:
        previous=None
        for n in [40,80,160]:
            def solution(t):
                rho=ex.exact_fourier(n,1.,[[.1,1,0]],t,.4)
                return np.stack((rho,.4*rho,2.5+.08*rho),axis=-1)
            dt=.2*(1/n)**(5/3);count=int(np.ceil(end/dt));dt=end/count
            results=[]
            for refine in [1,2]:
                final,d,_,_,_=evolve(solution(0),variant,'periodic',end,1/n,dt/refine)
                results.append(final);e=errors(final,solution(end))
                rows.append(dict(variant=variant,cells=n,dt=dt/refine,refinement=refine,**e,diagnostics=d,
                    order=None if previous is None else np.log2(previous[refine-1]/e['l1'])))
            previous=[r['l1'] for r in rows[-2:]]
            rows[-2]['temporal_cauchy_l1']=float(np.mean(abs(results[0]-results[1])))
            print('temporal',variant,n,rows[-2]['l1'],flush=True)
    write(output,rows)

def cases():
    return [dict(id='sod',kind='riemann',system='euler',left=[1.,0.,1.],right=[.125,0.,.1],jump=.5,time=.15,boundary='transmissive'),
        dict(id='lax',kind='riemann',system='euler',left=[.445,.698,3.528],right=[.5,0.,.571],jump=.5,time=.1,boundary='transmissive'),
        dict(id='two_rarefactions',kind='riemann',system='euler',left=[1.,-2.,.4],right=[1.,2.,.4],jump=.5,time=.08,boundary='transmissive'),
        dict(id='contrast',kind='riemann',system='euler',left=[10.,0.,100.],right=[.01,0.,.01],jump=.5,time=.012,boundary='transmissive'),
        dict(id='near_vacuum',kind='riemann',system='euler',left=[1e-5,-.01,1e-8],right=[1e-5,.01,1e-8],jump=.5,time=.1,boundary='transmissive'),
        dict(id='shu_osher',kind='shu-osher',system='euler',time=.15,length=10.,boundary='transmissive'),
        dict(id='blast_smoke',kind='blast',system='euler',time=.001,boundary='reflective-wall'),
        dict(id='blast_full',kind='blast',system='euler',time=.038,boundary='reflective-wall')]

def euler(output,grids=(64,128)):
    output.mkdir(exist_ok=False);rows=[]
    for n in grids:
        for case in cases():
            u=benchmark_state(case,n);dx=case.get('length',1.)/n
            speed=np.max(abs(np.asarray(spds_jax(jnp.asarray(u)))))
            dt=case['time']/max(1,int(np.ceil(case['time']*speed*1.5/(.25*dx))))
            reference=benchmark_state(case,n,case['time']) if case['kind']=='riemann' else None
            if case['id']=='shu_osher':
                # Existing independently generated cell-average references at
                # T=1.8 cannot be used at T=.15. Keep this as stability-only.
                pass
            for variant in VARIANTS:
                target=output/f'{case["id"]}_{n}_{variant}.json'
                if target.exists():raise FileExistsError(target)
                final,d,h,t,states=evolve(u,variant,case['boundary'],case['time'],dx,dt)
                row=dict(case=case['id'],definition=case,cells=n,variant=variant,dt=dt,dx=dx,**d)
                if reference is not None and d['success']:
                    row.update(errors(final,reference));row['reference']='analytic Euler Riemann cell averages'
                    row['error_over_time']=[dict(time=float(q),**errors(v,benchmark_state(case,n,q))) for q,v in zip(t,states)]
                    row['density_tv']=float(np.sum(abs(np.diff(final[:,0]))))
                    row['pressure_tv']=float(np.sum(abs(np.diff(np.asarray(pressure(jnp.asarray(final)))))))
                    row['density_overshoot']=max(0.,float(np.max(final[:,0])-np.max(reference[:,0])))
                    row['density_undershoot']=max(0.,float(np.min(reference[:,0])-np.min(final[:,0])))
                else:row['reference']='none; stability-only' if reference is None else 'failed trajectory; no error score'
                write(target,dict(**row,history=h))
                np.savez_compressed(target.with_suffix('.npz'),initial=u,final=final,
                    reference=np.array([]) if reference is None else reference,times=t,states=states)
                rows.append(row);print('euler',case['id'],n,variant,d['success'],row.get('l1'),d['limited_face_stages'],flush=True)
    # Pair on identical case/grids; norms remain per component and physics.
    for row in rows:
        base=next(x for x in rows if x['case']==row['case'] and x['cells']==row['cells'] and x['variant']=='historical_teno')
        if row.get('l1') is not None and base.get('l1',0)>1e-12:
            row['ratio_to_historical_teno']=row['l1']/base['l1'];row['percent_change']=100*(row['ratio_to_historical_teno']-1)
        base=next(x for x in rows if x['case']==row['case'] and x['cells']==row['cells'] and x['variant']=='fv_teno')
        if row.get('l1') is not None and base.get('l1',0)>1e-12:row['ratio_to_fv_teno']=row['l1']/base['l1']
    write(output/'summary.json',rows)

def baseline_comparison(output):
    fresh=json.loads((OUT/'results/baseline_smoke/per_case.json').read_text())
    old=json.loads((ROOT/'docs/research_release/results/full-validation/per_case.json').read_text())
    key=lambda r:tuple(r[k] for k in ['case_id','physics','cells','mode','strength'])
    lookup={key(r):r for r in old};rows=[]
    for r in fresh:
        match=lookup[key(r)]
        delta=max(abs(r[k]-match[k]) for k in ['l1','l2','linf'])
        rows.append(dict(key=key(r),maximum_norm_difference=delta,valid_equal=r['valid']==match['valid']))
    write(output,dict(passed=all(r['maximum_norm_difference']<1e-12 and r['valid_equal'] for r in rows),rows=rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['spatial','temporal','euler','baseline'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--grids',type=int,nargs='+',default=[64,128]);a=p.parse_args()
    if a.action=='euler':euler(a.output,tuple(a.grids))
    else:{'spatial':spatial,'temporal':temporal,'baseline':baseline_comparison}[a.action](a.output)
