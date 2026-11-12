"""Execute frozen matched raw-method tests, references and same-backbone ablations."""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts import methods as m
import jax
import jax.numpy as jnp
import numpy as np
from src.initial_conditions.InitialConditions import shuOsher

OUT=m.OUT; RAW=OUT/'results/raw'; MANIFEST=OUT/'configs/benchmark_manifest.json'

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def write(path,v):path.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')

@lru_cache(None)
def exact_module():
    path=ROOT/'data/nnteno_v2/build_dataset.py'
    spec=importlib.util.spec_from_file_location('comparison_exact',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def sine_entropy(x,t):
    """Entropy solution of u0=.5+.25 sin(2*pi*x), including post-shock branches.

    In the moving frame the shock stays at y=.5. The root on [0,.5]
    is the left entropy branch; reflection gives the right branch.
    The value exactly on the shock is the mean of its traces.
    """
    y=np.mod(x-.5*t,1.);z=np.minimum(y,1-y)
    lo=np.zeros_like(z);hi=np.full_like(z,.5)
    for _ in range(65):
        mid=.5*(lo+hi);mapped=mid+.25*t*np.sin(2*np.pi*mid)
        lower=mapped<z;lo=np.where(lower,mid,lo);hi=np.where(lower,hi,mid)
    v=.25*np.sin(2*np.pi*.5*(lo+hi));v=np.where(y<=.5,v,-v)
    v=np.where(abs(y-.5)<2e-14,0.,v)
    return .5+v

def initial(case,n):
    x=(np.arange(n)+(.5 if case.get('sampling')=='point-cell-centers' else 0.))*case['length']/n;id=case['id']
    if 'left' in case:
        return exact_module().ExactEuler(case['left'],case['right'],case['jump'])(x,0.)
    if id=='shu_osher':return shuOsher()(x)
    if id.startswith('blast'):
        p=np.where(x<.1,1000.,np.where(x>.9,100.,.01))
        return np.stack((np.ones(n),np.zeros(n),p/.4),axis=-1)
    if id.startswith('burgers'):
        return ((x>=.2)&(x<.6)).astype(float) if id=='burgers_hat' else sine_entropy(x,0.)
    if id=='advection_square':return ((x>=.25)&(x<.75)).astype(float)
    return .5+.2*np.sin(2*np.pi*x)+.1*np.cos(4*np.pi*x)

def reference(case,x,t):
    if 'left' in case:return exact_module().ExactEuler(case['left'],case['right'],case['jump'])(x,t)
    if case['id'] in ('burgers_pre','burgers_post'):return sine_entropy(x,t)
    if case['id']=='burgers_hat':
        if t==0:return ((x>=.2)&(x<.6)).astype(float)
        return np.where((x>=.2)&(x<.2+t),(x-.2)/t,np.where((x>=.2+t)&(x<.6+.5*t),1.,0.))
    if case['id']=='advection_square':
        y=np.mod(x-t,1.);return ((y>=.25)&(y<.75)).astype(float)
    if case['id']=='advection_smooth':return .5+.2*np.sin(2*np.pi*(x-t))+.1*np.cos(4*np.pi*(x-t))
    return None

def timestep(case,n,manifest):
    dx=case['length']/n
    speed=float(jnp.max(jnp.abs(m.spds_jax(jnp.asarray(initial(case,n)))))) if case['physics']=='euler' else 1.
    maxdt=manifest['cfl']*dx/(manifest['initial_wave_speed_safety']*speed)
    if case.get('smooth'):maxdt=min(maxdt,.15*dx**(5/3))
    steps=max(1,int(np.ceil(case['time']/maxdt)))
    return case['time']/steps,dx,steps

def block(tree):return jax.tree.map(lambda a:a.block_until_ready(),tree)

def evolve(case,n,method,manifest,dt_refine=1,timing=True):
    physics=case['physics'];u0=jnp.asarray(initial(case,n));u=u0
    if not bool(m.admissible(u0,physics)):raise ValueError('Invalid initial state; refusing to evaluate flux/eigenbasis')
    dt,dx,steps=timestep(case,n,manifest);dt/=dt_refine;steps*=dt_refine
    jd,jx=jnp.array(dt),jnp.array(dx)
    kernel=m.checked_step(method,physics,case['boundary'],case['equation'])
    start=time.perf_counter();compiled=kernel.lower(u0,jd,jx).compile();step_compile=time.perf_counter()-start
    start=time.perf_counter();block(compiled(u0,jd,jx));first=time.perf_counter()-start
    repeats=[]
    if timing:
        for _ in range(manifest['runtime']['warm_step_repeats']):
            start=time.perf_counter();block(compiled(u0,jd,jx));repeats.append(time.perf_counter()-start)
    exchange=np.zeros(3) if physics=='euler' else np.array(0.)
    raw_min_rho=float(np.min(np.asarray(u0)[:,0])) if physics=='euler' else float(np.min(np.asarray(u0)))
    raw_min_p=float(np.min(m.pressure(u0))) if physics=='euler' else None
    frames=[np.asarray(u0)];times=[0.];history=[];success=True;failure=None
    failure_candidate=np.array([]);last_admissible_stage=np.array([]);maxstage_cfl=0.
    start=time.perf_counter();accepted=0
    for i in range(steps):
        if physics=='euler':maxstage_cfl=max(maxstage_cfl,float(jnp.max(jnp.abs(m.spds_jax(u))))*dt/dx)
        v,diag=compiled(u,jd,jx);d=jax.tree.map(np.asarray,diag)
        for r in d['stage_min_density']:
            if np.isfinite(r):raw_min_rho=min(raw_min_rho,float(r))
        if physics=='euler':
            for p in d['stage_min_pressure']:
                if np.isfinite(p):raw_min_p=min(raw_min_p,float(p))
        history.append([float((i+1)*dt),*d['stage_min_density'].tolist(),*d['stage_min_pressure'].tolist()])
        if not d['success']:
            success=False;failure=dict(step=i+1,stage=int(d['first_invalid_stage']),step_start_time=i*dt,
                stage_min_density=d['stage_min_density'],stage_min_pressure=d['stage_min_pressure'],
                nonfinite_values=int(np.sum(~np.isfinite(d['invalid_candidate']))))
            failure_candidate=d['invalid_candidate'];last_admissible_stage=d['last_admissible_stage'];break
        u=v;accepted+=1;exchange+=d['boundary_exchange']
        if (i+1)%max(1,steps//12)==0 or i==steps-1:
            frames.append(np.asarray(u));times.append((i+1)*dt)
    instrumented=time.perf_counter()-start
    final=np.asarray(u)
    # A whole-trajectory device loop separates execution from Python diagnostics.
    full_compile=None;full_times=[]
    if timing:
        def whole(state,dt,dx,count):
            ex=jnp.zeros(3) if physics=='euler' else jnp.array(0.)
            def cond(carry):return (carry[0]<count)&carry[2]
            def body(carry):
                i,v,ok,ex=carry;v,d=kernel(v,dt,dx)
                return i+1,v,d['success'],ex+d['boundary_exchange']
            return jax.lax.while_loop(cond,body,(jnp.array(0),state,jnp.array(True),ex))
        start=time.perf_counter()
        full=jax.jit(whole).lower(u0,jd,jx,jnp.array(steps)).compile();full_compile=time.perf_counter()-start
        block(full(u0,jd,jx,jnp.array(steps)))
        for _ in range(manifest['runtime']['full_rollout_repeats']):
            start=time.perf_counter();res=block(full(u0,jd,jx,jnp.array(steps)));full_times.append(time.perf_counter()-start)
        if bool(res[2])!=success or not np.allclose(np.asarray(res[1]),final,atol=3e-11,rtol=3e-11):
            raise AssertionError('Timed and instrumented trajectories disagree')
    if times[-1]!=accepted*dt:
        frames.append(final);times.append(accepted*dt)
    x=(np.arange(n)+(.5 if case.get('sampling')=='point-cell-centers' else 0.))*dx
    row=dict(case=case['id'],physics=physics,method=method,cells=n,requested_time=case['time'],
        actual_time=accepted*dt,dt=dt,dx=dx,planned_steps=steps,accepted_steps=accepted,success=success,
        status='NEW MATCHED EVALUATION',raw_min_density=raw_min_rho if physics=='euler' else None,
        raw_min_pressure=raw_min_p,failure=failure,conservation_residual=dx*np.sum(final-np.asarray(u0),axis=0)+exchange,
        boundary_exchange=exchange,repair_count=0,retry_count=0,maximum_step_start_cfl=maxstage_cfl,
        compile_step_seconds=step_compile,compile_rollout_seconds=full_compile,first_step_seconds=first,
        warm_step_repeats_seconds=repeats,rollout_repeats_seconds=full_times,instrumented_rollout_seconds=instrumented,
        reference=case['reference'],manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    arrays=dict(x=x,initial=np.asarray(u0),final=final,times=np.asarray(times),states=np.stack(frames),
        stage_history=np.asarray(history),invalid_candidate=failure_candidate,last_admissible_stage=last_admissible_stage)
    ref=reference(case,x,case['time'])
    arrays['reference']=np.array([]) if ref is None else ref
    if method=='learned_head' and physics=='euler' and case['id'] in ('sod','shu_osher'):
        arrays['lambda_final']=np.asarray(m.euler_lambda(jnp.asarray(final),case['boundary']))
        arrays['lambda_initial']=np.asarray(m.euler_lambda(u0,case['boundary']))
    return row,arrays

def save_run(base,row,arrays):
    write(base.with_suffix('.json'),row);np.savez_compressed(base.with_suffix('.npz'),**arrays)

def run_references(manifest):
    case=next(c for c in manifest['cases'] if c['id']=='shu_osher')
    for n in manifest['shu_reference']['grids']:
        for refine in (manifest['shu_reference']['finest_dt_refinements'] if n==max(manifest['shu_reference']['grids']) else [1]):
            base=RAW/f'reference_shu_{n}_dt{refine}'
            if base.with_suffix('.json').exists():continue
            row,arr=evolve(case,n,'teno5',manifest,refine,timing=False)
            row['status']='NEW MATCHED EVALUATION; NUMERICAL REFERENCE'
            save_run(base,row,arr)
            print('REFERENCE',n,refine,row['success'],flush=True)

def run_cases(manifest,case_ids=None):
    for case in manifest['cases']:
        if case_ids and case['id'] not in case_ids:continue
        for n in case['grids']:
            methods=list(manifest['methods'])
            if case['id'] in manifest['ablation']['cases'] and n in manifest['ablation']['grids']:
                methods+=['head_backbone_only','head_fixed_blend']
            for method in methods:
                base=RAW/f'run_{case["id"]}_{n}_{method}'
                if base.with_suffix('.json').exists():
                    assert json.loads(base.with_suffix('.json').read_text())['manifest_sha256']==hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
                    assert base.with_suffix('.npz').exists()
                    continue
                row,arr=evolve(case,n,method,manifest)
                save_run(base,row,arr)
                print('RUN',case['id'],n,method,row['success'],row['failure'],flush=True)

def main():
    global RAW
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['references','benchmarks','all'])
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--fresh-raw-output',type=Path,help='New empty directory for an independent numerical recomputation; never overwrites stored runs')
    args=parser.parse_args();manifest=json.loads(MANIFEST.read_text())
    if args.fresh_raw_output is not None:
        RAW=args.fresh_raw_output.resolve()
        if RAW.exists() and any(RAW.iterdir()):raise FileExistsError(RAW)
        RAW.mkdir(parents=True,exist_ok=True)
    expected=MANIFEST.with_suffix('.sha256').read_text().split()[0]
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest()==expected,'Frozen protocol changed'
    for cfg in manifest['models'].values():assert hashlib.sha256((ROOT/cfg['path']).read_bytes()).hexdigest()==cfg['sha256']
    if args.action in ('references','all'):run_references(manifest)
    if args.action in ('benchmarks','all'):run_cases(manifest,args.cases)

if __name__=='__main__':main()
