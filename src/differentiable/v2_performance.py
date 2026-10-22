"""Synchronized CPU timing of matched uninstrumented trajectories.

Run after other numerical workers finish. No diagnostic/plot/export overhead is
included in warmed kernels. Lowering, compilation and first execution are separate.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from .artifacts import create_output,finish_manifest,provenance,write_json
from .euler_jax import get_euler_flux_jax
from .evaluation import errors
from .model_io import REPO_ROOT,load_checkpoint
from .numerics_jax import lax_friedrichs_flux,rollout
from .reconstruction import make_reconstruction
from .v2_benchmark import state,settings


def benchmark(config,output,smoke=False):
    records=[]
    cases=[dict(id='timing-advection',kind='advection',system='advection',boundary='periodic',time=.5),
           dict(id='timing-sod',kind='riemann',system='euler',boundary='transmissive',time=.17,left=[1.,0.,1.],right=[.125,0.,.1],jump=.47)]
    if smoke:cases=cases[:1]
    grids=[64] if smoke else [64,128,256,512]
    for case in cases:
        physics='euler' if case['system']=='euler' else 'scalar'
        model=load_checkpoint(config['models'][physics]['path'],config['models'][physics]['sha256'])
        for n in grids:
            opts=settings(case,n);u=jnp.asarray(state(case,n));reference=state(case,n,case['time'])
            policies=[('teno5',0.),('gated-soft',.25),('gated-hard',.25),('neural-soft',1.)]
            for method,strength in policies[:2] if smoke else policies:
                rec=make_reconstruction(model,method,strength)
                flux=get_euler_flux_jax(rec,case['boundary']) if physics=='euler' else lambda u:lax_friedrichs_flux(u,lambda z:.83*z,.83,rec)
                fn=jax.jit(lambda v:rollout(v,flux,opts['dt'],opts['dx'],opts['steps']))
                started=time.perf_counter();lowered=fn.lower(u);lowering=time.perf_counter()-started
                started=time.perf_counter();executable=lowered.compile();compilation=time.perf_counter()-started
                started=time.perf_counter();final=executable(u);final.block_until_ready();first=time.perf_counter()-started
                durations=[]
                for _ in range(9):
                    started=time.perf_counter();final=executable(u);final.block_until_ready();durations.append(time.perf_counter()-started)
                memory=executable.memory_analysis();seconds=float(np.median(durations));interfaces=n+1 if physics=='euler' else n
                record=dict(case=case['id'],physics=physics,cells=n,method=method,strength=strength,
                     checkpoint_sha256=model.sha256,steps=opts['steps'],final_time=case['time'],
                     lowering_seconds=lowering,compilation_seconds=compilation,first_execution_seconds=first,
                     warmed_seconds=durations,warmed_median_seconds=seconds,
                     seconds_per_step=seconds/opts['steps'],seconds_per_cell_step=seconds/opts['steps']/n,
                     seconds_per_interface_stage=seconds/opts['steps']/interfaces/3,
                     executable_temporary_bytes=memory.temp_size_in_bytes if memory else None,
                     executable_output_bytes=memory.output_size_in_bytes if memory else None,
                     process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     memory_limitation='RSS is process lifetime high-water mark; executable temporary estimate is not measured device peak',
                     **errors(np.asarray(final),reference))
                records.append(record)
                print(f'{physics} N={n} {method}: {seconds:.6f}s; L1={record["l1"]:.6g}',flush=True)
            jax.clear_caches()
    comparisons=[]
    for physics in sorted({r['physics'] for r in records}):
        subset=[r for r in records if r['physics']==physics]
        for n in grids:
            classical=next(r for r in subset if r['cells']==n and r['method']=='teno5')
            neural=next(r for r in subset if r['cells']==n and r['method']=='gated-soft')
            budget=classical['warmed_median_seconds']
            feasible=[r for r in subset if r['method']=='gated-soft' and r['warmed_median_seconds']<=budget]
            target=classical['l1'];accurate=[r for r in subset if r['method']=='gated-soft' and r['l1']<=target]
            comparisons.append(dict(physics=physics,classical_cells=n,same_grid_l1_ratio=neural['l1']/target,
                same_grid_runtime_ratio=neural['warmed_median_seconds']/budget,
                same_grid_inference_overhead_seconds=neural['warmed_median_seconds']-budget,
                equal_budget_best=min(feasible,key=lambda r:r['l1']) if feasible else None,
                cells_reaching_classical_accuracy=min(accurate,key=lambda r:r['cells'])['cells'] if accurate else None,
                limitation='Discrete tested grids and one timed IVP per physics; no extrapolation or speedup guarantee'))
    write_json(output/'performance.json',dict(smoke=smoke,repetitions=9,warmup=1,precision='float64',backend=jax.default_backend(),
          timing='block_until_ready; compilation excluded from warmed execution; uninstrumented production rollout',
          results=records,comparisons=comparisons))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=REPO_ROOT/'configs/v2/full.json');p.add_argument('--smoke',action='store_true');args=p.parse_args(argv)
    config=json.loads(args.config.read_text());out=create_output(args.output)
    write_json(out/'configuration.json',dict(config,smoke=args.smoke))
    environment=provenance(args.config)
    environment['timing_policy']='Production kernels; lowering, compilation and first execution separate; nine warmed block_until_ready repeats'
    environment['numerical_defaults']['neural']['inference_projection']='explicit per-method soft/hard; see performance.json'
    write_json(out/'provenance.json',environment)
    benchmark(config,out,args.smoke);finish_manifest(out)


if __name__=='__main__':main()
