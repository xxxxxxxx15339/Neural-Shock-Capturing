"""Isolated synchronized production timings, after all other numerical jobs finish."""
import argparse
from pathlib import Path
import resource
import time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from experiments.accuracy_upgrade.study import reconstruction,formulation,VARIANTS,cases,write
from src.differentiable.accuracy_upgrade import flux_variant,safeguarded_step
from src.differentiable.numerics_jax import ssprk3_step
from src.differentiable.v2_benchmark import state
from src.differentiable.euler_jax import spds_jax
from src.differentiable.evaluation import errors


def run(output):
    rows=[];case=cases()[0]
    for n in [64,128,256]:
        u=jnp.asarray(state(case,n));reference=state(case,n,case['time']);dx=1/n
        steps=int(np.ceil(case['time']*float(jnp.max(jnp.abs(spds_jax(u))))*1.5/(.25*dx)));dt=case['time']/steps
        for variant in VARIANTS:
            rec=reconstruction(variant);flux=flux_variant(rec,'transmissive',formulation(variant))
            def execute(u):
                def body(carry,_):
                    v,valid=carry
                    if variant.startswith('safe'):
                        v,d=safeguarded_step(v,dt,dx,rec);valid=valid & d['success']
                    else:v=ssprk3_step(v,flux,dt,dx)
                    return (v,valid),None
                return jax.lax.scan(body,(u,jnp.array(True)),None,length=steps)[0]
            start=time.perf_counter();lowered=jax.jit(execute).lower(u);lower_time=time.perf_counter()-start
            start=time.perf_counter();compiled=lowered.compile();compile_time=time.perf_counter()-start
            start=time.perf_counter();final,valid=compiled(u);final.block_until_ready();first=time.perf_counter()-start
            timings=[]
            for _ in range(9):
                start=time.perf_counter();final,valid=compiled(u);final.block_until_ready();timings.append(time.perf_counter()-start)
            assert bool(valid) and np.isfinite(final).all()
            stats=compiled.memory_analysis()
            row=dict(variant=variant,cells=n,case='Sod',time=case['time'],dt=dt,steps=steps,
                lower_s=lower_time,compile_s=compile_time,first_s=first,warm_repeats_s=timings,
                warm_median_s=float(np.median(timings)),warm_per_step_s=float(np.median(timings))/steps,
                executable_temporary_bytes=stats.temp_size_in_bytes,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**errors(np.asarray(final),reference),
                diagnostic_overhead=False,retry_policy='single attempt, no retries needed on this case')
            rows.append(row);print('performance',n,variant,row['warm_median_s'],row['l1'],flush=True)
    equal_budget=[]
    for n in [64,128,256]:
        base=next(r for r in rows if r['cells']==n and r['variant']=='historical_teno')
        for variant in VARIANTS:
            eligible=[r for r in rows if r['variant']==variant and r['warm_median_s']<=base['warm_median_s']]
            best=min(eligible,key=lambda r:r['l1']) if eligible else None
            equal_budget.append(dict(baseline_cells=n,budget_s=base['warm_median_s'],variant=variant,
                best_measured_grid=None if best is None else best['cells'],
                error_ratio=None if best is None else best['l1']/base['l1'],
                note='discrete measured grids only; absence means no measured affordable grid'))
    write(output,dict(protocol='CPU float64; no concurrent experiment jobs; one first execution plus nine synchronized repeats',
                       rows=rows,equal_budget=equal_budget))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
