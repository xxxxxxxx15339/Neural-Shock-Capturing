"""Run once after freezing paired candidates; no tuning from these results."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from experiments.accuracy_upgrade import study
from experiments.accuracy_upgrade.study import OUT,write,evolve
from src.differentiable.model_io import load_checkpoint
from src.differentiable.reconstruction import make_reconstruction
from src.differentiable.euler_jax import spds_jax
from src.differentiable.v2_benchmark import state,exact_module
from src.differentiable.evaluation import errors


def holdout(output):
    # Require all predeclared pilot candidates before looking at holdout results.
    training=json.loads((OUT/'checkpoints/paired_pilot/summary.json').read_text())
    assert len(training)==6
    output.mkdir(exist_ok=False)
    variants=list(study.VARIANTS)
    hashes={}
    for r in training:
        p=OUT/f'checkpoints/paired_pilot/{r["variant"]}_seed{r["seed"]}_final.pkl'
        m=load_checkpoint(p);hashes[str(p.relative_to(OUT))]=m.sha256
        name=('fv_faceguard' if r['variant']=='C' else 'historical')+f'_pilot{r["variant"]}_seed{r["seed"]}'
        study.EXTRA_RECONSTRUCTIONS[name]=make_reconstruction(m,'gated-soft',.25);variants.append(name)
    config=OUT/'configs/final_holdout.json'
    write(output/'frozen_candidates.json',dict(checkpoints=hashes,holdout_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        selection='all predeclared candidates; no selection or tuning on holdout'))
    rows=[]
    for case in json.loads(config.read_text())['cases']:
        for n in case['grids']:
            u=state(case,n);ref=state(case,n,case['time']);dx=1/n
            speed=float(jnp.max(jnp.abs(spds_jax(jnp.asarray(u)))))
            dt=case['time']/int(np.ceil(case['time']*speed*1.5/(.25*dx)))
            for variant in variants:
                final,d,h,t,states=evolve(u,variant,case['boundary'],case['time'],dx,dt)
                row=dict(case=case['id'],cells=n,variant=variant,**d)
                if d['success']:row.update(errors(final,ref))
                rows.append(row);print('holdout',case['id'],n,variant,d['success'],row.get('l1'),flush=True)
    write(output/'per_case.json',rows)
    # Independent trajectories are resampled as pairs at each grid; never
    # pretend the two grids or multiple snapshots are independent IVPs.
    stats=[]
    for n in [64,128]:
        for variant in variants:
            for comparator in ['historical_teno','fv_teno']:
                group=[r for r in rows if r['cells']==n and r['variant']==variant]
                base=[next(x for x in rows if x['cells']==n and x['case']==r['case'] and x['variant']==comparator) for r in group]
                if any(not r['success'] for r in group+base):
                    stats.append(dict(cells=n,variant=variant,baseline=comparator,failures=sum(not r['success'] for r in group)));continue
                a=np.array([r['l1'] for r in group]);b=np.array([r['l1'] for r in base]);ratio=a/b
                rng=np.random.default_rng(2026111202);ix=rng.integers(0,len(ratio),(2000,len(ratio)))
                ci=np.quantile(np.mean(ratio[ix],axis=1),[.025,.975])
                stats.append(dict(cells=n,variant=variant,baseline=comparator,trajectories=len(a),failures=0,
                    mean_l1=a.mean(),median_l1=np.median(a),maximum_l1=a.max(),mean_ratio=ratio.mean(),median_ratio=np.median(ratio),
                    worst_ratio=ratio.max(),ci95=ci,percent_change=100*(ratio.mean()-1)))
    write(output/'statistics.json',stats)


def shu_reference(output):
    output.mkdir(exist_ok=False)
    case=next(c for c in study.cases() if c['id']=='shu_osher');refs={};rows=[]
    for n,refine in [(256,1),(512,1),(1024,1),(1024,2)]:
        u=state(case,n);dx=10/n;speed=float(jnp.max(jnp.abs(spds_jax(jnp.asarray(u)))))
        dt=case['time']/int(np.ceil(case['time']*speed*1.5/(.25*dx)))/refine
        final,d,h,t,states=evolve(u,'fv_teno','transmissive',case['time'],dx,dt)
        if not d['success']:raise RuntimeError('Invalid reference; no replacement fabricated')
        refs[n,refine]=final;np.savez_compressed(output/f'N{n}_dt{refine}.npz',final=final)
        rows.append(dict(cells=n,refinement=refine,dt=dt,**d));print('Shu reference',n,refine,flush=True)
    comparisons=[]
    for n in [64,128]:
        restrict=lambda u:u.reshape(n,-1,3).mean(axis=1)
        r256=restrict(refs[256,1]);r512=restrict(refs[512,1]);r1024=restrict(refs[1024,1]);ref=restrict(refs[1024,2])
        uncertainty=2*(np.mean(abs(r1024-r512))+np.mean(abs(ref-r1024)))
        density_uncertainty=2*(np.mean(abs(r1024[:,0]-r512[:,0]))+np.mean(abs(ref[:,0]-r1024[:,0])))
        base=np.load(OUT/f'results/euler_actual_stages/shu_osher_{n}_historical_teno.npz')['final']
        baseline_error=errors(base,ref)['l1']
        for variant in study.VARIANTS:
            final=np.load(OUT/f'results/euler_actual_stages/shu_osher_{n}_{variant}.npz')['final']
            row=dict(cells=n,variant=variant,**errors(final,ref),uncertainty_proxy=uncertainty,
                density_uncertainty_proxy=density_uncertainty,space_cauchy_coarse=np.mean(abs(r512-r256)),
                space_cauchy_fine=np.mean(abs(r1024-r512)),time_cauchy=np.mean(abs(ref-r1024)),
                density_oscillation_range=float(np.ptp(final[int(.25*n):,0])),
                reference_density_oscillation_range=float(np.ptp(ref[int(.25*n):,0])),
                ranking_resolved=uncertainty<.25*abs(errors(final,ref)['l1']-baseline_error),
                reference='FV TENO5 256/512/1024 plus finest dt/2; empirical Cauchy proxy, not error bound')
            comparisons.append(row)
        np.savez_compressed(output/f'reference_N{n}.npz',final=ref)
    write(output/'reference_runs.json',rows);write(output/'comparisons.json',comparisons)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['holdout','shu']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    (holdout if a.action=='holdout' else shu_reference)(a.output)
