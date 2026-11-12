"""Measure rejection/intervention on an analytically smooth transport trajectory."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts import methods as m
from experiments.four_method_comparison.scripts.run_matched_benchmarks import write,RAW
from src.differentiable.teno5_jax import teno5_selection_values
import numpy as np

def main():
    rows=[]
    for n in (64,128,256):
        for method in m.MAIN:
            with np.load(RAW/f'run_advection_smooth_{n}_{method}.npz') as a:
                for stage in ('initial','final'):
                    u=m.jnp.asarray(a[stage]);s=m.gather(u,(-2,-1,0,1,2,3) if method=='neno6' else (-2,-1,0,1,2),'periodic')
                    if method=='neno6':keep=m.neno_probabilities(s)<.5
                    elif method=='nn_teno5':keep=m.original_probabilities(s)<m.THRESHOLDS
                    elif method=='teno5':keep=teno5_selection_values(s)>=1e-5
                    else:
                        c=m.load_checkpoint(m.HEADS['scalar']);keep=m.toy_keep_weight(c.backbone,s)>.5
                    r=dict(case='advection_smooth',cells=n,method=method,stage=stage,
                        any_candidate_discard_fraction=float(m.jnp.mean(m.jnp.any(~keep,axis=-1))),
                        discarded_candidate_fraction=float(m.jnp.mean(~keep)),
                        all_discard_fraction=float(m.jnp.mean(m.jnp.all(~keep,axis=-1))),
                        interpretation='Every exact stencil is smooth; rejection fraction is descriptive, not a TENO teacher classification score')
                    if method=='learned_head':
                        lam=np.asarray(m.lambda_function('scalar')(s));r.update(mean_lambda=float(lam.mean()),minimum_lambda=float(lam.min()),maximum_lambda=float(lam.max()))
                    rows.append(r)
    write(RAW/'smooth_input_diagnostics.json',rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
