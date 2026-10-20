"""CPU-reproducible V2 validation CLI; no training or original test access.

python -m src.differentiable.v2_evaluate sweep --config configs/v2/full.json --output NEW
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import h5py

from .artifacts import create_output, finish_manifest, provenance, write_json
from .diagnostics import diagnostic_rollout
from .evaluation import errors
from .gated_teno5_jax import gated_nn_teno5_reconstruct
from .model_io import REPO_ROOT, load_checkpoint, sha256_file

DEFAULT_CONFIG = REPO_ROOT/'configs/v2/full.json'


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('split') != 'validation':
        raise ValueError('This study uses validation only; original test is historical evidence')
    if not config['strengths'] or any(not np.isfinite(v) or not 0 <= v <= 1 for v in config['strengths']):
        raise ValueError('Strengths must be finite and in [0,1]')
    if 0 not in config['strengths'] or set(config['modes']) - {'soft','hard'}:
        raise ValueError('Include the classical zero endpoint and explicit candidate modes')
    if config['batch_size'] < 1 or config['bootstrap_samples'] < 1 or config['take_per_family'] < 0:
        raise ValueError('Invalid sampling sizes')
    return config


def load_validation(config, physics):
    root = REPO_ROOT/config['data_root']
    report = json.loads((root/'quality_report.json').read_text())
    name = f'{physics}_validation.h5'
    if sha256_file(root/name) != report['files'][name]['sha256']:
        raise ValueError('Validation dataset hash mismatch')
    buckets = []
    with h5py.File(root/name,'r') as h5:
        for n in config['resolutions']:
            group = h5[f'N{n:03d}']
            decode = lambda values: np.array([v.decode() if isinstance(v,bytes) else str(v) for v in values])
            families = decode(group['family'][:]); ids = decode(group['case_id'][:])
            selected = []
            for family in sorted(set(families)):
                ix = np.flatnonzero(families == family)
                if config['take_per_family']:
                    ix = ix[:config['take_per_family']]
                selected.extend(ix)
            selected = np.sort(selected)
            states = np.asarray(group['states'][:],dtype=float)
            buckets.append(dict(n=n, family=families[selected], ids=ids[selected],
                                initial=states[selected,0], target=np.asarray(group['short_target'][:])[selected],
                                dt=np.asarray(group['short_dt'][:])[selected],
                                speed=np.asarray(group['advection_speed'][:])[selected]))
    return buckets


def make_predictor(model, n, system, mode, batch_size):
    """One executable per grid/PDE/mode; strength and case dt remain dynamic."""
    params = tuple((jnp.asarray(w),jnp.asarray(b)) for w,b in model.backbone)

    @jax.jit
    def predict(initial, dt, speed, strength):
        rec = lambda s: gated_nn_teno5_reconstruct(params,s,mode,strength)
        def single(u,t,a):
            return diagnostic_rollout(u,rec,t,1/n,15,system,
                      'transmissive' if system == 'euler' else 'periodic',
                      wave_speed=a,alpha='dynamic' if system == 'burgers' else jnp.abs(a),
                      return_device=True)
        return jax.vmap(single)(initial,dt,speed)
    return predict


def case_metrics(initial, final, reference, history, n, physics):
    dx = 1/n
    first = dx*np.atleast_1d(np.sum(initial,axis=0))
    last = dx*np.atleast_1d(np.sum(final,axis=0))
    exchange = np.sum(history['boundary_exchange'],axis=0)
    finite = bool(np.isfinite(final).all())
    bad = int(np.sum(history['invalid_rows']))
    periodic = physics == 'scalar'
    differences = final-np.roll(final,1,axis=0) if periodic else np.diff(final,axis=0)
    mins = np.atleast_1d(np.min(final,axis=0)); maxs = np.atleast_1d(np.max(final,axis=0))
    ref_min = np.atleast_1d(np.min(reference,axis=0)); ref_max = np.atleast_1d(np.max(reference,axis=0))
    result = dict(**errors(final,reference), valid=finite and bad == 0,
                  nonfinite=not finite, invalid_stage_rows=bad,
                  total_variation=np.atleast_1d(np.sum(np.abs(differences),axis=0)).tolist(),
                  state_min=mins.tolist(),state_max=maxs.tolist(),
                  overshoot=np.maximum(maxs-ref_max,0).tolist(),
                  undershoot=np.maximum(ref_min-mins,0).tolist(),
                  initial_integral=first.tolist(),final_integral=last.tolist(),
                  boundary_exchange=exchange.tolist(),
                  conservation_residual=(last-first-exchange).tolist(),
                  maximum_effective_cfl=float(np.max(history['max_courant'])),
                  minimum_stage_state=float(np.min(history['minimum_density'])))
    if physics == 'euler':
        result['minimum_density'] = result.pop('minimum_stage_state')
        result['minimum_pressure'] = float(np.min(history['minimum_pressure']))
    return result


def aggregate(rows, config):
    """Paired trajectory bootstrap; macro gives each family×resolution equal weight.

    All failures remain in denominators and disqualify selection. Bootstrap
    intervals condition on valid paired cases; this limitation is explicit.
    Constants and errors below the declared baseline floor have null ratios.
    """
    baseline = {r['case_id']:r for r in rows if r['strength']==0 and r['mode']==config['modes'][0]}
    rng = np.random.default_rng(config['seed'])
    output = []
    for physics in config['physics']:
        for mode in config['modes']:
            for strength in config['strengths']:
                chosen = [r for r in rows if r['physics']==physics and r['mode']==mode and r['strength']==strength]
                groups = {}
                for row in chosen:
                    groups.setdefault((row['family'],row['cells']),[]).append(row)
                group_stats=[]; eligible=[]; draws=[]; numerator_draws=[]; denominator_draws=[]
                for (family,n), group in sorted(groups.items()):
                    valid = [r for r in group if r['valid'] and baseline[r['case_id']]['valid']]
                    metric = lambda key:float(np.mean([r[key] for r in valid])) if valid else None
                    base = float(np.mean([baseline[r['case_id']]['l1'] for r in valid])) if valid else 0.
                    use = family!='constant' and base>config['ratio_error_floor']
                    ratios = []; sampled_values=[]; sampled_bases=[]
                    if use:
                        eligible.extend(valid)
                        values=np.array([r['l1'] for r in valid]);bases=np.array([baseline[r['case_id']]['l1'] for r in valid])
                        for start in range(0, config['bootstrap_samples'],128):
                            ix=rng.integers(0,len(valid),size=(min(128,config['bootstrap_samples']-start),len(valid)))
                            means=values[ix].mean(axis=1);base_means=bases[ix].mean(axis=1)
                            ratios.extend((means/base_means).tolist())
                            sampled_values.extend((means*len(valid)).tolist())
                            sampled_bases.extend((base_means*len(valid)).tolist())
                        draws.append(np.asarray(ratios))
                        numerator_draws.append(sampled_values);denominator_draws.append(sampled_bases)
                    group_stats.append(dict(family=family,cells=n,trajectories=len(group),failures=sum(not r['valid'] for r in group),
                           l1=metric('l1'),l2=metric('l2'),linf=metric('linf'),
                           baseline_l1=base,relative_l1=metric('l1')/base if use else None,
                           ratio_ci95=np.quantile(ratios,[.025,.975]).tolist() if ratios else None,
                           ratio_excluded_reason=None if use else 'constant control or baseline below error floor'))
                valid = [r for r in chosen if r['valid']]
                weights=np.array([sum(1 for r in eligible if r['family']==g['family'] and r['cells']==g['cells']) for g in group_stats if g['relative_l1'] is not None])
                ratio_values=[g['relative_l1'] for g in group_stats if g['relative_l1'] is not None]
                # Bootstrap weighted ratio is formed from resampled group errors;
                # group ratios are weighted by fixed baseline error contributions.
                base_weights=np.array([g['baseline_l1']*w for g,w in zip([g for g in group_stats if g['relative_l1'] is not None],weights)])
                boot=np.stack(draws) if draws else np.empty((0,config['bootstrap_samples']))
                record=dict(physics=physics,mode=mode,strength=strength,trajectories=len(chosen),failures=sum(not r['valid'] for r in chosen),
                     macro_relative_l1=float(np.mean(ratio_values)) if ratio_values else None,
                     trajectory_weighted_relative_l1=(sum(r['l1'] for r in eligible)/sum(baseline[r['case_id']]['l1'] for r in eligible)) if eligible else None,
                     macro_ci95=np.quantile(boot.mean(axis=0),[.025,.975]).tolist() if draws else None,
                     trajectory_weighted_ci95=np.quantile(np.sum(numerator_draws,axis=0)/np.sum(denominator_draws,axis=0),[.025,.975]).tolist() if draws else None,
                     trajectory_weighted_absolute={k:float(np.mean([r[k] for r in valid])) if valid else None for k in ('l1','l2','linf')},
                     macro_absolute={k:float(np.mean([g[k] for g in group_stats if g[k] is not None])) for k in ('l1','l2','linf')},
                     max_conservation_residual=max((float(np.max(np.abs(r['conservation_residual']))) for r in valid),default=None),
                     max_cfl=max((r['maximum_effective_cfl'] for r in valid),default=None),groups=group_stats)
                output.append(record)
    return dict(summary=output,bootstrap=dict(seed=config['seed'],replicates=config['bootstrap_samples'],
          method='paired, stratified by family and resolution; percentile 95%; independent trajectories',
          failure_policy='Failures counted and disqualify selection; intervals conditional on valid trajectories',
          weighted_definition='ratio of trajectory mean L1 errors; macro is mean family-by-resolution ratio'),
          ratio_error_floor=config['ratio_error_floor'])


def sweep(config, output):
    all_rows=[]
    for physics in config['physics']:
        identity=config['models'][physics]
        model=load_checkpoint(identity['path'],identity['sha256'])
        for bucket in load_validation(config,physics):
            n=bucket['n']
            systems=['euler'] if physics=='euler' else ['advection','burgers']
            for system in systems:
                choose=np.ones(len(bucket['ids']),bool) if physics=='euler' else np.char.startswith(bucket['family'],'burgers') == (system=='burgers')
                ix=np.flatnonzero(choose)
                if not len(ix):continue
                for mode in config['modes']:
                    predict=make_predictor(model,n,system,mode,config['batch_size'])
                    for strength in config['strengths']:
                        start=time.perf_counter()
                        for lo in range(0,len(ix),config['batch_size']):
                            selected=ix[lo:lo+config['batch_size']];actual=len(selected)
                            padded=np.pad(selected,(0,config['batch_size']-actual),mode='edge')
                            final,history=jax.tree.map(np.asarray,predict(bucket['initial'][padded],bucket['dt'][padded],bucket['speed'][padded],jnp.asarray(strength,dtype=jnp.float64)))
                            for k,j in enumerate(selected):
                                info=case_metrics(bucket['initial'][j],final[k],bucket['target'][j],{key:v[k] for key,v in history.items()},n,physics)
                                all_rows.append(dict(case_id=bucket['ids'][j],physics=physics,family=bucket['family'][j],cells=n,
                                      system=system,mode=mode,strength=strength,dt=float(bucket['dt'][j]),steps=15,
                                      final_time=float(15*bucket['dt'][j]),checkpoint_sha256=model.sha256,**info))
                        print(f'{physics} N={n} {system} {mode} lambda={strength}: {len(ix)} cases in {time.perf_counter()-start:.2f}s',flush=True)
                    jax.clear_caches()
    write_json(output/'per_case.json',all_rows)
    report=aggregate(all_rows,config)
    write_json(output/'summary.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['sweep'])
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);config=load_config(args.config)
    output=create_output(args.output)
    write_json(output/'configuration.json',config)
    write_json(output/'provenance.json',provenance(args.config))
    report=sweep(config,output)
    finish_manifest(output)
    return int(any(r['failures'] for r in report['summary']))


if __name__=='__main__':
    raise SystemExit(main())
