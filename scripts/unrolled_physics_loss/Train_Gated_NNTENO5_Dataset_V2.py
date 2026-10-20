#!/usr/bin/env python3
"""Dataset-backed retraining for the existing 5-64-32-16-8-3 NN-TENO5.

Copy to scripts/unrolled_physics_loss/ in Shocks-and-WENO.

Uses *only* the exact-reference HDF5 corpus produced by NN_TENO5_V2_Exact_Reference_Dataset.zip.
- Scalar: advection and inviscid Burgers (different physical fluxes).
- Euler: compressible ideal-gas Euler, characteristic reconstruction.
- Each case uses its OWN short_dt and (for advection) its OWN physical speed.
- Train/validation are disjoint. The test split is NEVER accessed in train mode.
- Candidate-selection backbone remains 5->64->32->16->8->3 (3,155 parameters).
- The deterministic TENO smoothness gate and fixed lambda are applied DURING training.
- Saved .pkl is a backbone list compatible with gated_nn_teno5_reconstruct().
- Euler training uses a positivity penalty/repair for gradient robustness, but
  validation/test rollouts are UNGUARDED and reject invalid states.

Research limitations: this corpus consists of mathematical, analytic solutions,
not experimental sensor data. Its short targets use K=15 fixed SSPRK3 steps and
include no generic post-interaction Burgers/Shu-Osher trajectories. Gating does
not constitute a general high-order proof at smooth critical points. Euler
repair may be nonconservative; validation results never use repaired states.
"""

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import optax
import h5py

from src.differentiable.gated_teno5_jax import gated_nn_teno5_reconstruct
from src.differentiable.teno5_jax import (
    teno5_reconstruct_jax, teno5_selection_values, DEFAULT_CUTOFF,
)
from src.differentiable.teno5_jax_toy import init_toy_params, toy_keep_weight
from src.differentiable.numerics_jax import (
    periodic_stencils, lax_friedrichs_flux, rollout,
    rollout_with_positivity_penalty,
)
from src.differentiable.euler_jax import (
    GAMMA, get_euler_flux_jax, euler_characteristic_state_stencils,
    euler_positivity_repair, positivity_penalty_jax,
)

ARCHITECTURE = (5, 64, 32, 16, 8, 3)
GRIDS = (64, 128, 256)
SCALAR_ADVECTION = ('constant', 'square', 'fourier', 'gaussian')
SCALAR_BURGERS = ('burgers_smooth', 'burgers_top_hat')
FAMILY_PROBS = {
    'scalar': {
        'constant': .04, 'square': .22, 'fourier': .18,
        'gaussian': .12, 'burgers_smooth': .22, 'burgers_top_hat': .22,
    },
    'euler': {
        'constant': .08, 'contact': .18, 'random_riemann': .38,
        'sod_like': .19, 'two_rarefactions': .17,
    },
}


def decode_strings(x):
    return np.array([v.decode('utf8') if isinstance(v, bytes) else str(v) for v in x])


def resolve_data_root(path):
    path = Path(path).expanduser().resolve()
    if (path / 'data' / 'scalar_train.h5').is_file():
        return path / 'data'
    if (path / 'scalar_train.h5').is_file():
        return path
    raise FileNotFoundError(
        f'Dataset not found at {path}; expected scalar_train.h5 or data/scalar_train.h5'
    )


def verify_file_hash(root, filename, report):
    expected = report['files'][filename]['sha256']
    digest = hashlib.sha256()
    with (root / filename).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f'Dataset SHA-256 mismatch: {filename}')


def load_split(root, physics, split, verify_hash=True):
    filename = f'{physics}_{split}.h5'
    report_path = root / 'quality_report.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'Missing quality manifest: {report_path}')
    report = json.loads(report_path.read_text())
    if verify_hash:
        verify_file_hash(root, filename, report)
    result = {}
    all_ids = set()
    with h5py.File(root / filename, 'r') as h5:
        if h5.attrs['physics'] != physics or h5.attrs['split'] != split:
            raise ValueError('HDF5 physics/split metadata mismatch')
        for n in GRIDS:
            name = f'N{n:03d}'
            if name not in h5:
                raise ValueError(f'Missing grid {name} in {filename}')
            g = h5[name]
            steps = int(g.attrs['short_steps'])
            if steps != 15:
                raise ValueError(f'Expected exactly K=15, got {steps} in {name}')
            # Read exact-cell-average states and their short-horizon target only.
            initial = np.asarray(g['states'][:, 0], dtype=np.float64)
            target = np.asarray(g['short_target'][:], dtype=np.float64)
            dt = np.asarray(g['short_dt'][:], dtype=np.float64)
            endtime = np.asarray(g['short_time'][:], dtype=np.float64)
            speed = np.asarray(g['advection_speed'][:], dtype=np.float64)
            family = decode_strings(g['family'][:])
            case_id = decode_strings(g['case_id'][:])
            expected_shape = (len(dt), n) if physics == 'scalar' else (len(dt), n, 3)
            if initial.shape != expected_shape or target.shape != expected_shape:
                raise ValueError(f'Invalid shape in {filename}/{name}')
            if not (len(dt) == len(speed) == len(family) == len(case_id)):
                raise ValueError('Inconsistent dataset metadata lengths')
            if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(target)):
                raise ValueError('Nonfinite reference or initial state')
            if not (np.all(dt > 0) and np.allclose(endtime, steps * dt, rtol=0, atol=1e-14)):
                raise ValueError('Incorrect case-specific timestep or horizon')
            for f in np.unique(family):
                if f not in FAMILY_PROBS[physics]:
                    raise ValueError(f'Unknown PDE family {f}')
            if physics == 'scalar':
                if np.any(np.isin(family, SCALAR_ADVECTION) & (speed <= 0)):
                    raise ValueError('Invalid advection speed')
                if np.any(np.isin(family, SCALAR_BURGERS) & (speed != 0)):
                    raise ValueError('Burgers examples must not have advection-speed metadata')
            else:
                for states in (initial, target):
                    rho = states[..., 0]
                    p = (GAMMA-1.)*(states[..., 2] - .5*states[..., 1]**2/rho)
                    if np.any(rho <= 0) or np.any(p <= 0) or not np.isfinite(p).all():
                        raise ValueError('Invalid Euler reference density/pressure')
            if len(set(case_id)) != len(case_id) or all_ids.intersection(case_id):
                raise ValueError('Duplicate case ID within a split')
            all_ids.update(case_id)
            by_family = {f: np.flatnonzero(family == f) for f in np.unique(family)}
            result[n] = dict(initial=initial, target=target, dt=dt, speed=speed,
                             family=family, ids=case_id, indices=by_family, steps=steps)
    if sum(len(g['ids']) for g in result.values()) != report['counts'][f'{physics}_{split}']:
        raise RuntimeError('Dataset count differs from quality manifest')
    return result, all_ids


def make_reconstruct(params, strength, neural=True):
    if not neural:
        return teno5_reconstruct_jax
    return lambda stencil: gated_nn_teno5_reconstruct(
        params, stencil, mode='soft', strength=strength)


def scalar_rollout(params, u0, dt, speed, n, family, strength, neural=True):
    rec = make_reconstruct(params, strength, neural)
    if family in SCALAR_ADVECTION:
        # Conservatively reconstructed Lax-Friedrichs flux with per-case a.
        flux = lambda u: lax_friedrichs_flux(u, lambda z: speed*z, jnp.abs(speed), rec)
    elif family in SCALAR_BURGERS:
        # Positive global LF bound recalculated at each SSPRK stage.
        flux = lambda u: lax_friedrichs_flux(
            u, lambda z: .5*z*z, jnp.maximum(jnp.max(jnp.abs(u)), .01), rec)
    else:
        raise ValueError(f'Unknown scalar family: {family}')
    return rollout(u0, flux, dt, 1. / n, 15)


def euler_rollout(params, u0, dt, strength, neural=True, train_guard=False):
    rec = make_reconstruct(params, strength, neural)
    flux = get_euler_flux_jax(rec, boundary='transmissive')
    if train_guard:
        final, p = rollout_with_positivity_penalty(
            u0, flux, dt, 1. / u0.shape[0], 15,
            state_guard=euler_positivity_repair,
            penalty_fn=positivity_penalty_jax,
        )
        return final, p
    # Intentionally unguarded during validation/test.
    return rollout(u0, flux, dt, 1. / u0.shape[0], 15)


def prediction_fn(physics, n, family, strength, neural):
    @jax.jit
    def predict(params, states, dt, speed):
        if physics == 'scalar':
            return jax.vmap(lambda u, t, a: scalar_rollout(
                params, u, t, a, n, family, strength, neural))(states, dt, speed)
        return jax.vmap(lambda u, t: euler_rollout(
            params, u, t, strength, neural, False))(states, dt)
    return predict


def training_step_fn(physics, n, family, strength, lr, tv_weight, pos_weight):
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(lr))

    def loss_fn(params, states, target, dt, speed):
        if physics == 'scalar':
            predicted = jax.vmap(lambda u, t, a: scalar_rollout(
                params, u, t, a, n, family, strength, True))(states, dt, speed)
            scales = jnp.maximum(jnp.mean(target*target, axis=-1), .05)
            mse = jnp.mean(jnp.mean((predicted-target)**2, axis=-1) / scales)
            tv_pred = jnp.sum(jnp.abs(predicted-jnp.roll(predicted,1,axis=1)),axis=1)
            tv_init = jnp.sum(jnp.abs(states-jnp.roll(states,1,axis=1)),axis=1)
            tv = jnp.mean(jnp.maximum(tv_pred-tv_init,0.)) / n
            return mse + tv_weight*tv, (mse, tv, jnp.array(0.))
        finals, penalties = jax.vmap(lambda u,t: euler_rollout(
            params, u, t, strength, True, True))(states, dt)
        scales = jnp.maximum(jnp.mean(target*target,axis=(0,1)), .05)
        mse = jnp.mean((finals-target)**2/scales)
        penalty = jnp.mean(penalties)/15.
        return mse + pos_weight*penalty, (mse, jnp.array(0.), penalty)

    @jax.jit
    def step(params, optimizer_state, states, target, dt, speed):
        (value, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params, states, target, dt, speed)
        updates, state = optimizer.update(grad, optimizer_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, state, value, metrics
    return optimizer, step


def training_pool(data, physics, rng, per_family, max_pool):
    """Teacher-label warm-up uses only *training* examples' real PDE stencils."""
    sampled = []
    for n, group in data.items():
        for family, ids in group['indices'].items():
            selected = rng.choice(ids, size=min(per_family, len(ids)), replace=False)
            for j in selected:
                sampled.append((n, family, group['initial'][j]))
    all_stencils = []
    for n in GRIDS:
        selected = [s for N, _, s in sampled if N == n]
        if not selected:
            continue
        chunk = 8
        for lo in range(0, len(selected), chunk):
            x = jnp.asarray(np.stack(selected[lo:lo+chunk]))
            if physics == 'scalar':
                stencils = jax.vmap(periodic_stencils)(x)
            else:
                stencils = jax.vmap(
                    lambda u: euler_characteristic_state_stencils(
                        u, boundary='transmissive'))(x)
            all_stencils.append(np.asarray(stencils).reshape(-1,5))
    pool = np.concatenate(all_stencils)
    if len(pool) > max_pool:
        pool = pool[rng.choice(len(pool),size=max_pool,replace=False)]
    chi = np.asarray(teno5_selection_values(jnp.asarray(pool)))
    target = (chi >= DEFAULT_CUTOFF).astype(np.float64)
    troubled = np.flatnonzero(~np.all(target == 1., axis=1))
    smooth = np.flatnonzero(np.all(target == 1., axis=1))
    if not len(troubled):
        raise RuntimeError('Pretraining pool has no troubled interfaces')
    if len(smooth):
        k = min(len(troubled), len(smooth))
        chosen = np.concatenate((rng.choice(troubled,k,replace=False),
                                 rng.choice(smooth,k,replace=False)))
        pool, target = pool[chosen], target[chosen]
    return jnp.asarray(pool), jnp.asarray(target), len(troubled)


def warmup_train(params, dataset, labels, rng, steps, lr, batch_size):
    opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(lr))
    state = opt.init(params)

    @jax.jit
    def single_step(p, s, x, y):
        def loss_fn(q):
            keep = jnp.clip(toy_keep_weight(q,x),1e-7,1.-1e-7)
            bce = -y*jnp.log(keep)-(1.-y)*jnp.log(1.-keep)
            # Mild weighting counters the dominance of all-keep decisions.
            w = jnp.where(y<.5,2.,1.)
            return jnp.mean(bce*w)
        value, grad = jax.value_and_grad(loss_fn)(p)
        update,s=opt.update(grad,s,p)
        return optax.apply_updates(p,update), s, value

    for step in range(steps):
        chosen = rng.integers(0,len(dataset),size=batch_size)
        params,state,loss = single_step(params,state,dataset[chosen],labels[chosen])
        if step == 0 or (step+1)%50==0 or step == steps-1:
            print(f'Warm-up {step+1:4d}/{steps} | weighted BCE={float(loss):.6g}', flush=True)
        if not np.isfinite(float(loss)):
            raise FloatingPointError('Warm-up diverged')
    return params


def eval_state_metrics(predicted, reference, physics):
    if not np.isfinite(predicted).all():
        return None
    if physics == 'euler':
        rho = predicted[...,0]
        if np.any(rho<=0):
            return None
        pressure = (GAMMA-1.)*(predicted[...,2]-.5*predicted[...,1]**2/rho)
        if not np.isfinite(pressure).all() or np.any(pressure<=0):
            return None
    return float(np.mean(np.abs(predicted-reference)))


def validation_cases(data, physics, take, rng):
    """Frozen validation subset: same cases reused at every checkpoint check."""
    groups=[]
    for n, bucket in data.items():
        for family, indices in sorted(bucket['indices'].items()):
            selected = indices if take == 0 else rng.choice(indices, size=min(take, len(indices)),replace=False)
            groups.append(dict(n=n, family=family,
                               u=jnp.asarray(bucket['initial'][selected]),
                               ref=np.asarray(bucket['target'][selected]),
                               dt=jnp.asarray(bucket['dt'][selected]),
                               speed=jnp.asarray(bucket['speed'][selected]),
                               ids=bucket['ids'][selected].tolist()))
    return groups


def predict_chunks(fn, params, case, chunk_size=16):
    """Bound peak JAX/XLA memory while testing whole held-out split."""
    size=len(case['ids'])
    predictions=[]
    for left in range(0,size,chunk_size):
        right=min(size,left+chunk_size)
        predictions.append(np.asarray(fn(
            params, case['u'][left:right], case['dt'][left:right],
            case['speed'][left:right])))
    return np.concatenate(predictions,axis=0)


def score_validation(params, cases, physics, strength, predictors=None, baselines=None, eval_batch_size=16):
    scores=[]
    by_family={}
    failures=[]
    if predictors is None:
        predictors={(c['n'],c['family']):prediction_fn(physics,c['n'],c['family'],strength,True)
                    for c in cases}
    for c in cases:
        key=(c['n'],c['family'])
        pred=predict_chunks(predictors[key],params,c,eval_batch_size)
        error=eval_state_metrics(pred,c['ref'],physics)
        if error is None:
            failures.extend(c['ids'])
            continue
        base_error = 1. if baselines is None else baselines[key]
        # Relative to deterministic TENO5 to compare different N and fields.
        score=(error+1e-8)/(base_error+1e-8)
        scores.append(score)
        by_family.setdefault(c['family'],[]).append(score)
    valid = not failures and len(scores)==len(cases) and np.isfinite(scores).all()
    return dict(valid=bool(valid),relative_score=float(np.mean(scores)) if valid else 1e30,
                family_ratios={f:float(np.mean(x)) for f,x in by_family.items()},
                failures=failures)


def baseline_validation(cases,physics,strength,eval_batch_size=16):
    baseline={}
    for c in cases:
        key=(c['n'],c['family'])
        pred=predict_chunks(prediction_fn(physics,c['n'],c['family'],strength,False),
                            None,c,eval_batch_size)
        error=eval_state_metrics(pred,c['ref'],physics)
        if error is None:
            raise RuntimeError(f'Deterministic TENO baseline invalid on {key}')
        baseline[key]=error
    return baseline


def freeze_checkpoint(params, filename, meta):
    filename.parent.mkdir(parents=True,exist_ok=True)
    ckpt = [(np.asarray(w,dtype=np.float64),np.asarray(b,dtype=np.float64)) for w,b in params]
    with filename.with_suffix('.tmp').open('wb') as stream:
        pickle.dump(ckpt,stream,pickle.HIGHEST_PROTOCOL)
    filename.with_suffix('.tmp').replace(filename)
    meta_path=filename.with_suffix('.json')
    meta_path.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')


def train(args, root):
    physics=args.physics
    train_data,train_ids = load_split(root,physics,'train',not args.skip_hash_check)
    valid_data,valid_ids = load_split(root,physics,'validation',not args.skip_hash_check)
    if train_ids.intersection(valid_ids):
        raise RuntimeError('Training/validation contamination detected')
    out = Path(args.output_dir).expanduser()
    if not out.is_absolute():
        out=ROOT/out
    run=args.run_name or f'{physics}_gated_dataset_v2_seed{args.seed}'
    checkpoint=out/f'{run}.pkl'
    if checkpoint.exists() or checkpoint.with_suffix('.json').exists():
        raise FileExistsError(f'Will not overwrite historical or existing checkpoint: {checkpoint}')
    strength=args.strength if args.strength is not None else (.75 if physics=='scalar' else .25)
    rng=np.random.default_rng(args.seed)
    validation=validation_cases(valid_data,physics,args.validation_per_family,np.random.default_rng(args.seed+777))
    print(f'Physics={physics} | architecture={ARCHITECTURE} | strength={strength} | '+
          f'cases={len(train_ids)} train / {len(valid_ids)} validation | K=15',flush=True)
    print(f'Validation groups={len(validation)} (each family at each resolution)',flush=True)
    baselines=baseline_validation(validation,physics,strength)
    print(f'TENO5 validation baseline { {str(k):v for k,v in baselines.items()} }',flush=True)
    params=init_toy_params(jax.random.PRNGKey(args.seed),hidden_sizes=ARCHITECTURE[1:-1])
    pool,labels,troubled_count=training_pool(train_data,physics,rng,args.pretrain_cases,args.pretrain_pool)
    print(f'Warm-up pool: {len(pool)} stencils; {troubled_count} troubled in source pool',flush=True)
    params=warmup_train(params,pool,labels,rng,args.pretrain_steps,args.learning_rate,args.pretrain_batch)

    predictors={(v['n'],v['family']):prediction_fn(physics,v['n'],v['family'],strength,True)
                for v in validation}
    best=score_validation(params,validation,physics,strength,predictors,baselines)
    if not best['valid']:
        raise RuntimeError(f'Warm-up produced invalid unguarded validation: {best}')
    print(f'Initial relative validation score: {best["relative_score"]:.6g}',flush=True)

    optimizer=optax.chain(optax.clip_by_global_norm(1.0),optax.adam(args.learning_rate))
    opt_state=optimizer.init(params)
    methods={(n,f):training_step_fn(physics,n,f,strength,args.learning_rate,
                                   args.tv_weight,args.positivity_weight)[1]
             for n in GRIDS for f in FAMILY_PROBS[physics]}
    # 1 optimizer state shared across all groups; the Optax transform is identical.
    combinations=[(n,f) for n in GRIDS for f in FAMILY_PROBS[physics]]
    weights=np.array([FAMILY_PROBS[physics][f] for n,f in combinations],dtype=float)
    weights /= weights.sum()
    start=time.perf_counter()
    history=[]
    best_step=-1
    patience_left=args.patience
    base_meta=dict(physics=physics,architecture=list(ARCHITECTURE),checkpoint_schema='backbone',
                   data_root=str(root),dataset_quality_sha256=hashlib.sha256((root/'quality_report.json').read_bytes()).hexdigest(),
                   physical_measurements=False,reference='analytic cell averages',
                   seed=args.seed,strength=strength,training_steps=15,
                   train_trajectories=len(train_ids),validation_trajectories=len(valid_ids),
                   train_resolutions=list(GRIDS),validation_case_ids=[id for c in validation for id in c['ids']],
                   training_config=vars(args),selection='ungarded validation vs matched deterministic TENO5',
                   limitations=['No independent real experimental measurement labels',
                                'Euler training positivity repairs may be nonconservative',
                                'No universal fifth-order/stability guarantee'])

    def save_best(current_params, current_score, step):
        metadata=dict(base_meta,best_step=step,best_score=current_score,
                      history=history,time_elapsed_s=time.perf_counter()-start)
        # mkdir after first finite validation; intentionally exclusive per run name.
        freeze_checkpoint(current_params,checkpoint,metadata)

    save_best(params,best,-1)
    print(f'Initial checkpoint saved: {checkpoint}',flush=True)
    for it in range(args.updates):
        group_index=int(rng.choice(len(combinations),p=weights))
        n,family=combinations[group_index]
        bucket=train_data[n]
        ix=bucket['indices'][family]
        selected=rng.choice(ix,size=args.batch_size,replace=len(ix)<args.batch_size)
        u=jnp.asarray(bucket['initial'][selected]); ref=jnp.asarray(bucket['target'][selected])
        dt=jnp.asarray(bucket['dt'][selected]); speed=jnp.asarray(bucket['speed'][selected])
        params,opt_state,loss,details=methods[(n,family)](params,opt_state,u,ref,dt,speed)
        loss_val=float(loss)
        if not np.isfinite(loss_val) or not all(np.isfinite(np.asarray(x)).all() for pair in params for x in pair):
            print(f'Nonfinite update at step {it}: restoring last best model',flush=True)
            break
        if it % args.check_every == 0 or it == args.updates - 1:
            score=score_validation(params,validation,physics,strength,predictors,baselines)
            record=dict(step=it,grid=n,family=family,train_loss=loss_val,
                        state_loss=float(details[0]),
                        tv_penalty=float(details[1]),positivity_penalty=float(details[2]),
                        validation=score)
            history.append(record)
            print(f'Step {it:5d} | train={loss_val:.6g} | '+
                  f'val_ratio={score["relative_score"]:.6g} | '+
                  f'valid={score["valid"]} | '+
                  f'families={score["family_ratios"]}',flush=True)
            if score['valid'] and score['relative_score'] < best['relative_score'] - args.improvement_tol:
                best, best_step=score,it
                patience_left=args.patience
                save_best(params,best,it)
                print('  -> New best checkpoint saved',flush=True)
            else:
                patience_left-=1
                if patience_left<=0:
                    print('Early stopping: validation score did not improve',flush=True)
                    break
    print(f'Done | best step={best_step} | val ratio={best["relative_score"]:.6g} '
          f'| checkpoint={checkpoint}',flush=True)
    # Expose every check in a separate history artifact. No use of test split.
    history_path=checkpoint.with_suffix('.history.json')
    history_path.write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')


def test(args, root):
    if not args.checkpoint:
        raise ValueError('--mode test requires --checkpoint PATH')
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with checkpoint.open('rb') as stream:
        loaded=pickle.load(stream)  # Only load your own trusted checkpoints.
    if len(loaded)!=5:
        raise ValueError('Expected existing 5-layer NN-TENO5 backbone')
    params=[(jnp.asarray(w,dtype=jnp.float64),jnp.asarray(b,dtype=jnp.float64)) for w,b in loaded]
    meta=json.loads(checkpoint.with_suffix('.json').read_text())
    physics=meta['physics']
    strength=float(meta['strength'])
    if physics!=args.physics:
        raise ValueError('Checkpoint physics mismatch')
    testing,ids=load_split(root,physics,'test',not args.skip_hash_check)
    cases=validation_cases(testing,physics,args.test_per_family,np.random.default_rng(args.seed+999))
    baselines=baseline_validation(cases,physics,strength,args.eval_batch_size)
    score=score_validation(params,cases,physics,strength,baselines=baselines,
                           eval_batch_size=args.eval_batch_size)
    print('UNTOUCHED TEST-SPLIT RESULT (never used during training/selection)',flush=True)
    print(json.dumps(score,indent=2,allow_nan=False))
    report_path=checkpoint.with_suffix('.test.json')
    if report_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing test report: {report_path}')
    report_path.write_text(json.dumps(dict(checkpoint=str(checkpoint),
                              tested_cases=len(cases),tested_ids=[id for c in cases for id in c['ids']],
                              data_root=str(root),result=score),indent=2,allow_nan=False)+'\n')
    print('Saved held-out test report:',report_path,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['train','test'],default='train')
    ap.add_argument('--physics',choices=['scalar','euler'],required=True)
    ap.add_argument('--data-root',default=str(ROOT/'data'/'nnteno_v2'))
    ap.add_argument('--skip-hash-check',action='store_true',help='Not recommended: skip dataset SHA-256 verification')
    ap.add_argument('--output-dir',default='models/v2')
    ap.add_argument('--run-name',default=None)
    ap.add_argument('--checkpoint',default=None,help='Required for --mode test')
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--strength',type=float,default=None,help='Fixed lambda; default .75 scalar/.25 Euler')
    ap.add_argument('--updates',type=int,default=300)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--learning-rate',type=float,default=2e-4)
    ap.add_argument('--tv-weight',type=float,default=.05)
    ap.add_argument('--positivity-weight',type=float,default=.1)
    ap.add_argument('--pretrain-steps',type=int,default=150)
    ap.add_argument('--pretrain-cases',type=int,default=12)
    ap.add_argument('--pretrain-pool',type=int,default=15000)
    ap.add_argument('--pretrain-batch',type=int,default=256)
    ap.add_argument('--validation-per-family',type=int,default=2)
    ap.add_argument('--test-per-family',type=int,default=0,
                    help='0 means ALL test cases from every family and resolution')
    ap.add_argument('--check-every',type=int,default=20)
    ap.add_argument('--eval-batch-size',type=int,default=16)
    ap.add_argument('--patience',type=int,default=8)
    ap.add_argument('--improvement-tol',type=float,default=1e-5)
    args=ap.parse_args()
    if (not 0<args.learning_rate or not np.isfinite(args.learning_rate)
        or any(getattr(args,key)<=0 for key in (
            'batch_size','pretrain_batch','pretrain_cases','pretrain_pool',
            'validation_per_family','check_every','patience','eval_batch_size'))
        or args.updates<0 or args.pretrain_steps<0 or args.test_per_family<0):
        ap.error('Invalid training sizes or learning rate')
    if args.strength is not None and not (0 < args.strength <=1.):
        ap.error('Strength must be in (0,1]')
    root=resolve_data_root(args.data_root)
    if args.mode=='train':
        train(args,root)
    else:
        test(args,root)


if __name__=='__main__':
    main()
