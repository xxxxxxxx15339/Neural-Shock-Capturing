# -*- coding: utf-8 -*-
"""Fine-tunes the best existing Euler checkpoint (Phase 14/15's
`..._curriculum_phase15_smoothmix.pkl`) THROUGH the convex-mix
reconstruction (Results.md Phase 16), not just applying the mix at
inference the way Phase 16 first verified it.

Train_NNTENO5euler_UnrolledJAX.py's `single_example_loss`/`network_rollout`
were updated (Phase 17) to route through
`projection_jax.mixed_reconstruct` instead of the network's raw candidate
combination -- this script is what actually exercises that during training,
starting from an already-stable, already-converged-at-K=15 checkpoint
rather than restarting the whole K=2->4->8->15 curriculum from a cold
scalar-pretrained start. The curriculum's staged ramp-up existed for
stability starting cold; a checkpoint that already trained cleanly through
K=15 once doesn't need re-staging to keep training at K=15.

Run inside `teno-jax` (needs the Phase 15 checkpoint to exist):

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX_MixFinetune.py
"""

import os
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import pickle
import time

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import optax

import Train_NNTENO5euler_UnrolledJAX as base

WARM_START_PATH = os.path.join(_REPO_ROOT, 'models', 'nn_teno5euler_unrolled_jax_params_curriculum_phase15_smoothmix.pkl')
OUTPUT_PATH = os.path.join(_REPO_ROOT, 'models', 'nn_teno5euler_unrolled_jax_params_curriculum_phase17_mixtrained.pkl')
N_STEPS = 150
STEPS_K = 15


def main():
    with open(WARM_START_PATH, 'rb') as f:
        loaded = pickle.load(f)
    params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded]
    anchor_params = params
    print('Loaded Phase 15 checkpoint as warm start/anchor for mix fine-tuning.')

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0e-4))
    opt_state = optimizer.init(params)

    # Track the best-by-validation checkpoint, not just whatever's left at
    # the final step: a first attempt at this fine-tune hit a real loss
    # spike mid-run (to 156 at step 60) that the network partially but not
    # fully recovered from by step 149, and the FINAL params -- not the
    # best-seen ones -- were what got saved and evaluated, silently taking
    # a mid-recovery checkpoint rather than the run's actual best point.
    validation_x = np.arange(base.CELLS, dtype=float) / base.CELLS
    validation_u0 = base.riemann_state_at((1.0, 0.0, 1.0, 0.125, 0.0, 0.1), 0.5, 1.0e-6)
    validation_exact = jnp.asarray(base.exact_sod_solution(validation_x, STEPS_K * base.DT))
    best_params = params
    best_error = float('inf')

    @jax.jit
    def train_step(params, opt_state, batch_u0, batch_exact):
        (loss, breakdown), grads = jax.value_and_grad(base.batched_loss, has_aux=True)(
            params, batch_u0, batch_exact, anchor_params, STEPS_K
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, breakdown

    rng = np.random.default_rng(3)
    started = time.perf_counter()
    for step in range(N_STEPS):
        # smooth_fraction=0.5 matches base.sample_batch's own default (set
        # in Phase 14 and unchanged since) -- passed explicitly so this
        # script's data mix is self-documenting, not implicit in a default
        # elsewhere.
        batch_u0, batch_exact = base.sample_batch(rng, 8, STEPS_K, smooth_fraction=0.5)
        params, opt_state, loss, breakdown = train_step(params, opt_state, batch_u0, batch_exact)
        if step == 0:
            print('first (compiling) step took {:.1f}s'.format(time.perf_counter() - started))
        if not np.isfinite(float(loss)):
            print('step {:4d}: loss went non-finite -- stopping'.format(step))
            break
        if step % 20 == 0 or step == N_STEPS - 1:
            print('step {:4d}: l_total={:.6e} l_mse={:.6e} l_tv={:.6e} l_bvd={:.6e} l_reg={:.6e} l_pos={:.6e}'.format(
                step, breakdown['l_total'], breakdown['l_mse'], breakdown['l_tv'],
                breakdown['l_bvd'], breakdown['l_reg'], breakdown['l_positivity'],
            ))
        if step % 10 == 0 or step == N_STEPS - 1:
            nn_final = base.network_rollout(params, validation_u0, STEPS_K)
            validation_error = base.l2_error(nn_final, validation_exact)
            if np.isfinite(validation_error) and validation_error < best_error:
                best_error = validation_error
                best_params = params

    print('\nTotal fine-tuning time: {:.1f}s'.format(time.perf_counter() - started))
    print('Best validation (Sod, K steps) L2 seen during fine-tuning: {:.6e}'.format(best_error))
    params = best_params

    all_finite = all(
        bool(np.isfinite(np.asarray(w)).all()) and bool(np.isfinite(np.asarray(b)).all())
        for w, b in params
    )
    print('Final checkpoint finite: {}'.format(all_finite))

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump([(np.asarray(w), np.asarray(b)) for w, b in params], f)
    print('Trained parameters written to {}'.format(OUTPUT_PATH))

    print('\n' + '=' * 70)
    print('Held-out evaluation vs. deterministic TENO5euler')
    print('=' * 70)
    base.evaluate_case(params, 'held-out Riemann, K steps', *base.sample_riemann_case(np.random.default_rng(42), STEPS_K), STEPS_K)
    long_steps = STEPS_K * 3
    u0_long, exact_long = base.sample_riemann_case(np.random.default_rng(42), long_steps)
    base.evaluate_case(params, 'held-out Riemann, 3x K steps', u0_long, exact_long, long_steps)
    sod_x = np.arange(base.CELLS, dtype=float) / base.CELLS
    u0_sod = base.riemann_state_at((1.0, 0.0, 1.0, 0.125, 0.0, 0.1), 0.5, 1.0e-6)
    exact_sod = jnp.asarray(base.exact_sod_solution(sod_x, STEPS_K * base.DT))
    base.evaluate_case(params, 'canonical Sod, K steps', u0_sod, exact_sod, STEPS_K)


if __name__ == '__main__':
    main()
