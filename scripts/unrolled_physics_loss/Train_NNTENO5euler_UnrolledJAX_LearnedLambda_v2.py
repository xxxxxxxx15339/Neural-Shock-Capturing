# -*- coding: utf-8 -*-
"""Results.md Phase 20: continues Phase 18's Euler mixing-head training with
the two things it flagged as not-yet-tried, targeting specifically why Sod
(unlike Shu-Osher, and unlike Phase 19's scalar Advection/Burgers) never
crossed from a near-tie into an outright win.

Phase 19 (scalar) found the SAME dedicated-head mechanism gave a free win on
BOTH benchmarks before any training, from a small uniform mix. Checking the
same "before training" point for Euler (this phase) found an asymmetry
Phase 18 never isolated: Shu-Osher was ALREADY a win pre-training (0.9804x,
matching Phase 18's trained 0.982x almost exactly -- training barely moved
it), but Sod started as a LOSS (1.0067x) and Phase 18's training only
partially recovered it (to 1.0035x), never crossing to a win. That is a
genuine, harder, Sod-specific weakness a uniform blend doesn't fix on its
own -- worth more training signal specifically for lambda, not more of the
same recipe.

Two changes from Phase 18, both flagged there as the concrete next step:
1. Per-parameter-group learning rates via `optax.multi_transform`: the
   backbone (already well-trained through Phases 13-15) gets a LOWER rate
   than Phase 18 used (1e-4, not 1e-3) so it moves less and doesn't risk
   losing what Shu-Osher already has; the lambda head gets a HIGHER rate
   (5e-3, 5x Phase 18's shared 1e-3) since Phase 18's own diagnosis was that
   its gradient signal was diluted by clipping and the L_Reg pull toward its
   own conservative init, and it "barely moved" (bias -3.000 -> -3.015 over
   200 steps).
2. More steps (400, not 200) and continuing FROM Phase 18's checkpoint
   (already ahead of the untrained baseline on average) rather than
   restarting from the untrained head.

Run inside `teno-jax` (needs Phase 18's checkpoint to exist):

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX_LearnedLambda_v2.py
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
import Train_NNTENO5euler_UnrolledJAX_LearnedLambda as v1

WARM_START_PATH = os.path.join(_REPO_ROOT, 'models', 'nn_teno5euler_unrolled_jax_params_curriculum_phase18_learnedlambda.pkl')
OUTPUT_PATH = os.path.join(_REPO_ROOT, 'models', 'nn_teno5euler_unrolled_jax_params_curriculum_phase20_learnedlambda_v2.pkl')
N_STEPS = 400
STEPS_K = 15

BACKBONE_LR = 1.0e-4
LAMBDA_LR = 5.0e-3


def main():
    with open(WARM_START_PATH, 'rb') as f:
        loaded = pickle.load(f)
    backbone_params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded['backbone']]
    lambda_params = (jnp.asarray(loaded['lambda_head'][0]), jnp.asarray(loaded['lambda_head'][1]))
    params = (backbone_params, lambda_params)
    anchor_params = params
    print('Loaded Phase 18 checkpoint (backbone + lambda head) as warm start/anchor.')
    print('Learning rates: backbone={:.1e}, lambda_head={:.1e}'.format(BACKBONE_LR, LAMBDA_LR))

    # Label each leaf 'backbone' or 'lambda' for optax.multi_transform --
    # params = (backbone_params, lambda_params), backbone_params a list of
    # (w, b) tuples, lambda_params a single (w, b) tuple.
    param_labels = (
        [('backbone', 'backbone') for _ in backbone_params],
        ('lambda', 'lambda'),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {'backbone': optax.adam(BACKBONE_LR), 'lambda': optax.adam(LAMBDA_LR)},
            param_labels,
        ),
    )
    opt_state = optimizer.init(params)

    validation_x = np.arange(base.CELLS, dtype=float) / base.CELLS
    validation_u0 = base.riemann_state_at((1.0, 0.0, 1.0, 0.125, 0.0, 0.1), 0.5, 1.0e-6)
    validation_exact = jnp.asarray(base.exact_sod_solution(validation_x, STEPS_K * base.DT))
    best_params = params
    best_error = float('inf')

    @jax.jit
    def train_step(params, opt_state, batch_u0, batch_exact):
        (loss, breakdown), grads = jax.value_and_grad(v1.batched_loss, has_aux=True)(
            params, batch_u0, batch_exact, anchor_params, STEPS_K
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, breakdown

    rng = np.random.default_rng(5)
    started = time.perf_counter()
    for step in range(N_STEPS):
        batch_u0, batch_exact = base.sample_batch(rng, 8, STEPS_K, smooth_fraction=0.5)
        params, opt_state, loss, breakdown = train_step(params, opt_state, batch_u0, batch_exact)
        if step == 0:
            print('first (compiling) step took {:.1f}s'.format(time.perf_counter() - started))
        if not np.isfinite(float(loss)):
            print('step {:4d}: loss went non-finite -- stopping'.format(step))
            break
        if step % 40 == 0 or step == N_STEPS - 1:
            lam_bias = float(params[1][1][0])
            print('step {:4d}: l_total={:.6e} l_mse={:.6e} l_pos={:.6e} lambda_bias={:.4f}'.format(
                step, breakdown['l_total'], breakdown['l_mse'], breakdown['l_positivity'], lam_bias,
            ))
        if step % 10 == 0 or step == N_STEPS - 1:
            nn_final = v1.network_rollout(params, validation_u0, STEPS_K)
            validation_error = base.l2_error(nn_final, validation_exact)
            if np.isfinite(validation_error) and validation_error < best_error:
                best_error = validation_error
                best_params = params

    print('\nTotal training time: {:.1f}s'.format(time.perf_counter() - started))
    print('Best validation (Sod, K steps) L2 seen during training: {:.6e}'.format(best_error))
    params = best_params
    backbone_params, lambda_params = params

    backbone_finite = all(
        bool(np.isfinite(np.asarray(w)).all()) and bool(np.isfinite(np.asarray(b)).all())
        for w, b in backbone_params
    )
    lambda_finite = bool(np.isfinite(np.asarray(lambda_params[0])).all()) and bool(np.isfinite(np.asarray(lambda_params[1])).all())
    print('Final checkpoint finite: backbone={} lambda_head={}'.format(backbone_finite, lambda_finite))

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'backbone': [(np.asarray(w), np.asarray(b)) for w, b in backbone_params],
            'lambda_head': (np.asarray(lambda_params[0]), np.asarray(lambda_params[1])),
        }, f)
    print('Trained parameters written to {}'.format(OUTPUT_PATH))
    print('final lambda head: w={} b={}'.format(np.asarray(lambda_params[0]).ravel(), np.asarray(lambda_params[1])))


if __name__ == '__main__':
    main()
