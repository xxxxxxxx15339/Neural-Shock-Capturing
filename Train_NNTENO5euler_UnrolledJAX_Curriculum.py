# -*- coding: utf-8 -*-
"""Literature-motivated fix for the Euler training instability
(Train_NNTENO5euler_UnrolledJAX.py, and PROJ_IMPLEMENTATION_GUIDE.md's
Phase 6/7 status note): curriculum learning on rollout length K.

Um et al. ("Solver-in-the-Loop", NeurIPS 2020) report the exact failure
this project hit -- "performing time integration via the PDE then typically
leads to exponential increases of existing oscillations and a diverging
calculation" -- and give a specific, tested mitigation: "we found it
important to pre-train networks with small look-aheads, and then continue
training with longer recurrent iterations for the look-ahead." That
mitigation was never tried for the Euler run in this project -- the
original attempt trained straight at K=15. This script tests it directly:
same data, same network, same loss, same optimizer, staged as
K = 2 -> 4 -> 8 -> 15 instead of jumping straight to 15, carrying the Adam
optimizer state across stages (not resetting it) so this is a genuine
curriculum, not four independent training runs.

Everything else (sampling, loss, positivity guard, evaluation) is reused
unmodified from Train_NNTENO5euler_UnrolledJAX.py -- this script isolates
the one variable the literature review flagged.

Run inside `teno-jax` (needs run3_pretrained's weights to exist):

    conda run -n teno-jax python -u Train_NNTENO5euler_UnrolledJAX_Curriculum.py
"""

import functools
import pickle
import time

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import optax

import Train_NNTENO5euler_UnrolledJAX as base

CURRICULUM = [
    (2, 60),
    (4, 60),
    (8, 60),
    (15, 80),
]


def main():
    with open('models/nn_teno5_unrolled_jax_params_run3_pretrained.pkl', 'rb') as f:
        loaded = pickle.load(f)
    params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded]
    anchor_params = params
    print('Loaded pretrained scalar-run weights as Euler warm start/anchor.')
    print('Curriculum: K = {}'.format([k for k, _ in CURRICULUM]))

    optimizer = optax.adam(1.0e-3)
    opt_state = optimizer.init(params)

    rng = np.random.default_rng(2)
    started = time.perf_counter()
    global_step = 0

    for stage_index, (steps_k, n_steps) in enumerate(CURRICULUM):

        @functools.partial(jax.jit, static_argnames=('steps_k',))
        def train_step(params, opt_state, batch_u0, batch_exact, steps_k):
            (loss, breakdown), grads = jax.value_and_grad(base.batched_loss, has_aux=True)(
                params, batch_u0, batch_exact, anchor_params, steps_k
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, breakdown

        print('\n--- Stage {}: K={}, {} steps ---'.format(stage_index, steps_k, n_steps))
        stage_started = time.perf_counter()
        diverged = False
        for local_step in range(n_steps):
            batch_u0, batch_exact = base.sample_batch(rng, 8, steps_k)
            params, opt_state, loss, breakdown = train_step(
                params, opt_state, batch_u0, batch_exact, steps_k
            )
            if local_step == 0:
                print('  first (compiling) step took {:.1f}s'.format(time.perf_counter() - stage_started))
            if not np.isfinite(float(loss)):
                print('  step {:4d}: loss went non-finite -- stage diverged'.format(local_step))
                diverged = True
                break
            if local_step % 20 == 0 or local_step == n_steps - 1:
                print('  step {:4d}: l_total={:.6e} l_mse={:.6e} l_tv={:.6e} l_bvd={:.6e} l_reg={:.6e} l_pos={:.6e}'.format(
                    local_step, breakdown['l_total'], breakdown['l_mse'], breakdown['l_tv'],
                    breakdown['l_bvd'], breakdown['l_reg'], breakdown['l_positivity'],
                ))
            global_step += 1
        if diverged:
            print('Curriculum stopped early at K={} after divergence.'.format(steps_k))
            break

    print('\nTotal training time: {:.1f}s ({} steps)'.format(time.perf_counter() - started, global_step))

    with open('models/nn_teno5euler_unrolled_jax_params_curriculum.pkl', 'wb') as f:
        pickle.dump([(np.asarray(w), np.asarray(b)) for w, b in params], f)

    print('\n' + '=' * 70)
    print('Held-out evaluation vs. deterministic TENO5euler (same cases as the')
    print('non-curriculum run, for a direct comparison)')
    print('=' * 70)

    eval_rng = np.random.default_rng(999)
    results = []

    long_steps = base.TRAIN_STEPS_K * 3
    riemann_params, discontinuity = base.sample_riemann_problem(eval_rng, steps_k=long_steps)
    u0_a = base.riemann_state_at(riemann_params, discontinuity, 1.0e-6)
    exact_a = base.riemann_state_at(riemann_params, discontinuity, base.TRAIN_STEPS_K * base.DT)
    results.append(base.evaluate_case(params, 'held-out Riemann, K steps', u0_a, exact_a, base.TRAIN_STEPS_K))

    exact_b = base.riemann_state_at(riemann_params, discontinuity, long_steps * base.DT)
    results.append(base.evaluate_case(params, 'held-out Riemann, 3x K steps', u0_a, exact_b, long_steps))

    u0_sod = jnp.asarray(base.sod()(np.asarray(base.X)))
    exact_sod = jnp.asarray(base.exact_sod_solution(np.asarray(base.X), base.TRAIN_STEPS_K * base.DT))
    results.append(base.evaluate_case(params, 'canonical Sod, K steps', u0_sod, exact_sod, base.TRAIN_STEPS_K))

    beats_count = sum(1 for beats, _, _ in results if beats)
    print('\nSummary: curriculum-trained NN-TENO5euler(JAX) beats deterministic TENO5 '
          'on {}/{} held-out cases.'.format(beats_count, len(results)))


if __name__ == '__main__':
    main()
