# -*- coding: utf-8 -*-
"""Results.md Phase 18: a dedicated, separately-trained mixing head for
`projection_jax.mixed_reconstruct`, instead of Phase 16/17's zero-new-
parameter reuse of the network's existing troubled-probability output as
`lambda`.

Phase 17 trained through the mix three times (two Euler fine-tunes, one
full scalar retrain), reusing `max(troubled_probability)` as `lambda`
directly, and got a consistent, three-times-repeated regression relative
to Phase 16's free inference-only application. Working theory written up
there: that signal does double duty (selects which TENO5 candidates to
keep, AND, reused, decides how much to trust the network's own
reconstruction), so training through the mix creates conflicting gradient
pressure on the same weights. `teno5_jax_toy.init_lambda_head_params`/
`toy_lambda` decouple this: a small, separate (3 -> 1) head, taking the
existing troubled-probability vector as an input FEATURE rather than as
`lambda` itself, initialized to start almost entirely deferring to TENO5
(`sigmoid(-3.0) ~= 0.047`) so training can only move away from a safe
point where the loss actually justifies it.

Warm-starts the BACKBONE from the best existing Euler checkpoint
(Phase 15's smooth-mix+clip result) -- unaffected by adding the new head,
since the new head's parameters are entirely separate. The lambda head
itself has no prior checkpoint to warm-start from; `anchor_params` includes
its OWN initial (safe, TENO5-deferring) state, so `L_Reg` discourages it
from drifting away from that safe point without loss justification, the
same way `L_Reg` already discourages the backbone from drifting from its
own pretrained state.

Run inside `teno-jax` (needs the Phase 15 checkpoint to exist):

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX_LearnedLambda.py
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
from src.differentiable.projection_jax import mixed_reconstruct, project_keep_weight
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import init_lambda_head_params, toy_keep_weight, toy_lambda

BACKBONE_WARM_START = os.path.join(_REPO_ROOT, 'models', 'nn_teno5euler_unrolled_jax_params_curriculum_phase15_smoothmix.pkl')
OUTPUT_PATH = os.path.join(_REPO_ROOT, 'models', 'nn_teno5euler_unrolled_jax_params_curriculum_phase18_learnedlambda.pkl')
CURRICULUM = [(15, 100), (15, 100)]  # two passes at K=15: the backbone is already stable there; the new head just needs steps, not a K ramp
STEPS_K = 15


def learned_network_reconstruct(backbone_params, lambda_params, stencils, mode='soft'):
    troubled = 1.0 - toy_keep_weight(backbone_params, stencils)

    def network_reconstruct(s):
        return teno5_soft_reconstruct_jax(s, project_keep_weight(troubled, mode=mode))

    lam = toy_lambda(lambda_params, troubled)
    return mixed_reconstruct(troubled, stencils, network_reconstruct, teno5_reconstruct_jax, mode=mode, lam=lam)


def single_example_loss(params, u0, exact_final, anchor_params, steps_k):
    backbone_params, lambda_params = params

    def network_reconstruct(stencils):
        return learned_network_reconstruct(backbone_params, lambda_params, stencils, mode='soft')

    flux_fn = base.get_euler_flux_jax(network_reconstruct, boundary='transmissive')
    final_state, positivity_penalty = base.rollout_with_positivity_penalty(
        u0, flux_fn, base.DT, base.DX, steps_k,
        state_guard=base.euler_positivity_repair, penalty_fn=base.positivity_penalty_jax,
    )
    network_bv = base.euler_boundary_variation_jax(final_state, network_reconstruct, boundary='transmissive')
    deterministic_bv = base.euler_boundary_variation_jax(final_state, teno5_reconstruct_jax, boundary='transmissive')
    total, breakdown = base.unrolled_physics_loss(
        final_state, exact_final, u0, network_bv, deterministic_bv,
        params=params, anchor_params=anchor_params,
        positivity_penalty=positivity_penalty,
        **base.LAMBDAS,
    )
    return total, breakdown


def batched_loss(params, batch_u0, batch_exact, anchor_params, steps_k):
    totals, breakdowns = jax.vmap(
        lambda u0, ex: single_example_loss(params, u0, ex, anchor_params, steps_k),
        in_axes=(0, 0),
    )(batch_u0, batch_exact)
    return jnp.mean(totals), jax.tree_util.tree_map(jnp.mean, breakdowns)


def network_rollout(params, u0, steps):
    backbone_params, lambda_params = params

    def reconstruct(stencils):
        return learned_network_reconstruct(backbone_params, lambda_params, stencils, mode='hard')

    flux_fn = base.get_euler_flux_jax(reconstruct, boundary='transmissive')
    return base.rollout(u0, flux_fn, base.DT, base.DX, steps, state_guard=base.euler_positivity_repair)


def main():
    with open(BACKBONE_WARM_START, 'rb') as f:
        loaded = pickle.load(f)
    backbone_params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded]

    lambda_key = jax.random.PRNGKey(11)
    lambda_params = init_lambda_head_params(lambda_key)

    params = (backbone_params, lambda_params)
    anchor_params = (backbone_params, lambda_params)
    print('Loaded Phase 15 backbone as warm start; fresh lambda head, sigmoid(-3.0) ~= {:.4f} initial mix.'.format(
        float(jax.nn.sigmoid(-3.0))
    ))

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0e-3))
    opt_state = optimizer.init(params)

    validation_x = np.arange(base.CELLS, dtype=float) / base.CELLS
    validation_u0 = base.riemann_state_at((1.0, 0.0, 1.0, 0.125, 0.0, 0.1), 0.5, 1.0e-6)
    validation_exact = jnp.asarray(base.exact_sod_solution(validation_x, STEPS_K * base.DT))
    best_params = params
    best_error = float('inf')

    @jax.jit
    def train_step(params, opt_state, batch_u0, batch_exact):
        (loss, breakdown), grads = jax.value_and_grad(batched_loss, has_aux=True)(
            params, batch_u0, batch_exact, anchor_params, STEPS_K
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, breakdown

    rng = np.random.default_rng(4)
    started = time.perf_counter()
    global_step = 0
    for stage_index, (steps_k, n_steps) in enumerate(CURRICULUM):
        print('\n--- Stage {}: K={}, {} steps ---'.format(stage_index, steps_k, n_steps))
        stage_started = time.perf_counter()
        for local_step in range(n_steps):
            batch_u0, batch_exact = base.sample_batch(rng, 8, STEPS_K, smooth_fraction=0.5)
            params, opt_state, loss, breakdown = train_step(params, opt_state, batch_u0, batch_exact)
            if local_step == 0:
                print('  first (compiling) step took {:.1f}s'.format(time.perf_counter() - stage_started))
            if not np.isfinite(float(loss)):
                print('  step {:4d}: loss went non-finite -- stopping'.format(local_step))
                break
            if local_step % 20 == 0 or local_step == n_steps - 1:
                lam_bias = float(params[1][1][0])
                print('  step {:4d}: l_total={:.6e} l_mse={:.6e} l_pos={:.6e} lambda_bias={:.4f}'.format(
                    local_step, breakdown['l_total'], breakdown['l_mse'], breakdown['l_positivity'], lam_bias,
                ))
            if local_step % 10 == 0 or local_step == n_steps - 1:
                nn_final = network_rollout(params, validation_u0, STEPS_K)
                validation_error = base.l2_error(nn_final, validation_exact)
                if np.isfinite(validation_error) and validation_error < best_error:
                    best_error = validation_error
                    best_params = params
            global_step += 1

    print('\nTotal training time: {:.1f}s ({} steps)'.format(time.perf_counter() - started, global_step))
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

    print('\n' + '=' * 70)
    print('Held-out evaluation vs. deterministic TENO5euler')
    print('=' * 70)
    rng_eval = np.random.default_rng(42)
    u0_a, exact_a = base.sample_riemann_case(rng_eval, STEPS_K)
    nn_a = network_rollout(params, u0_a, STEPS_K)
    teno_a = base.deterministic_rollout(u0_a, STEPS_K)
    print('  held-out Riemann, K steps   : NN L2={:.6e}  TENO5euler L2={:.6e}  {}'.format(
        base.l2_error(nn_a, exact_a), base.l2_error(teno_a, exact_a),
        'BEATS TENO5' if base.l2_error(nn_a, exact_a) <= base.l2_error(teno_a, exact_a) else 'loses to TENO5',
    ))
    nn_sod = network_rollout(params, validation_u0, STEPS_K)
    teno_sod = base.deterministic_rollout(validation_u0, STEPS_K)
    print('  canonical Sod, K steps      : NN L2={:.6e}  TENO5euler L2={:.6e}  {}'.format(
        base.l2_error(nn_sod, validation_exact), base.l2_error(teno_sod, validation_exact),
        'BEATS TENO5' if base.l2_error(nn_sod, validation_exact) <= base.l2_error(teno_sod, validation_exact) else 'loses to TENO5',
    ))
    print('  final lambda head: w={} b={}'.format(np.asarray(lambda_params[0]).ravel(), np.asarray(lambda_params[1])))


if __name__ == '__main__':
    main()
