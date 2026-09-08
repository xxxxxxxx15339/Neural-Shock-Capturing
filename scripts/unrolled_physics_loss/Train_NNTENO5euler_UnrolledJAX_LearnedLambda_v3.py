# -*- coding: utf-8 -*-
"""Results.md Phase 21: a richer mixing-gate head for Euler, to test whether
giving it more local context than 3 troubled-probability numbers can fix the
specific structural limit Phase 20 measured for Sod.

Phase 20's fixed-lambda sweep showed Sod's ratio getting monotonically WORSE
with any mixing coefficient at all, while Shu-Osher's got monotonically
better -- both more-training (higher lambda-head learning rate, Phase 20's
own v2 attempt) and less-training (Phase 18's original) sit in the same
place on this curve, because neither changes what information the gate has
access to. `teno5_jax_toy.toy_lambda_v2`/`init_lambda_head_params_v2` (new
this phase) feed the gate the raw normalized stencil values on top of
troubled_probability -- 18 numbers instead of 3, run through a small hidden
layer instead of a single linear layer -- so it has an actual chance to
tell a Sod-like interface apart from a Shu-Osher-like one, not just how
troubled the candidates locally look (which both can show similarly).

Same conservative-init philosophy, same backbone warm start (Phase 15,
unaffected -- this only replaces the lambda head), same training recipe as
Phase 18 (not Phase 20's higher-lr variant, which didn't help): K=15,
gradient clipping, mixed Riemann/smooth-entropy-wave batches, best-by-
validation checkpoint tracked against the real Sod rollout during training.

Run inside `teno-jax` (needs the Phase 15 checkpoint to exist):

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX_LearnedLambda_v3.py
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
from src.differentiable.teno5_jax_toy import init_lambda_head_params_v2, toy_keep_weight, toy_lambda_v2

BACKBONE_WARM_START = os.path.join(_REPO_ROOT, 'models', 'nn_teno5euler_unrolled_jax_params_curriculum_phase15_smoothmix.pkl')
OUTPUT_PATH = os.path.join(_REPO_ROOT, 'models', 'nn_teno5euler_unrolled_jax_params_curriculum_phase21_richgate.pkl')
CURRICULUM = [(15, 100), (15, 100)]
STEPS_K = 15


def learned_network_reconstruct(backbone_params, lambda_params, stencils, mode='soft'):
    troubled = 1.0 - toy_keep_weight(backbone_params, stencils)

    def network_reconstruct(s):
        return teno5_soft_reconstruct_jax(s, project_keep_weight(troubled, mode=mode))

    lam = toy_lambda_v2(lambda_params, troubled, stencils)
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

    lambda_params = init_lambda_head_params_v2(jax.random.PRNGKey(31))
    params = (backbone_params, lambda_params)
    anchor_params = (backbone_params, lambda_params)
    print('Loaded Phase 15 backbone as warm start; fresh RICH gate head (18 inputs, hidden=8), sigmoid(-3.0) ~= {:.4f} initial mix.'.format(
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

    rng = np.random.default_rng(7)
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
                print('  step {:4d}: l_total={:.6e} l_mse={:.6e} l_pos={:.6e}'.format(
                    local_step, breakdown['l_total'], breakdown['l_mse'], breakdown['l_positivity'],
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
    lambda_finite = all(
        bool(np.isfinite(np.asarray(w)).all()) and bool(np.isfinite(np.asarray(b)).all())
        for w, b in lambda_params
    )
    print('Final checkpoint finite: backbone={} lambda_head={}'.format(backbone_finite, lambda_finite))

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'backbone': [(np.asarray(w), np.asarray(b)) for w, b in backbone_params],
            'lambda_head_v2': [(np.asarray(w), np.asarray(b)) for w, b in lambda_params],
        }, f)
    print('Trained parameters written to {}'.format(OUTPUT_PATH))


if __name__ == '__main__':
    main()
