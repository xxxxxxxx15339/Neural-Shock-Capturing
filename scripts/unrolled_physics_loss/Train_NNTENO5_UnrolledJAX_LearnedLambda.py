# -*- coding: utf-8 -*-
"""Results.md Phase 19: ports the one mechanism that actually produced a win
(Phase 18's dedicated, separately-trained mixing head for Euler) to the
scalar NN-TENO5(JAX) network, which has never had it tried.

Everything tried so far on the scalar side either didn't help or made things
worse: Phase 15's K-curriculum (15->40->80->128) + Burgers-flux training
family, run cleanly but the canonical-benchmark numbers got slightly WORSE
(Advection 1.42x -> 1.45x); Phase 17's "train through the mix" reusing the
network's own `max(troubled_probability)` as `lambda` made Advection worse
still (1.42x -> 1.49x) for the same reason it hurt Euler -- that signal does
double duty (candidate selection AND mix-trust), so training through it
perturbs weights that were already good at the first job. Only Phase 16's
zero-training, inference-only convex mix helped at all (1.42x -> 1.39x
Advection, 3.62x -> 3.25x Burgers) -- and Phase 18 showed, on Euler, that a
DECOUPLED, separately-trained mixing head can improve substantially on that
free baseline (to an outright win on Shu-Osher). This script is the
scalar-side test of the same idea, never attempted before this phase.

Two things ported from Euler's Phase 18, plus one addition specific to why
scalar's gap is larger than Euler's ever was:
1. `teno5_jax_toy.init_lambda_head_params`/`toy_lambda`: a small (3 -> 1)
   head taking troubled_probability as an input feature, initialized
   conservative (`sigmoid(-3.0) ~= 0.047`, mostly defers to TENO5), anchored
   to its own safe init via L_Reg -- exactly Phase 18's mechanism, physics-
   agnostic already (works on scalar (n,3) or Euler (n,3,3) troubled-
   probability shapes without modification).
2. Burgers-flux training examples mixed in alongside linear advection
   (reusing Train_NNTENO5_UnrolledJAX_Curriculum.py's
   `sample_burgers_batch`/`burgers_characteristic_general`) -- Burgers is
   explicitly out-of-distribution for `run3_pretrained` (trained only on
   linear advection physics), a real, separate gap from the mixing
   architecture question, worth addressing at the same time since this
   script already needs a Burgers loss branch to validate against Burgers
   at all.
3. Validated against the ACTUAL canonical-benchmark horizon during training
   (ValidationMatrix_ProjPDF.py's own 128-step Advection / 64-step Burgers
   rollouts, via `run_scalar`), not a short K=15 proxy -- training itself
   still unrolls only K=15 steps (cheap), but the best-checkpoint selection
   uses the real target metric directly, avoiding Phase 17's failure mode
   of a checkpoint that looks good at the trained horizon but isn't what's
   actually being asked to win.

Run inside `teno-jax` (needs run3_pretrained's weights to exist):

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/Train_NNTENO5_UnrolledJAX_LearnedLambda.py
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

import Train_NNTENO5_UnrolledJAX as base
import Train_NNTENO5_UnrolledJAX_Curriculum as curriculum
import ValidationMatrix_ProjPDF as vm
from src.differentiable.projection_jax import mixed_reconstruct, project_keep_weight
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import init_lambda_head_params, toy_keep_weight, toy_lambda

WARM_START_PATH = os.path.join(_REPO_ROOT, 'models', 'nn_teno5_unrolled_jax_params_run3_pretrained.pkl')
OUTPUT_PATH = os.path.join(_REPO_ROOT, 'models', 'nn_teno5_unrolled_jax_params_run6_learnedlambda.pkl')
STEPS_K = 15
N_STEPS = 200
ADV_BATCH_SIZE = 16
BURGERS_BATCH_SIZE = 8


def learned_network_reconstruct(backbone_params, lambda_params, stencils, mode='soft'):
    troubled = 1.0 - toy_keep_weight(backbone_params, stencils)

    def network_reconstruct(s):
        return teno5_soft_reconstruct_jax(s, project_keep_weight(troubled, mode=mode))

    lam = toy_lambda(lambda_params, troubled)
    return mixed_reconstruct(troubled, stencils, network_reconstruct, teno5_reconstruct_jax, mode=mode, lam=lam)


def single_example_loss_advection(params, u0, exact_final, anchor_params, steps_k):
    backbone_params, lambda_params = params

    def network_reconstruct(stencils):
        return learned_network_reconstruct(backbone_params, lambda_params, stencils, mode='soft')

    final_state = base.rollout(u0, lambda u: base.lax_friedrichs_flux(u, base.physical_flux, base.WAVE_SPEED, network_reconstruct), base.DT, base.DX, steps_k)
    network_bv = base.boundary_variation_jax(final_state, network_reconstruct)
    deterministic_bv = base.boundary_variation_jax(final_state, teno5_reconstruct_jax)
    total, breakdown = base.unrolled_physics_loss(
        final_state, exact_final, u0, network_bv, deterministic_bv,
        params, anchor_params, **base.LAMBDAS,
    )
    return total, breakdown


def single_example_loss_burgers(params, u0, exact_final, anchor_params, steps_k):
    backbone_params, lambda_params = params

    def network_reconstruct(stencils):
        return learned_network_reconstruct(backbone_params, lambda_params, stencils, mode='soft')

    final_state = base.rollout(u0, lambda u: base.lax_friedrichs_flux(u, curriculum.burgers_flux, 1.0, network_reconstruct), base.DT, base.DX, steps_k)
    network_bv = base.boundary_variation_jax(final_state, network_reconstruct)
    deterministic_bv = base.boundary_variation_jax(final_state, teno5_reconstruct_jax)
    total, breakdown = base.unrolled_physics_loss(
        final_state, exact_final, u0, network_bv, deterministic_bv,
        params, anchor_params, **base.LAMBDAS,
    )
    return total, breakdown


def batched_loss_advection(params, batch_u0, batch_exact, anchor_params, steps_k):
    totals, breakdowns = jax.vmap(
        lambda u0, ex: single_example_loss_advection(params, u0, ex, anchor_params, steps_k),
        in_axes=(0, 0),
    )(batch_u0, batch_exact)
    return jnp.mean(totals), jax.tree_util.tree_map(jnp.mean, breakdowns)


def batched_loss_burgers(params, batch_u0, batch_exact, anchor_params, steps_k):
    totals, breakdowns = jax.vmap(
        lambda u0, ex: single_example_loss_burgers(params, u0, ex, anchor_params, steps_k),
        in_axes=(0, 0),
    )(batch_u0, batch_exact)
    return jnp.mean(totals), jax.tree_util.tree_map(jnp.mean, breakdowns)


def network_reconstruct_hard(params):
    backbone_params, lambda_params = params

    def reconstruct(stencils):
        return learned_network_reconstruct(backbone_params, lambda_params, stencils, mode='hard')

    return reconstruct


def validate_canonical(params):
    """Full canonical-benchmark rollouts (ValidationMatrix_ProjPDF.py's own
    128-step Advection / 64-step Burgers), NOT the K=15 training horizon --
    see module docstring point 3 for why this is the selection metric.
    """
    reconstruct = network_reconstruct_hard(params)

    u0 = vm.step_profile(vm.X)
    exact = vm.step_profile(vm.X - 1.0)
    nn_final = np.asarray(vm.run_scalar(lambda u: u, 1.0, u0, reconstruct, 1.0))
    teno_final = np.asarray(vm.run_scalar(lambda u: u, 1.0, u0, teno5_reconstruct_jax, 1.0))
    _, nn_l2_adv = vm.l1_l2(nn_final, exact)
    _, teno_l2_adv = vm.l1_l2(teno_final, exact)

    burgers_time = 0.5
    u0b = vm.burgers_initial_profile(vm.X)
    exactb = vm.burgers_characteristic_reference(vm.X, burgers_time)
    nn_final_b = np.asarray(vm.run_scalar(lambda u: 0.5 * u ** 2, 1.0, u0b, reconstruct, burgers_time))
    teno_final_b = np.asarray(vm.run_scalar(lambda u: 0.5 * u ** 2, 1.0, u0b, teno5_reconstruct_jax, burgers_time))
    _, nn_l2_burg = vm.l1_l2(nn_final_b, exactb)
    _, teno_l2_burg = vm.l1_l2(teno_final_b, exactb)

    return {
        'adv_ratio': nn_l2_adv / teno_l2_adv,
        'burgers_ratio': nn_l2_burg / teno_l2_burg,
        'adv_nn': nn_l2_adv, 'adv_teno': teno_l2_adv,
        'burgers_nn': nn_l2_burg, 'burgers_teno': teno_l2_burg,
    }


def main():
    with open(WARM_START_PATH, 'rb') as f:
        loaded = pickle.load(f)
    backbone_params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded]

    lambda_params = init_lambda_head_params(jax.random.PRNGKey(21))
    params = (backbone_params, lambda_params)
    anchor_params = (backbone_params, lambda_params)
    print('Loaded run3_pretrained backbone as warm start; fresh lambda head, sigmoid(-3.0) ~= {:.4f} initial mix.'.format(
        float(jax.nn.sigmoid(-3.0))
    ))

    initial_validation = validate_canonical(params)
    print('Initial canonical validation: Advection ratio={:.4f}x, Burgers ratio={:.4f}x'.format(
        initial_validation['adv_ratio'], initial_validation['burgers_ratio']
    ))

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0e-3))
    opt_state = optimizer.init(params)

    best_params = params
    best_score = initial_validation['adv_ratio'] + initial_validation['burgers_ratio']
    best_validation = initial_validation

    @jax.jit
    def train_step(params, opt_state, adv_u0, adv_exact, burg_u0, burg_exact):
        def combined_loss(params):
            adv_total, adv_bd = batched_loss_advection(params, adv_u0, adv_exact, anchor_params, STEPS_K)
            burg_total, burg_bd = batched_loss_burgers(params, burg_u0, burg_exact, anchor_params, STEPS_K)
            total = 0.5 * adv_total + 0.5 * burg_total
            breakdown = {'l_total': total, 'l_advection': adv_total, 'l_burgers': burg_total}
            return total, breakdown

        (loss, breakdown), grads = jax.value_and_grad(combined_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, breakdown

    rng = np.random.default_rng(6)
    started = time.perf_counter()
    for step in range(N_STEPS):
        adv_u0, adv_exact = base.sample_batch(rng, ADV_BATCH_SIZE, STEPS_K)
        burg_u0, burg_exact = curriculum.sample_burgers_batch(rng, BURGERS_BATCH_SIZE, STEPS_K)
        params, opt_state, loss, breakdown = train_step(params, opt_state, adv_u0, adv_exact, burg_u0, burg_exact)
        if step == 0:
            print('first (compiling) step took {:.1f}s'.format(time.perf_counter() - started))
        if not np.isfinite(float(loss)):
            print('step {:4d}: loss went non-finite -- stopping'.format(step))
            break
        if step % 20 == 0 or step == N_STEPS - 1:
            lam_bias = float(params[1][1][0])
            print('step {:4d}: l_total={:.6e} l_advection={:.6e} l_burgers={:.6e} lambda_bias={:.4f}'.format(
                step, breakdown['l_total'], breakdown['l_advection'], breakdown['l_burgers'], lam_bias,
            ))
        if step % 20 == 0 or step == N_STEPS - 1:
            validation = validate_canonical(params)
            score = validation['adv_ratio'] + validation['burgers_ratio']
            print('  validation: Advection ratio={:.4f}x, Burgers ratio={:.4f}x'.format(
                validation['adv_ratio'], validation['burgers_ratio']
            ))
            if score < best_score:
                best_score = score
                best_params = params
                best_validation = validation

    print('\nTotal training time: {:.1f}s'.format(time.perf_counter() - started))
    print('Best validation: Advection ratio={:.4f}x, Burgers ratio={:.4f}x'.format(
        best_validation['adv_ratio'], best_validation['burgers_ratio']
    ))
    params = best_params
    backbone_params, lambda_params = params

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'backbone': [(np.asarray(w), np.asarray(b)) for w, b in backbone_params],
            'lambda_head': (np.asarray(lambda_params[0]), np.asarray(lambda_params[1])),
        }, f)
    print('Trained parameters written to {}'.format(OUTPUT_PATH))
    print('final lambda head: w={} b={}'.format(np.asarray(lambda_params[0]).ravel(), np.asarray(lambda_params[1])))


if __name__ == '__main__':
    main()
