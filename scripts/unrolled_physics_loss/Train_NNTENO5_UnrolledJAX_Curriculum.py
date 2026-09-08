# -*- coding: utf-8 -*-
"""Widens Train_NNTENO5_UnrolledJAX.py's (run3_pretrained's) training scope
to close the two gaps ValidationMatrix_ProjPDF.py measured (Results.md
Phase 7): Advection loses at `final_time=1.0` (~128 SSPRK3 steps, ~8.5x the
trained K=15 horizon); Burgers loses on an EXPLICITLY out-of-distribution
case -- run3_pretrained's training data is 100% linear-advection physics
(`physical_flux(u) = u`), so it never once saw Burgers' nonlinear
`0.5*u**2` flux during training.

Two additions, mirroring the Euler curriculum script's own two lessons
(Results.md Phase 13/15 -- stage K gradually, and clip gradients once the
training distribution is widened):

1. A K-curriculum (15 -> 40 -> 80 -> 128, matching ValidationMatrix's own
   Advection step count exactly at the final stage) warm-started from
   run3_pretrained, carrying Adam state across stages -- same reasoning
   Um et al. gave for Euler (Results.md Phase 11), applied here even though
   scalar advection/Burgers never had Euler's stability problem: widening K
   is still a bigger jump in unroll depth than any single stage should take
   cold.
2. A Burgers training family with its own exact target -- the method of
   characteristics used by TENO5_vs_NNTENO5_Scalar.py's/ValidationMatrix's
   `burgers_characteristic_reference`, generalized here
   (`burgers_characteristic_general`) from one fixed profile to any
   random smooth sine profile, with amplitude/frequency kept low enough
   (rejection-sampled, same pattern as the Euler Riemann sampler) that the
   profile provably stays pre-shock at every curriculum stage up to the
   final K=128 -- so this never needs a real simulation to know its own
   training target, the same reason the advection/Burgers families were
   chosen at all in Train_NNTENO5_UnrolledJAX.py's own docstring.

Advection and Burgers batches are trained as two separate batches per step
(not mixed within one vmapped batch), combined as a straight average --
simpler than making one flux function branch per-example inside a single
vmap, at the cost of one extra vmap call per step.

Run inside `teno-jax` (needs run3_pretrained's weights to exist):

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/Train_NNTENO5_UnrolledJAX_Curriculum.py
"""

import os
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import functools
import pickle
import time

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import optax

import Train_NNTENO5_UnrolledJAX as base

CURRICULUM = [(15, 100), (40, 100), (80, 100), (128, 120)]
BURGERS_BATCH_SIZE = 8
ADVECTION_BATCH_SIZE = 16


def burgers_flux(u):
    return 0.5 * u ** 2


def burgers_profile_np(x, amplitude, frequency, phase, offset):
    return offset + amplitude * np.sin(2.0 * np.pi * frequency * x + phase)


def burgers_characteristic_general(x, final_time, amplitude, frequency, phase, offset, iters=60):
    """Method-of-characteristics exact Burgers solution, generalized from
    ValidationMatrix_ProjPDF.py's `burgers_characteristic_reference` (itself
    copied from TENO5_vs_NNTENO5_Scalar.py) to an arbitrary sine profile
    instead of one fixed one. Same bisection technique, same validity
    condition (strictly pre-shock) -- see that function's docstring for why
    this is copied rather than imported (keeps Keras out of this
    teno-jax-environment script).
    """
    def profile(xx):
        return burgers_profile_np(xx, amplitude, frequency, phase, offset)

    mapped_start = final_time * profile(np.array([0.0]))[0]
    target = np.mod(x, 1.0)
    target = np.where(target < mapped_start, target + 1.0, target)

    lower = np.zeros_like(target)
    upper = np.ones_like(target)
    for _ in range(iters):
        midpoint = 0.5 * (lower + upper)
        mapped_midpoint = midpoint + final_time * profile(midpoint)
        belongs_right = mapped_midpoint < target
        lower = np.where(belongs_right, midpoint, lower)
        upper = np.where(belongs_right, upper, midpoint)

    foot = 0.5 * (lower + upper)
    return profile(foot)


def sample_burgers_case(rng, steps_k, max_attempts=50):
    """Rejection-sampled so every draw stays pre-shock even at this
    curriculum's final K=128 (total_time=1.0): amplitude/frequency ranges
    are chosen so the loosest case (amplitude=0.15, frequency=1.5) has
    shock_time = 1/(0.15*2*pi*1.5) ~= 0.707, still checked with an explicit
    0.85 safety margin below, rather than trusted from the range alone.
    """
    total_time = steps_k * base.DT
    for _ in range(max_attempts):
        amplitude = rng.uniform(0.05, 0.15)
        frequency = rng.uniform(1.0, 1.5)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        offset = rng.uniform(0.3, 0.7)
        shock_time = 1.0 / (amplitude * 2.0 * np.pi * frequency)
        if total_time < 0.85 * shock_time:
            u0 = burgers_profile_np(np.asarray(base.X), amplitude, frequency, phase, offset)
            exact = burgers_characteristic_general(
                np.asarray(base.X), total_time, amplitude, frequency, phase, offset
            )
            return jnp.asarray(u0), jnp.asarray(exact)
    raise RuntimeError('Could not sample a valid pre-shock Burgers problem after {} attempts.'.format(max_attempts))


def sample_burgers_batch(rng, batch_size, steps_k):
    u0_list, exact_list = [], []
    for _ in range(batch_size):
        u0, exact = sample_burgers_case(rng, steps_k)
        u0_list.append(u0)
        exact_list.append(exact)
    return jnp.stack(u0_list), jnp.stack(exact_list)


def _burgers_flux_fn(params, u, mode='soft'):
    # Fixed alpha=1.0, matching ValidationMatrix_ProjPDF.py's own Burgers
    # Lax-Friedrichs convention (a global dissipation bound, not adaptive
    # per-state) -- keeps this consistent with how the held-out eval itself
    # actually runs Burgers, not a different discretization choice.
    # Routes through base._mixed_network_reconstruct (Results.md Phase 16/17),
    # not the raw network reconstruction -- same reasoning as base._flux_fn.
    return base.lax_friedrichs_flux(
        u, burgers_flux, 1.0, lambda s: base._mixed_network_reconstruct(params, s, mode=mode)
    )


def single_example_loss_burgers(params, u0, exact_final, anchor_params, steps_k):
    def network_reconstruct(stencils):
        return base._network_reconstruct(params, stencils, mode='soft')

    final_state = base.rollout(u0, lambda u: _burgers_flux_fn(params, u, mode='soft'), base.DT, base.DX, steps_k)
    network_bv = base.boundary_variation_jax(final_state, network_reconstruct)
    deterministic_bv = base.boundary_variation_jax(final_state, base.teno5_reconstruct_jax)
    total, breakdown = base.unrolled_physics_loss(
        final_state, exact_final, u0, network_bv, deterministic_bv,
        params, anchor_params, **base.LAMBDAS,
    )
    return total, breakdown


def batched_loss_burgers(params, batch_u0, batch_exact, anchor_params, steps_k):
    totals, breakdowns = jax.vmap(
        lambda u0, ex: single_example_loss_burgers(params, u0, ex, anchor_params, steps_k),
        in_axes=(0, 0),
    )(batch_u0, batch_exact)
    return jnp.mean(totals), jax.tree_util.tree_map(jnp.mean, breakdowns)


def deterministic_rollout_burgers(u0, steps):
    flux_fn = lambda u: base.lax_friedrichs_flux(u, burgers_flux, 1.0, base.teno5_reconstruct_jax)
    return base.rollout(u0, flux_fn, base.DT, base.DX, steps)


def network_rollout_burgers(params, u0, steps):
    flux_fn = lambda u: _burgers_flux_fn(params, u, mode='hard')
    return base.rollout(u0, flux_fn, base.DT, base.DX, steps)


def main():
    with open(os.path.join(_REPO_ROOT, 'models', 'nn_teno5_unrolled_jax_params_run3_pretrained.pkl'), 'rb') as f:
        loaded = pickle.load(f)
    params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded]
    anchor_params = params
    print('Loaded run3_pretrained weights as warm start/anchor.')
    print('Curriculum: K = {}'.format([k for k, _ in CURRICULUM]))

    # See Train_NNTENO5euler_UnrolledJAX.py's matching comment
    # (Results.md Phase 14): clips gradient global norm before Adam so one
    # pathological batch (more likely once the training distribution is
    # widened to Burgers and a much longer K) can't wreck an otherwise-good
    # checkpoint in a single destructive update.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1.0e-3))
    opt_state = optimizer.init(params)

    rng = np.random.default_rng(2)
    started = time.perf_counter()
    global_step = 0

    for stage_index, (steps_k, n_steps) in enumerate(CURRICULUM):

        @functools.partial(jax.jit, static_argnames=('steps_k',))
        def train_step(params, opt_state, adv_u0, adv_exact, burg_u0, burg_exact, steps_k):
            def combined_loss(params):
                adv_total, adv_bd = base.batched_loss(params, adv_u0, adv_exact, anchor_params, steps_k)
                burg_total, burg_bd = batched_loss_burgers(params, burg_u0, burg_exact, anchor_params, steps_k)
                total = 0.5 * adv_total + 0.5 * burg_total
                breakdown = {
                    'l_total': total,
                    'l_advection': adv_total,
                    'l_burgers': burg_total,
                    'l_mse_adv': adv_bd['l_mse'],
                    'l_mse_burgers': burg_bd['l_mse'],
                }
                return total, breakdown

            (loss, breakdown), grads = jax.value_and_grad(combined_loss, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, breakdown

        print('\n--- Stage {}: K={}, {} steps ---'.format(stage_index, steps_k, n_steps))
        stage_started = time.perf_counter()
        diverged = False
        for local_step in range(n_steps):
            adv_u0, adv_exact = base.sample_batch(rng, ADVECTION_BATCH_SIZE, steps_k)
            burg_u0, burg_exact = sample_burgers_batch(rng, BURGERS_BATCH_SIZE, steps_k)
            params, opt_state, loss, breakdown = train_step(
                params, opt_state, adv_u0, adv_exact, burg_u0, burg_exact, steps_k
            )
            if local_step == 0:
                print('  first (compiling) step took {:.1f}s'.format(time.perf_counter() - stage_started))
            if not np.isfinite(float(loss)):
                print('  step {:4d}: loss went non-finite -- stage diverged'.format(local_step))
                diverged = True
                break
            if local_step % 20 == 0 or local_step == n_steps - 1:
                print('  step {:4d}: l_total={:.6e} l_advection={:.6e} l_burgers={:.6e}'.format(
                    local_step, breakdown['l_total'], breakdown['l_advection'], breakdown['l_burgers'],
                ))
            global_step += 1
        if diverged:
            print('Curriculum stopped early at K={} after divergence.'.format(steps_k))
            break

    print('\nTotal training time: {:.1f}s ({} steps)'.format(time.perf_counter() - started, global_step))

    run_tag = 'run4_widehorizon'
    params_path = os.path.join(_REPO_ROOT, 'models', 'nn_teno5_unrolled_jax_params_{}.pkl'.format(run_tag))
    with open(params_path, 'wb') as f:
        pickle.dump([(np.asarray(w), np.asarray(b)) for w, b in params], f)
    print('Trained parameters written to {}'.format(params_path))

    print('\n' + '=' * 70)
    print('Held-out evaluation vs. deterministic TENO5 (same cases as run3_pretrained,')
    print('for a direct comparison)')
    print('=' * 70)

    results = []
    u0_a = base.square_wave(base.X, center=0.35, width=0.28, amplitude=0.6, offset=0.1)
    exact_a = base.square_wave(base.exact_shifted(base.X, base.TRAIN_STEPS_K * base.DT), 0.35, 0.28, 0.6, 0.1)
    results.append(base.evaluate_case(params, 'held-out jump, K=15 steps', u0_a, exact_a, base.TRAIN_STEPS_K))

    long_steps = 128
    exact_b = base.square_wave(base.exact_shifted(base.X, long_steps * base.DT), 0.35, 0.28, 0.6, 0.1)
    results.append(base.evaluate_case(params, 'held-out jump, 128 steps', u0_a, exact_b, long_steps))

    u0_c = base.sine_wave(base.X, frequency=2.3, phase=0.7, amplitude=0.5, offset=0.0)
    exact_c = base.sine_wave(base.exact_shifted(base.X, base.TRAIN_STEPS_K * base.DT), 2.3, 0.7, 0.5, 0.0)
    results.append(base.evaluate_case(params, 'held-out smooth sine, K=15 steps', u0_c, exact_c, base.TRAIN_STEPS_K))

    beats_count = sum(1 for beats, _, _ in results if beats)
    print('\nSummary (advection cases): NN-TENO5(JAX) beats deterministic TENO5 on '
          '{}/{} held-out cases.'.format(beats_count, len(results)))


if __name__ == '__main__':
    main()
