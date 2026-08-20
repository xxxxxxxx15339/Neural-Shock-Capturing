# -*- coding: utf-8 -*-
"""Phase 6/7 Euler extension: does the pretrain+anchor fix that made the
scalar Unrolled Physics Loss run beat TENO5 (Train_NNTENO5_UnrolledJAX.py,
run3_pretrained) generalize to the Euler system?

Reuses run3_pretrained's weights directly as both the starting point and
the L_Reg anchor -- not retrained from scratch. This works because
imitation-pretraining (see Train_NNTENO5_UnrolledJAX.py's
pretrain_by_imitation) trains purely on "5-point sequence in -> TENO5's own
reconstruction out," which is representation-agnostic: the same network
applies unchanged whether that 5-point sequence is a raw scalar stencil or
one Euler characteristic field's split-flux stencil (established in Phase 1
-- the toy network's operations all broadcast over the extra
characteristic-field axis with no code changes needed). Re-pretraining for
Euler specifically would just relearn the same thing.

Training data: random two-state Riemann problems (same parameter ranges as
GenerateTrainingData_EulerExact.py's random_riemann_params), each with a
closed-form exact solution at every future time via
exact_riemann_solution -- so, like the scalar run's square/sine waves, no
rollout is needed to know the training target, only to know the network's
prediction.

Run inside `teno-jax` (needs run3_pretrained's weights to exist -- run
Train_NNTENO5_UnrolledJAX.py first):

    conda run -n teno-jax python -u Train_NNTENO5euler_UnrolledJAX.py
"""

import pickle
import time

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import optax

from src.analysis.euler_regression import exact_riemann_solution, exact_sod_solution
from src.initial_conditions.InitialConditions import sod
from src.differentiable.euler_jax import (
    euler_boundary_variation_jax,
    euler_positivity_repair,
    get_euler_flux_jax,
    positivity_penalty_jax,
)
from src.differentiable.losses_jax import unrolled_physics_loss
from src.differentiable.numerics_jax import rollout
from src.differentiable.projection_jax import project_keep_weight
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import toy_keep_weight

CELLS = 64
LENGTH = 1.0
MAX_WAVE_SPEED = 2.0
CFL = 0.5
DX = LENGTH / CELLS
DT = CFL * DX / MAX_WAVE_SPEED
TRAIN_STEPS_K = 15
X = jnp.asarray(np.arange(CELLS, dtype=float) * LENGTH / CELLS)

LAMBDAS = dict(
    lambda_mse=1.0, lambda_tv=0.1, lambda_bvd=0.1, lambda_reg=0.1,
    lambda_positivity=1.0,
    tv_normalize=True, reg_normalize=True,
)
# lambda_positivity=1.0: new per the professor's clarification (Q5) --
# euler_positivity_repair keeps the rollout numerically finite by
# neighbor-averaging isolated violations (and, as a last resort, flooring
# anything a cluster leaves unrepaired), but that repair is invisible to
# the loss unless something tells it a violation happened at all. Wiring
# positivity_penalty_jax in as a real loss term is what gives the network
# a gradient signal to avoid causing violations in the first place, rather
# than only ever seeing already-repaired, deceptively "clean" states.


def _network_reconstruct(params, stencils, mode='soft'):
    troubled = 1.0 - toy_keep_weight(params, stencils)
    keep_weight = project_keep_weight(troubled, mode=mode)
    return teno5_soft_reconstruct_jax(stencils, keep_weight)


def random_riemann_params(rng):
    """Same ranges as GenerateTrainingData_EulerExact.py's random_riemann_params."""
    rho_l = rng.uniform(0.15, 4.0)
    rho_r = rng.uniform(0.15, 4.0)
    vel_l = rng.uniform(-1.5, 1.5)
    vel_r = rng.uniform(-1.5, 1.5)
    p_l = rng.uniform(0.1, 11.0)
    p_r = rng.uniform(0.1, 11.0)
    return rho_l, vel_l, p_l, rho_r, vel_r, p_r


def sample_riemann_problem(rng, steps_k=TRAIN_STEPS_K, max_attempts=50):
    """Draw a random Riemann problem, rejecting any that produce a
    non-finite or non-positive-density state at the training horizon.

    GenerateTrainingData_EulerExact.py's own `generated_states` does this
    same filtering (`if np.any(state[:, 0] <= 0): continue`) for exactly
    this reason: wide-range random two-state Riemann parameters can produce
    vacuum-forming combinations (two strong rarefactions moving apart), and
    the exact solution's density there is non-positive/non-finite. Skipping
    this check here (an early version of this script did) let a single bad
    sample in an 8-example vmapped batch poison the entire batch's loss to
    NaN from step 0 -- easy to misdiagnose as a training/optimization
    failure when it was a missing data-validity check.
    """
    for _ in range(max_attempts):
        params = random_riemann_params(rng)
        discontinuity = rng.uniform(0.3, 0.7)
        probe = exact_riemann_solution(
            np.asarray(X), steps_k * DT, *params, discontinuity=discontinuity
        )
        if np.isfinite(probe).all() and np.all(probe[:, 0] > 0.0):
            return params, discontinuity
    raise RuntimeError('Could not sample a valid Riemann problem after {} attempts.'.format(max_attempts))


def riemann_state_at(riemann_params, discontinuity, time):
    return jnp.asarray(
        exact_riemann_solution(np.asarray(X), time, *riemann_params, discontinuity=discontinuity)
    )


def sample_riemann_case(rng, steps_k):
    riemann_params, discontinuity = sample_riemann_problem(rng, steps_k=steps_k)
    u0 = riemann_state_at(riemann_params, discontinuity, 1.0e-6)
    exact = riemann_state_at(riemann_params, discontinuity, steps_k * DT)
    return u0, exact


def sample_batch(rng, batch_size, steps_k):
    u0_list, exact_list = [], []
    for _ in range(batch_size):
        u0, exact = sample_riemann_case(rng, steps_k)
        u0_list.append(u0)
        exact_list.append(exact)
    return jnp.stack(u0_list), jnp.stack(exact_list)


def single_example_loss(params, u0, exact_final, anchor_params, steps_k):
    def network_reconstruct(stencils):
        return _network_reconstruct(params, stencils, mode='soft')

    flux_fn = get_euler_flux_jax(network_reconstruct, boundary='transmissive')
    final_state = rollout(u0, flux_fn, DT, DX, steps_k, state_guard=euler_positivity_repair)
    network_bv = euler_boundary_variation_jax(final_state, network_reconstruct, boundary='transmissive')
    deterministic_bv = euler_boundary_variation_jax(
        final_state, teno5_reconstruct_jax, boundary='transmissive'
    )
    total, breakdown = unrolled_physics_loss(
        final_state, exact_final, u0, network_bv, deterministic_bv,
        params=params, anchor_params=anchor_params,
        positivity_penalty=positivity_penalty_jax(final_state),
        **LAMBDAS,
    )
    return total, breakdown


def batched_loss(params, batch_u0, batch_exact, anchor_params, steps_k):
    totals, breakdowns = jax.vmap(
        lambda u0, ex: single_example_loss(params, u0, ex, anchor_params, steps_k),
        in_axes=(0, 0),
    )(batch_u0, batch_exact)
    return jnp.mean(totals), jax.tree_util.tree_map(jnp.mean, breakdowns)


def deterministic_rollout(u0, steps):
    flux_fn = get_euler_flux_jax(teno5_reconstruct_jax, boundary='transmissive')
    return rollout(u0, flux_fn, DT, DX, steps)


def network_rollout(params, u0, steps):
    def reconstruct(stencils):
        return _network_reconstruct(params, stencils, mode='hard')

    flux_fn = get_euler_flux_jax(reconstruct, boundary='transmissive')
    return rollout(u0, flux_fn, DT, DX, steps, state_guard=euler_positivity_repair)


def l2_error(state, exact):
    return float(jnp.sqrt(jnp.mean((state - exact) ** 2)))


def evaluate_case(params, name, u0, exact, steps):
    nn_final = network_rollout(params, u0, steps)
    teno_final = deterministic_rollout(u0, steps)
    nn_error = l2_error(nn_final, exact)
    teno_error = l2_error(teno_final, exact)
    beats = nn_error <= teno_error
    print('  {:28s}: NN-TENO5euler(JAX) L2={:.6e}  TENO5euler L2={:.6e}  {}'.format(
        name, nn_error, teno_error, 'BEATS TENO5' if beats else 'loses to TENO5'
    ))
    return beats, nn_error, teno_error


def main():
    with open('models/nn_teno5_unrolled_jax_params_run3_pretrained.pkl', 'rb') as f:
        loaded = pickle.load(f)
    params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded]
    anchor_params = params
    print('Loaded pretrained scalar-run weights as Euler warm start/anchor.')

    optimizer = optax.adam(1.0e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_u0, batch_exact):
        (loss, breakdown), grads = jax.value_and_grad(batched_loss, has_aux=True)(
            params, batch_u0, batch_exact, anchor_params, TRAIN_STEPS_K
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, breakdown

    rng = np.random.default_rng(2)
    n_steps = 200
    batch_size = 8

    print('Training NN-TENO5euler (JAX) with the Unrolled Physics Loss')
    print('K={} SSPRK3 steps, batch_size={}, {} training steps'.format(
        TRAIN_STEPS_K, batch_size, n_steps))

    started = time.perf_counter()
    for step in range(n_steps):
        batch_u0, batch_exact = sample_batch(rng, batch_size, TRAIN_STEPS_K)
        params, opt_state, loss, breakdown = train_step(params, opt_state, batch_u0, batch_exact)
        if step == 0:
            print('first (compiling) step took {:.1f}s'.format(time.perf_counter() - started))
        if step % 40 == 0 or step == n_steps - 1:
            print('step {:4d}: l_total={:.6e} l_mse={:.6e} l_tv={:.6e} l_bvd={:.6e} l_reg={:.6e} l_pos={:.6e}'.format(
                step, breakdown['l_total'], breakdown['l_mse'], breakdown['l_tv'],
                breakdown['l_bvd'], breakdown['l_reg'], breakdown['l_positivity'],
            ))
    print('Training took {:.1f}s total'.format(time.perf_counter() - started))

    with open('models/nn_teno5euler_unrolled_jax_params.pkl', 'wb') as f:
        pickle.dump([(np.asarray(w), np.asarray(b)) for w, b in params], f)

    print('\n' + '=' * 70)
    print('Held-out evaluation vs. deterministic TENO5euler')
    print('=' * 70)

    eval_rng = np.random.default_rng(999)  # not used during training
    results = []

    # Case A: held-out random Riemann problem, valid out to 3x the trained
    # horizon (Case B reuses the same problem at the longer time).
    long_steps = TRAIN_STEPS_K * 3
    riemann_params, discontinuity = sample_riemann_problem(eval_rng, steps_k=long_steps)
    u0_a = riemann_state_at(riemann_params, discontinuity, 1.0e-6)
    exact_a = riemann_state_at(riemann_params, discontinuity, TRAIN_STEPS_K * DT)
    results.append(evaluate_case(params, 'held-out Riemann, K steps', u0_a, exact_a, TRAIN_STEPS_K))

    # Case B: the same problem, 3x the trained horizon -- generalization
    # beyond the K the network was actually trained on.
    exact_b = riemann_state_at(riemann_params, discontinuity, long_steps * DT)
    results.append(evaluate_case(params, 'held-out Riemann, 3x K steps', u0_a, exact_b, long_steps))

    # Case C: canonical Sod problem specifically (not part of training's
    # random distribution edges -- rho_r=0.125 is near the sampled range's
    # lower bound but the exact combination is a fixed, well-known case).
    u0_sod = jnp.asarray(sod()(np.asarray(X)))
    exact_sod = jnp.asarray(exact_sod_solution(np.asarray(X), TRAIN_STEPS_K * DT))
    results.append(evaluate_case(params, 'canonical Sod, K steps', u0_sod, exact_sod, TRAIN_STEPS_K))

    beats_count = sum(1 for beats, _, _ in results if beats)
    print('\nSummary: NN-TENO5euler(JAX, unrolled-loss-trained) beats deterministic TENO5 '
          'on {}/{} held-out cases.'.format(beats_count, len(results)))


if __name__ == '__main__':
    main()
