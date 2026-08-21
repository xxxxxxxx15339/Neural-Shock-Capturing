# -*- coding: utf-8 -*-
"""Phase 6: a real training run of the Unrolled Physics Loss proposal.

This is the central empirical question the whole UNROLLED_PHYSICS_LOSS_GUIDE.md
exists to answer: trained genuinely end-to-end through K unrolled SSPRK3
steps (not imitating TENO5's labels, not oracle-relabeled per-stencil
targets -- trajectory-level feedback, exactly as the Unrolled Physics Loss
proposal's Equation 4-5 propose), does the resulting network actually beat deterministic TENO5
against the real solution? Report_Results.md-style honesty applies: this
script reports whatever the held-out comparison actually shows, win or
lose.

Scope, stated plainly (what a production Phase 6 would extend):
- Network: a JAX MLP with NN-TENO5's real architecture (5 -> 64 -> 32 -> 16
  -> 8 -> 3, `src/differentiable/teno5_jax_toy.py`'s `init_toy_params` with
  `hidden_sizes=(64, 32, 16, 8)`), trained from a random initialization in
  JAX -- not a port of the existing trained Keras checkpoint, and not
  pretrained by imitation first. `L_Reg`'s anchor is therefore this run's
  own random initialization, not a warm-started baseline the way
  Train_NNTENO5.py's `--anchor-weight` uses it (Results.md Phase 9) -- a
  meaningful simplification, noted so results aren't over-read.
- Physics: scalar linear advection only (TENO5's scalar path,
  `src/differentiable/numerics_jax.py`), not the full the Unrolled Physics Loss proposal problem list
  (Burgers/Shu-Osher/Sod/Woodward-Colella). Chosen because it is cheap, has
  a trivial closed-form exact solution at every step (pure translation, no
  shock formation), and still genuinely exercises TENO5's candidate
  selection at the jump in every training example -- the mechanism the Unrolled Physics Loss proposal
  is actually trying to improve. Extending to Burgers/Euler is Phase 6's
  natural continuation once this narrower question has an answer, reusing
  `src/differentiable/euler_jax.py` from Phase 1.
- Training data: batches of random square-wave and sine profiles, generated
  fresh (not drawn from Phase 3's dataset), since the point here is testing
  the *loss*, not the *data pipeline* Phase 3 already audited separately.

Run inside `teno-jax`:

    conda run -n teno-jax python -u scripts/unrolled_physics_loss/Train_NNTENO5_UnrolledJAX.py
"""

import os
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import time

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import optax

from src.differentiable.losses_jax import unrolled_physics_loss
from src.differentiable.numerics_jax import (
    boundary_variation_jax,
    lax_friedrichs_flux,
    rollout,
)
from src.differentiable.projection_jax import project_keep_weight
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import init_toy_params, toy_keep_weight

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

CELLS = 64
WAVE_SPEED = 1.0
CFL = 0.5
DX = 1.0 / CELLS
DT = CFL * DX / WAVE_SPEED
TRAIN_STEPS_K = 15  # SSPRK3 steps unrolled per training example (K in Eq. 4)
X = jnp.asarray(np.linspace(0.0, 1.0, CELLS, endpoint=False))

LAMBDAS = dict(
    lambda_mse=1.0, lambda_tv=0.1, lambda_bvd=0.1, lambda_reg=0.1,
    tv_normalize=True, reg_normalize=True,
)
# Run 1 (unnormalized L_TV/L_Reg, i.e. tv_normalize=reg_normalize=False) lost
# to deterministic TENO5 on 0/3 held-out cases. Diagnosis from that run's
# per-term logging: raw L_TV sat around 0.6-0.9 while L_MSE sat around
# 0.01-0.02 -- L_TV, even after its 0.1 weight, still dominated the gradient
# by roughly 4x, so training minimized total variation over getting the
# answer right. See losses_jax.py's l_tv/l_reg docstrings for the full
# reasoning; this run tests the fix directly.


def physical_flux(u):
    return u


def _network_reconstruct(params, stencils, mode='soft'):
    troubled = 1.0 - toy_keep_weight(params, stencils)
    keep_weight = project_keep_weight(troubled, mode=mode)
    return teno5_soft_reconstruct_jax(stencils, keep_weight)


def _flux_fn(params, u, mode='soft'):
    return lax_friedrichs_flux(
        u, physical_flux, WAVE_SPEED, lambda s: _network_reconstruct(params, s, mode=mode)
    )


def exact_shifted(x, shift):
    return (x - shift) % 1.0


def square_wave(x, center, width, amplitude, offset):
    shifted = (x - center + 0.5) % 1.0 - 0.5
    return jnp.where(jnp.abs(shifted) < 0.5 * width, offset + amplitude, offset - amplitude)


def sine_wave(x, frequency, phase, amplitude, offset):
    return offset + amplitude * jnp.sin(2.0 * jnp.pi * frequency * x + phase)


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def sample_batch(rng, batch_size, steps_k):
    """Random mix of square-wave (70%) and sine (30%) profiles, each with a
    trivially-known exact solution at every future time (pure translation).
    """
    u0_list, exact_list = [], []
    total_time = steps_k * DT
    for _ in range(batch_size):
        if rng.uniform() < 0.7:
            center = rng.uniform(0.0, 1.0)
            width = rng.uniform(0.15, 0.5)
            amplitude = rng.uniform(0.3, 1.0)
            offset = rng.uniform(-0.3, 0.3)
            u0 = square_wave(X, center, width, amplitude, offset)
            exact = square_wave(
                exact_shifted(X, total_time), center, width, amplitude, offset
            )
        else:
            frequency = rng.uniform(1.0, 4.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amplitude = rng.uniform(0.2, 0.8)
            offset = rng.uniform(-0.2, 0.2)
            u0 = sine_wave(X, frequency, phase, amplitude, offset)
            exact = sine_wave(exact_shifted(X, total_time), frequency, phase, amplitude, offset)
        u0_list.append(u0)
        exact_list.append(exact)
    return jnp.stack(u0_list), jnp.stack(exact_list)


# ---------------------------------------------------------------------------
# Per-example and batched loss
# ---------------------------------------------------------------------------

def single_example_loss(params, u0, exact_final, anchor_params, steps_k):
    def network_reconstruct(stencils):
        return _network_reconstruct(params, stencils, mode='soft')

    final_state = rollout(u0, lambda u: _flux_fn(params, u, mode='soft'), DT, DX, steps_k)
    network_bv = boundary_variation_jax(final_state, network_reconstruct)
    deterministic_bv = boundary_variation_jax(final_state, teno5_reconstruct_jax)
    total, breakdown = unrolled_physics_loss(
        final_state, exact_final, u0, network_bv, deterministic_bv,
        params, anchor_params, **LAMBDAS,
    )
    return total, breakdown


def batched_loss(params, batch_u0, batch_exact, anchor_params, steps_k):
    totals, breakdowns = jax.vmap(
        lambda u0, ex: single_example_loss(params, u0, ex, anchor_params, steps_k),
        in_axes=(0, 0),
    )(batch_u0, batch_exact)
    mean_breakdown = jax.tree_util.tree_map(jnp.mean, breakdowns)
    return jnp.mean(totals), mean_breakdown


# ---------------------------------------------------------------------------
# Held-out evaluation vs. deterministic TENO5
# ---------------------------------------------------------------------------

def deterministic_rollout(u0, steps):
    flux_fn = lambda u: lax_friedrichs_flux(u, physical_flux, WAVE_SPEED, teno5_reconstruct_jax)
    return rollout(u0, flux_fn, DT, DX, steps)


def network_rollout(params, u0, steps):
    flux_fn = lambda u: _flux_fn(params, u, mode='hard')
    return rollout(u0, flux_fn, DT, DX, steps)


def l2_error(state, exact):
    return float(jnp.sqrt(jnp.mean((state - exact) ** 2)))


def evaluate_case(params, name, u0, exact, steps):
    nn_final = network_rollout(params, u0, steps)
    teno_final = deterministic_rollout(u0, steps)
    nn_error = l2_error(nn_final, exact)
    teno_error = l2_error(teno_final, exact)
    beats = nn_error <= teno_error
    print('  {:28s}: NN-TENO5(JAX) L2={:.6e}  TENO5 L2={:.6e}  {}'.format(
        name, nn_error, teno_error, 'BEATS TENO5' if beats else 'loses to TENO5'
    ))
    return beats, nn_error, teno_error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pretrain_by_imitation(key, hidden_sizes, n_steps=500, batch_size=256, lr=1.0e-3):
    """Imitation-pretrain the network to reproduce deterministic TENO5's own
    reconstruction, before the unrolled-physics fine-tune -- the same
    warm-start pattern Results.md Phase 9 found necessary for the Keras
    NN-TENO5 (anchor-weight fine-tuning from a pretrained baseline, not
    training from scratch against the harder objective). Two runs of
    training this JAX network from scratch against the unrolled loss alone
    (UNROLLED_PHYSICS_LOSS_GUIDE.md Phase 6 status note, runs 1-2) both lost
    to TENO5 on every held-out case, including badly on smooth data where
    even reproducing TENO5's own behavior should be easy -- evidence the
    cold-start optimization itself, not just the loss weighting, was the
    problem. This pretraining objective is dense and well-conditioned (a
    direct per-stencil MSE against a known target, no rollout, no unrolling)
    and should converge fast.
    """
    params = init_toy_params(key, hidden_sizes=hidden_sizes)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, stencils):
        keep_weight = project_keep_weight(
            1.0 - toy_keep_weight(params, stencils), mode='soft'
        )
        prediction = teno5_soft_reconstruct_jax(stencils, keep_weight)
        target = teno5_reconstruct_jax(stencils)
        return jnp.mean((prediction - target) ** 2)

    @jax.jit
    def step_fn(params, opt_state, stencils):
        loss, grads = jax.value_and_grad(loss_fn)(params, stencils)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng = np.random.default_rng(1)
    print('\nImitation-pretraining (dense, per-stencil, no rollout)...')
    started = time.perf_counter()
    for step in range(n_steps):
        # Mix of random smooth-ish and discontinuity-like stencils so the
        # pretrained network has seen both regimes TENO5 itself handles.
        stencils = jnp.asarray(rng.normal(size=(batch_size, 5)).cumsum(axis=1))
        params, opt_state, loss = step_fn(params, opt_state, stencils)
        if step % 100 == 0 or step == n_steps - 1:
            print('  pretrain step {:4d}: imitation MSE = {:.6e}'.format(step, float(loss)))
    print('Pretraining took {:.1f}s'.format(time.perf_counter() - started))
    return params


def main():
    key = jax.random.PRNGKey(0)
    pretrain_key, _ = jax.random.split(key)
    params = pretrain_by_imitation(pretrain_key, hidden_sizes=(64, 32, 16, 8))
    anchor_params = params  # anchor = the pretrained baseline, matching Phase 9's pattern

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

    rng = np.random.default_rng(0)
    n_steps = 300
    batch_size = 16

    print('Training NN-TENO5 (JAX, real architecture) with the Unrolled Physics Loss')
    print('K={} SSPRK3 steps, batch_size={}, {} training steps'.format(
        TRAIN_STEPS_K, batch_size, n_steps))
    print('lambdas: {}'.format(LAMBDAS))

    history = []
    started = time.perf_counter()
    for step in range(n_steps):
        batch_u0, batch_exact = sample_batch(rng, batch_size, TRAIN_STEPS_K)
        params, opt_state, loss, breakdown = train_step(params, opt_state, batch_u0, batch_exact)
        history.append({k: float(v) for k, v in breakdown.items()})
        if step == 0:
            print('first (compiling) step took {:.1f}s'.format(time.perf_counter() - started))
        if step % 50 == 0 or step == n_steps - 1:
            print('step {:4d}: l_total={:.6e} l_mse={:.6e} l_tv={:.6e} l_bvd={:.6e} l_reg={:.6e}'.format(
                step, breakdown['l_total'], breakdown['l_mse'], breakdown['l_tv'],
                breakdown['l_bvd'], breakdown['l_reg'],
            ))
    total_seconds = time.perf_counter() - started
    print('Training took {:.1f}s total ({:.3f}s/step average)'.format(
        total_seconds, total_seconds / n_steps))

    run_tag = 'run3_pretrained'

    import csv
    history_path = os.path.join(_REPO_ROOT, 'models', 'nn_teno5_unrolled_jax_history_{}.csv'.format(run_tag))
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['l_total', 'l_mse', 'l_tv', 'l_bvd', 'l_reg'])
        writer.writeheader()
        writer.writerows(history)
    print('Training history written to {}'.format(history_path))

    import pickle
    params_path = os.path.join(_REPO_ROOT, 'models', 'nn_teno5_unrolled_jax_params_{}.pkl'.format(run_tag))
    with open(params_path, 'wb') as f:
        pickle.dump([(np.asarray(w), np.asarray(b)) for w, b in params], f)
    print('Trained parameters written to {}'.format(params_path))

    # -----------------------------------------------------------------
    # Held-out evaluation: NOT seen during training.
    # -----------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Held-out evaluation vs. deterministic TENO5')
    print('=' * 70)

    results = []

    # Case A: a jump profile with different center/width/amplitude than any
    # training sample (fixed, not drawn from sample_batch's rng stream).
    u0_a = square_wave(X, center=0.35, width=0.28, amplitude=0.6, offset=0.1)
    exact_a = square_wave(exact_shifted(X, TRAIN_STEPS_K * DT), 0.35, 0.28, 0.6, 0.1)
    results.append(evaluate_case(params, 'held-out jump, K steps', u0_a, exact_a, TRAIN_STEPS_K))

    # Case B: the same profile, advected for 3x the trained horizon K --
    # tests whether training generalizes beyond the unroll length it saw.
    long_steps = TRAIN_STEPS_K * 3
    exact_b = square_wave(exact_shifted(X, long_steps * DT), 0.35, 0.28, 0.6, 0.1)
    results.append(evaluate_case(params, 'held-out jump, 3x K steps', u0_a, exact_b, long_steps))

    # Case C: a smooth sine, held-out frequency -- sanity check (should be
    # close to a tie; smooth regions are where TENO5's candidate choice
    # matters least).
    u0_c = sine_wave(X, frequency=2.3, phase=0.7, amplitude=0.5, offset=0.0)
    exact_c = sine_wave(exact_shifted(X, TRAIN_STEPS_K * DT), 2.3, 0.7, 0.5, 0.0)
    results.append(evaluate_case(params, 'held-out smooth sine, K steps', u0_c, exact_c, TRAIN_STEPS_K))

    beats_count = sum(1 for beats, _, _ in results if beats)
    print('\nSummary: NN-TENO5(JAX, unrolled-loss-trained) beats deterministic TENO5 on '
          '{}/{} held-out cases.'.format(beats_count, len(results)))


if __name__ == '__main__':
    main()
