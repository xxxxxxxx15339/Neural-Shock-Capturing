# -*- coding: utf-8 -*-
"""Phase 0 feasibility spike (UNROLLED_PHYSICS_LOSS_GUIDE.md, Phase 0/1).

Two questions, in order:

1. Regression: does the JAX port (src/differentiable/teno5_jax.py,
   src/differentiable/numerics_jax.py) reproduce the existing NumPy solver
   (src/schemes/teno5.py, src/core/SimulationClasses.py) exactly, on both a
   single reconstruction call and a full multi-step rollout?

2. Differentiability: can `jax.grad` reach network weights through a
   reconstruction + K-step SSPRK3 rollout + loss chain? This is the actual
   question Phase 1 of the implementation guide exists to answer before any
   further work (feature vector, dataset, physics loss) is worth doing.

Run inside the `teno-jax` conda environment (jax + numpy + matplotlib only,
kept separate from the repo's pinned legacy Keras 2.2.4 / numpy 1.18.5
`weno-nn` environment used for training the real NN-TENO5 model):

    conda run -n teno-jax python scripts/unrolled_physics_loss/DiffSpike_TENO5_Advection.py
"""

import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax

# Must run before any module that creates a jnp array at import time (e.g.
# src.differentiable.teno5_jax's module-level LINEAR_WEIGHTS constant) --
# jax_enable_x64 only affects arrays created *after* this call. An array
# already materialized in float32 stays float32 forever no matter what the
# config says later; multiplying it into a float64 computation upcasts the
# dtype but not the precision that was already lost. This one line, in the
# wrong place, was the entire cause of the ~1e-7 regression mismatch the
# first time this script ran.
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

from src.core.Equations import adv
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.schemes import TENO5

from src.differentiable.numerics_jax import lax_friedrichs_flux, rollout
from src.differentiable.teno5_jax_toy import init_toy_params, toy_keep_weight
from src.differentiable.teno5_jax import (
    teno5_reconstruct_jax,
    teno5_smoothness_indicators,
    teno5_soft_reconstruct_jax,
)
from src.schemes.teno5 import teno5_reconstruct, teno5_smoothness_indicators as np_smoothness


def section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


# ---------------------------------------------------------------------------
# Part 1a: single-call regression, JAX deterministic port vs NumPy original.
# ---------------------------------------------------------------------------

section('1a. Single-stencil regression: teno5_jax vs teno5.py')

rng = np.random.default_rng(0)
random_stencils = rng.normal(size=(2000, 5))

jax_beta = np.asarray(teno5_smoothness_indicators(jnp.asarray(random_stencils)))
np_beta = np_smoothness(random_stencils)
beta_max_error = np.max(np.abs(jax_beta - np_beta))

jax_recon = np.asarray(teno5_reconstruct_jax(jnp.asarray(random_stencils)))
np_recon = teno5_reconstruct(random_stencils)
recon_max_error = np.max(np.abs(jax_recon - np_recon))

print('smoothness indicators, max abs error: {:.3e}'.format(beta_max_error))
print('reconstruction,        max abs error: {:.3e}'.format(recon_max_error))
part1a_pass = beta_max_error < 1e-10 and recon_max_error < 1e-10
print('PASS' if part1a_pass else 'FAIL')


# ---------------------------------------------------------------------------
# Part 1b: full-rollout regression, JAX solver vs the real NumPy Simulation.
# ---------------------------------------------------------------------------

section('1b. Full-rollout regression: JAX solver vs Simulation.run()')

nx = 64
cells = nx
final_time = 0.3
wave_speed = 1.0
cfl = 0.5
dx = 1.0 / cells
step_count = int(np.ceil(final_time * wave_speed / (cfl * dx)))
dt = final_time / step_count


def sine_ic(x):
    return np.sin(2.0 * np.pi * x)


x = np.linspace(0.0, 1.0, cells, endpoint=False)
u0_np = sine_ic(x)

# Reference: the real, unmodified NumPy solver.
numpy_sim = Simulation(
    cells,
    step_count + 1,
    1.0,
    final_time,
    SSPRK3(),
    LaxFriedrichs(adv(), wave_speed),
    TENO5(),
    sine_ic,
    max_cfl=cfl,
)
numpy_final = numpy_sim.run()[:, -1]

# JAX port, deterministic (hard cutoff) reconstruction.


def physical_flux(u):
    return u


def jax_flux_fn(u):
    return lax_friedrichs_flux(u, physical_flux, wave_speed, teno5_reconstruct_jax)


jax_final = np.asarray(rollout(jnp.asarray(u0_np), jax_flux_fn, dt, dx, step_count))

rollout_max_error = np.max(np.abs(jax_final - numpy_final))
print('grid size: {}, steps: {}, dt: {:.5f}, dx: {:.5f}'.format(cells, step_count, dt, dx))
print('final-state max abs error vs NumPy Simulation.run(): {:.3e}'.format(rollout_max_error))
part1b_pass = rollout_max_error < 1e-9
print('PASS' if part1b_pass else 'FAIL')


# ---------------------------------------------------------------------------
# Part 2: differentiability through a network-driven K-step rollout.
# ---------------------------------------------------------------------------

section('2. Gradient through reconstruction + K-step SSPRK3 rollout')

key = jax.random.PRNGKey(0)
params = init_toy_params(key)

u0_jax = jnp.asarray(u0_np)
exact_final = jnp.asarray(sine_ic(x - wave_speed * final_time))  # exact linear-advection solution


def soft_flux_fn(params, u):
    def reconstruct(stencils):
        keep_weight = toy_keep_weight(params, stencils)
        return teno5_soft_reconstruct_jax(stencils, keep_weight)

    return lax_friedrichs_flux(u, physical_flux, wave_speed, reconstruct)


def loss_fn(params):
    final_state = rollout(u0_jax, lambda u: soft_flux_fn(params, u), dt, dx, step_count)
    return jnp.mean((final_state - exact_final) ** 2)


loss_value = loss_fn(params)
print('K = {} substeps ({} SSPRK3 steps x 3 substeps each)'.format(step_count * 3, step_count))
print('initial loss (untrained network): {:.6e}'.format(float(loss_value)))

grads = jax.grad(loss_fn)(params)

print('\nper-layer gradient norms (weight matrix, bias vector):')
all_finite = True
all_nonzero = False
for i, (dw, db) in enumerate(grads):
    w_norm = float(jnp.linalg.norm(dw))
    b_norm = float(jnp.linalg.norm(db))
    print('  layer {}: |dW| = {:.6e}, |db| = {:.6e}'.format(i, w_norm, b_norm))
    if not (np.isfinite(w_norm) and np.isfinite(b_norm)):
        all_finite = False
    if w_norm > 0.0 or b_norm > 0.0:
        all_nonzero = True

part2_pass = all_finite and all_nonzero
print('\nall gradients finite: {}'.format(all_finite))
print('at least one nonzero gradient: {}'.format(all_nonzero))
print('PASS' if part2_pass else 'FAIL')


# ---------------------------------------------------------------------------
# Part 2b: repeat Part 2 on a discontinuous profile.
#
# A smooth sine barely exercises candidate *selection* -- away from a
# discontinuity, TENO5's three third-order candidates nearly agree, so the
# keep_weight the toy network assigns barely changes the output, and the
# gradients in Part 2 above are correspondingly tiny (~1e-9). That is exactly
# where the Unrolled Physics Loss proposal's plan does NOT need help: WENO/TENO already agree with
# each other and with the exact solution on smooth data. The regime that
# actually matters -- and that an NN-TENO5-style network exists to handle --
# is near a discontinuity, where candidate choice changes the answer a lot.
# Repeating the same gradient check there is a more honest test of whether
# training signal will actually be usable.
# ---------------------------------------------------------------------------

section('2b. Same gradient check, on a discontinuous (square-wave) profile')


def square_wave_ic(x):
    return np.where((x > 0.25) & (x < 0.75), 1.0, -1.0)


u0_step_np = square_wave_ic(x)
u0_step_jax = jnp.asarray(u0_step_np)
exact_step_final = jnp.asarray(square_wave_ic((x - wave_speed * final_time) % 1.0))


def loss_fn_step(params):
    final_state = rollout(u0_step_jax, lambda u: soft_flux_fn(params, u), dt, dx, step_count)
    return jnp.mean((final_state - exact_step_final) ** 2)


loss_step_value = loss_fn_step(params)
print('initial loss (untrained network), discontinuous IC: {:.6e}'.format(float(loss_step_value)))

grads_step = jax.grad(loss_fn_step)(params)

print('\nper-layer gradient norms (discontinuous IC):')
step_all_finite = True
step_all_nonzero = False
for i, (dw, db) in enumerate(grads_step):
    w_norm = float(jnp.linalg.norm(dw))
    b_norm = float(jnp.linalg.norm(db))
    print('  layer {}: |dW| = {:.6e}, |db| = {:.6e}'.format(i, w_norm, b_norm))
    if not (np.isfinite(w_norm) and np.isfinite(b_norm)):
        step_all_finite = False
    if w_norm > 0.0 or b_norm > 0.0:
        step_all_nonzero = True

part2b_pass = step_all_finite and step_all_nonzero
smooth_grad_norm = float(jnp.linalg.norm(grads[0][0]))
step_grad_norm = float(jnp.linalg.norm(grads_step[0][0]))
print('\nlayer-0 |dW| on smooth IC       : {:.3e}'.format(smooth_grad_norm))
print('layer-0 |dW| on discontinuous IC: {:.3e}  ({:.0f}x larger)'.format(
    step_grad_norm, step_grad_norm / max(smooth_grad_norm, 1e-30)))
print('PASS' if part2b_pass else 'FAIL')


# ---------------------------------------------------------------------------
# Part 3: one gradient-descent step actually reduces the loss.
# ---------------------------------------------------------------------------

section('3. Sanity check: does one gradient step reduce the loss?')

learning_rate = 1.0e-2
new_params = [
    (w - learning_rate * dw, b - learning_rate * db)
    for (w, b), (dw, db) in zip(params, grads)
]
new_loss = float(loss_fn(new_params))
print('loss before step: {:.6e}'.format(float(loss_value)))
print('loss after  step: {:.6e}'.format(new_loss))
part3_pass = new_loss < float(loss_value)
print('PASS' if part3_pass else 'FAIL (not necessarily a bug -- one step can overshoot; rerun with a smaller lr)')


# ---------------------------------------------------------------------------
# Part 4: does the loss actually keep decreasing over many steps, not just one?
# ---------------------------------------------------------------------------

section('4. 50-step training loop on the discontinuous profile')

train_params = params
train_lr = 3.0e-2
loss_history = []
for i in range(50):
    current_loss, current_grads = jax.value_and_grad(loss_fn_step)(train_params)
    loss_history.append(float(current_loss))
    train_params = [
        (w - train_lr * dw, b - train_lr * db)
        for (w, b), (dw, db) in zip(train_params, current_grads)
    ]

print('loss[0]  = {:.6e}'.format(loss_history[0]))
print('loss[9]  = {:.6e}'.format(loss_history[9]))
print('loss[49] = {:.6e}'.format(loss_history[49]))
part4_pass = loss_history[-1] < loss_history[0] and np.isfinite(loss_history).all()
print('PASS' if part4_pass else 'FAIL')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section('Summary')
print('1a. JAX/NumPy single-call regression match      : {}'.format('PASS' if part1a_pass else 'FAIL'))
print('1b. JAX/NumPy full-rollout regression match      : {}'.format('PASS' if part1b_pass else 'FAIL'))
print('2.  Finite, nonzero gradient (smooth IC)         : {}'.format('PASS' if part2_pass else 'FAIL'))
print('2b. Finite, nonzero gradient (discontinuous IC)  : {}'.format('PASS' if part2b_pass else 'FAIL'))
print('3.  Gradient step reduces the loss               : {}'.format('PASS' if part3_pass else 'FAIL'))
print('4.  50-step training loop converges              : {}'.format('PASS' if part4_pass else 'FAIL'))

overall = part1a_pass and part1b_pass and part2_pass and part2b_pass and part3_pass and part4_pass
print('\nPhase 0 exit criterion: {}'.format('MET' if overall else 'NOT MET'))
