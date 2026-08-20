# -*- coding: utf-8 -*-
"""Phase 1 Euler extension of DiffSpike_TENO5_Advection.py.

Answers the same two questions as the scalar spike, on the Euler system
instead: does src/differentiable/euler_jax.py reproduce the existing NumPy
Euler solver, and can jax.grad reach network weights through a Sod rollout?

Sod is discontinuous by construction, so unlike the scalar spike there is no
separate "smooth IC" vs "discontinuous IC" split needed here -- every check
below already exercises the regime that matters.

Run inside the `teno-jax` conda environment:

    conda run -n teno-jax python DiffSpike_TENO5euler_Sod.py
"""

import jax

jax.config.update('jax_enable_x64', True)

import time

import jax.numpy as jnp
import numpy as np

from src.analysis.euler_regression import exact_sod_solution
from src.core.eulerEquations import getEulerFlux
from src.core.SimulationClasses import eulerSimulation
from src.core.TimeSteppingMethods import SSPRK3
from src.initial_conditions.InitialConditions import sod
from src.schemes import TENO5euler

from src.differentiable.euler_jax import get_euler_flux_jax
from src.differentiable.numerics_jax import rollout
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import init_toy_params, toy_keep_weight


def section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


cells = 100
length = 1.0
final_time = 0.2
cfl = 0.5
max_wave_speed = 2.0

dx = length / cells
dt = cfl * dx / max_wave_speed
step_count = int(np.ceil(final_time / dt))
nt = step_count + 1

x = np.arange(cells, dtype=float) * length / cells
initial_np = sod()(x)


# ---------------------------------------------------------------------------
# Part 1: full Sod rollout regression, JAX vs the real NumPy eulerSimulation.
# ---------------------------------------------------------------------------

section('1. Sod rollout regression: euler_jax.py vs eulerSimulation.runEuler()')

numpy_sim = eulerSimulation(
    cells, nt, length, final_time, SSPRK3(),
    getEulerFlux(TENO5euler()), sod(), 3,
    max_wave_speed=max_wave_speed, max_cfl=cfl,
)
numpy_final = numpy_sim.runEuler()[:, -1, :]
actual_dt = numpy_sim.last_times[1] - numpy_sim.last_times[0]

jax_flux_fn = get_euler_flux_jax(teno5_reconstruct_jax, boundary='transmissive')
jax_final = np.asarray(
    rollout(jnp.asarray(initial_np), jax_flux_fn, actual_dt, dx, step_count)
)

rollout_max_error = np.max(np.abs(jax_final - numpy_final))
print('grid size: {}, steps: {}, dt: {:.6f}, dx: {:.6f}'.format(cells, step_count, actual_dt, dx))
print('final-state max abs error vs NumPy eulerSimulation.runEuler(): {:.3e}'.format(rollout_max_error))
part1_pass = rollout_max_error < 1e-9
print('PASS' if part1_pass else 'FAIL')


# ---------------------------------------------------------------------------
# Part 2: gradient through a network-driven Sod rollout.
#
# Reuses src/differentiable/teno5_jax_toy.py completely unmodified. It was
# written for the scalar (nx, 5) case; here it's called on (n, 3, 5)
# characteristic stencils (n interfaces x 3 characteristic fields x 5-point
# stencil) and works without changes, because every operation in it
# (min/max normalization over the last axis, matmul against (5, hidden)
# weight matrices) broadcasts correctly over the extra characteristic-field
# axis via ordinary JAX/NumPy batched-matmul rules. This is the payoff of
# having kept teno5_soft_reconstruct_jax and combine_teno_candidates_jax
# shape-agnostic in Phase 0: nothing Euler-specific had to be written for
# the network side of this check.
# ---------------------------------------------------------------------------

section('2. Gradient through reconstruction + Sod rollout')

key = jax.random.PRNGKey(0)
params = init_toy_params(key)

initial_jax = jnp.asarray(initial_np)
exact_final = jnp.asarray(exact_sod_solution(x, final_time))


def soft_euler_flux_fn(params, u):
    def reconstruct(stencils):
        keep_weight = toy_keep_weight(params, stencils)
        return teno5_soft_reconstruct_jax(stencils, keep_weight)

    full_flux = get_euler_flux_jax(reconstruct, boundary='transmissive')
    return full_flux(u)


def loss_fn(params):
    final_state = rollout(initial_jax, lambda u: soft_euler_flux_fn(params, u), actual_dt, dx, step_count)
    return jnp.mean((final_state - exact_final) ** 2)


print('K = {} substeps ({} SSPRK3 steps x 3 substeps each)'.format(step_count * 3, step_count))

# The scalar spike (DiffSpike_TENO5_Advection.py) got away with an un-jitted
# rollout: the flux function there is a handful of jnp ops per substep. The
# Euler flux function is much heavier per substep -- a batched 3x3 Roe
# eigenbasis inversion, two einsum projections, several stencil gathers --
# and un-jitted JAX pays real per-op dispatch overhead for every one of
# those, every substep, every training step. That combination made the
# first version of this script's 20-step training loop take upwards of ten
# minutes and counting before being killed; jax.jit brought a single
# value-and-grad call down to about the time printed below. This is the
# concrete case the "unrolled loop now, lax.scan/jit later" note in
# PROJ_IMPLEMENTATION_GUIDE.md's Phase 1 writeup was anticipating -- "later"
# turned out to mean "as soon as the flux function has real linear algebra
# in it," which is Euler, not scalar advection.
value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

compile_started = time.perf_counter()
loss_value, grads = value_and_grad_fn(params)
loss_value.block_until_ready()
compile_seconds = time.perf_counter() - compile_started
print('first (compiling) value_and_grad call: {:.1f}s'.format(compile_seconds))

warm_started = time.perf_counter()
loss_value, grads = value_and_grad_fn(params)
loss_value.block_until_ready()
warm_seconds = time.perf_counter() - warm_started
print('second (compiled) value_and_grad call: {:.2f}s'.format(warm_seconds))

print('initial loss (untrained network) vs exact Riemann solution: {:.6e}'.format(float(loss_value)))

print('\nper-layer gradient norms:')
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
# Part 3: short training loop.
# ---------------------------------------------------------------------------

section('3. 20-step training loop on Sod')

train_started = time.perf_counter()
train_params = params
train_lr = 3.0e-2
loss_history = []
for i in range(20):
    current_loss, current_grads = value_and_grad_fn(train_params)
    loss_history.append(float(current_loss))
    train_params = [
        (w - train_lr * dw, b - train_lr * db)
        for (w, b), (dw, db) in zip(train_params, current_grads)
    ]

train_seconds = time.perf_counter() - train_started
print('20 steps took {:.1f}s total ({:.2f}s/step, jitted)'.format(train_seconds, train_seconds / 20))
print('loss[0]  = {:.6e}'.format(loss_history[0]))
print('loss[9]  = {:.6e}'.format(loss_history[9]))
print('loss[19] = {:.6e}'.format(loss_history[19]))
part3_pass = loss_history[-1] < loss_history[0] and np.isfinite(loss_history).all()
print('PASS' if part3_pass else 'FAIL')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section('Summary')
print('1. JAX/NumPy Sod rollout regression match : {}'.format('PASS' if part1_pass else 'FAIL'))
print('2. Finite, nonzero gradient through rollout: {}'.format('PASS' if part2_pass else 'FAIL'))
print('3. 20-step training loop converges         : {}'.format('PASS' if part3_pass else 'FAIL'))

overall = part1_pass and part2_pass and part3_pass
print('\nPhase 1 Euler exit criterion: {}'.format('MET' if overall else 'NOT MET'))
