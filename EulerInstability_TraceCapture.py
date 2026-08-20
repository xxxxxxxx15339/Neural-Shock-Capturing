# -*- coding: utf-8 -*-
"""Capture the Euler training-instability trace referenced in
PROJ_IMPLEMENTATION_GUIDE.md's Phase 6 status note and Results.md Phase 10,
for plotting (plots/04_euler_instability.png). Reruns the same steep (~p=9)
held-out-batch example that first exposed the instability, at several
Lax-Friedrichs alpha safety margins, and records max density per SSPRK3
step.

Deliberately uses a FROZEN COPY of the original clamp-only positivity
guard (`_euler_positivity_guard_v1` below), not the current
`euler_jax.euler_positivity_repair` (which replaced it after the professor's
clarification, Results.md Phase 11) -- this script exists specifically to
reproduce the historical finding about the *old* guard's insufficiency
(clamping alone delays but doesn't prevent the blowup, and more dissipation
makes it worse). Pointing it at the new repair function would silently
change what this script measures and invalidate the plot it documents.

Run inside `teno-jax`:

    conda run -n teno-jax python -u EulerInstability_TraceCapture.py
"""

import csv

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import pickle

from src.differentiable.euler_jax import (
    GAMMA,
    flux_jax,
    project_to_characteristic_jax,
    project_to_conservative_jax,
    roe_eigenbasis_jax,
    spds_jax,
    euler_stencils_jax,
)
from src.differentiable.numerics_jax import ssprk3_step

import Train_NNTENO5euler_UnrolledJAX as T


def _euler_positivity_guard_v1(u, eps=5.0e-2, max_velocity=50.0):
    """Frozen copy of the original euler_jax.euler_positivity_guard (a
    blind clamp, not the neighbor-average repair that replaced it) -- see
    module docstring for why this script keeps its own copy.
    """
    density = jnp.maximum(u[:, 0], eps)
    velocity = jnp.clip(u[:, 1] / density, -max_velocity, max_velocity)
    pressure = jnp.maximum((GAMMA - 1) * (u[:, 2] - 0.5 * density * velocity ** 2), eps)
    momentum = density * velocity
    energy = pressure / (GAMMA - 1) + 0.5 * density * velocity ** 2
    return jnp.stack([density, momentum, energy], axis=-1)


def get_euler_flux_with_margin(reconstruct, boundary='transmissive', alpha_margin=1.0):
    def full_flux(u):
        fu = flux_jax(u)
        c = jnp.max(jnp.abs(spds_jax(u)), axis=1)
        right_speed = jnp.concatenate([c[1:], c[-1:]])
        alpha = alpha_margin * jnp.maximum(c, right_speed)
        left_basis, right_basis = roe_eigenbasis_jax(u, boundary=boundary)
        u_part = euler_stencils_jax(u, offset=0, boundary=boundary)
        fu_part = euler_stencils_jax(fu, offset=0, boundary=boundary)
        u_part_neg = jnp.flip(euler_stencils_jax(u, offset=1, boundary=boundary), axis=2)
        fu_part_neg = jnp.flip(euler_stencils_jax(fu, offset=1, boundary=boundary), axis=2)
        wp = project_to_characteristic_jax(u_part, left_basis)
        fwp = project_to_characteristic_jax(fu_part, left_basis)
        wm = project_to_characteristic_jax(u_part_neg, left_basis)
        fwm = project_to_characteristic_jax(fu_part_neg, left_basis)
        f_pos = 0.5 * (fwp + alpha[:, None, None] * wp)
        f_neg = 0.5 * (fwm - alpha[:, None, None] * wm)
        f_half_pos = reconstruct(f_pos)
        f_half_neg = reconstruct(f_neg)
        flux_char = f_half_pos + f_half_neg
        flux_cons = project_to_conservative_jax(flux_char, right_basis)
        left_edge = flux_cons[0:1, :] - flux_jax(u[0:1, :])
        interior = flux_cons[1:, :] - flux_cons[:-1, :]
        return jnp.concatenate([left_edge, interior], axis=0)
    return full_flux


with open('models/nn_teno5_unrolled_jax_params_run3_pretrained.pkl', 'rb') as f:
    loaded = pickle.load(f)
params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in loaded]

rng = np.random.default_rng(2)
batch_u0, _ = T.sample_batch(rng, 8, T.TRAIN_STEPS_K)
u0 = batch_u0[7]  # the steep (~p=9) example that first exposed the instability


def network_reconstruct(stencils):
    return T._network_reconstruct(params, stencils, mode='soft')


rows = []
for margin in [1.0, 1.5, 2.0, 3.0, 5.0]:
    flux_fn = get_euler_flux_with_margin(network_reconstruct, alpha_margin=margin)
    u = u0
    for step in range(T.TRAIN_STEPS_K):
        u = ssprk3_step(u, flux_fn, T.DT, T.DX, state_guard=_euler_positivity_guard_v1)
        finite = bool(jnp.isfinite(u).all())
        max_density = float(jnp.max(u[:, 0])) if finite else float('nan')
        rows.append({'alpha_margin': margin, 'step': step, 'max_density': max_density, 'finite': finite})
        print('margin={:.1f} step={:2d} max_density={:.3e} finite={}'.format(
            margin, step, max_density, finite))
        if not finite:
            break

with open('models/euler_instability_trace.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['alpha_margin', 'step', 'max_density', 'finite'])
    writer.writeheader()
    writer.writerows(rows)
print('\nTrace written to models/euler_instability_trace.csv')
