# -*- coding: utf-8 -*-
"""JAX port of the characteristic-wise Euler solver.

Lives under `src/differentiable/` for the same import-isolation reason as
`teno5_jax.py` -- see `src/differentiable/__init__.py`.

Ports, one function each, all from `src/core/eulerEquations.py`:

- `flux_jax`                  <- flux                  (eulerEquations.py:55)
- `spds_jax`                  <- spds                  (eulerEquations.py:66)
- `roe_eigenbasis_jax`        <- roe_eigenbasis         (eulerEquations.py:16)
- `project_to_characteristic_jax` <- project_to_characteristic (eulerEquations.py:48)
- `project_to_conservative_jax`   <- project_to_conservative   (eulerEquations.py:52)
- `euler_stencils_jax`         <- FiniteVolumeMethodEuler.partU (SimulationClasses.py:212)
- `get_euler_flux_jax`         <- getEulerFlux                  (eulerEquations.py:79)

`ssprk3_step` and `rollout` from `numerics_jax.py` are reused unchanged for
the Euler system -- both are already shape-agnostic elementwise arithmetic
over whatever `flux_fn` returns, so nothing Euler-specific was needed there.

One deliberate simplification versus the original: `getEulerFlux`'s inner
`for i in range(0, m)` loop over the 3 characteristic fields is written out
as a single vectorized expression here. The loop was already fixed at
exactly 3 iterations with no data-dependent trip count (it's inside a
function that always processes a (n, 3, 5)-shaped array); `alpha` only
varies over the leading interface axis, so broadcasting it as
`alpha[:, None, None]` against `(n, 3, 5)` reproduces the loop's per-field
assignment exactly, with one fewer moving part.
"""

import jax.numpy as jnp

GAMMA = 1.4


def flux_jax(u):
    """Port of eulerEquations.py:55 flux."""
    g = GAMMA
    u1 = u[:, 0]
    u2 = u[:, 1]
    u3 = u[:, 2]
    f0 = u2
    f1 = 0.5 * (3 - g) * u2 ** 2 / u1 + (g - 1) * u3
    f2 = g * u2 * u3 / u1 - 0.5 * (g - 1) * u2 ** 3 / u1 ** 2
    return jnp.stack([f0, f1, f2], axis=-1)


def spds_jax(ws):
    """Port of eulerEquations.py:66 spds."""
    g = GAMMA
    rho = ws[:, 0]
    vel = ws[:, 1] / ws[:, 0]
    energy = ws[:, 2]
    pressure = (g - 1) * (energy - 0.5 * rho * vel ** 2)
    sound_speed = jnp.sqrt(g * pressure / rho)
    return jnp.stack([vel, vel + sound_speed, vel - sound_speed], axis=-1)


def roe_eigenbasis_jax(u, boundary='transmissive'):
    """Port of eulerEquations.py:16 roe_eigenbasis."""
    if boundary == 'periodic':
        right_state = jnp.roll(u, -1, axis=0)
    elif boundary == 'transmissive':
        right_state = jnp.concatenate([u[1:, :], u[-1:, :]], axis=0)
    else:
        raise ValueError('Unsupported Euler boundary: {}'.format(boundary))

    rho_l, rho_r = u[:, 0], right_state[:, 0]
    vel_l, vel_r = u[:, 1] / rho_l, right_state[:, 1] / rho_r
    p_l = (GAMMA - 1) * (u[:, 2] - 0.5 * rho_l * vel_l ** 2)
    p_r = (GAMMA - 1) * (right_state[:, 2] - 0.5 * rho_r * vel_r ** 2)
    h_l = (u[:, 2] + p_l) / rho_l
    h_r = (right_state[:, 2] + p_r) / rho_r
    root_l, root_r = jnp.sqrt(rho_l), jnp.sqrt(rho_r)
    denominator = root_l + root_r
    velocity = (root_l * vel_l + root_r * vel_r) / denominator
    enthalpy = (root_l * h_l + root_r * h_r) / denominator
    sound_speed = jnp.sqrt((GAMMA - 1) * (enthalpy - 0.5 * velocity ** 2))

    ones = jnp.ones_like(velocity)
    zeros = jnp.zeros_like(velocity)
    right_row0 = jnp.stack([ones, ones, ones], axis=-1)
    right_row1 = jnp.stack(
        [velocity - sound_speed, velocity, velocity + sound_speed], axis=-1
    )
    right_row2 = jnp.stack(
        [
            enthalpy - velocity * sound_speed,
            0.5 * velocity ** 2,
            enthalpy + velocity * sound_speed,
        ],
        axis=-1,
    )
    right = jnp.stack([right_row0, right_row1, right_row2], axis=1)
    left = jnp.linalg.inv(right)
    return left, right


def project_to_characteristic_jax(values, left_eigenvectors):
    """Port of eulerEquations.py:48 project_to_characteristic."""
    return jnp.einsum('nij,njk->nik', left_eigenvectors, values)


def project_to_conservative_jax(values, right_eigenvectors):
    """Port of eulerEquations.py:52 project_to_conservative."""
    return jnp.einsum('nij,nj->ni', right_eigenvectors, values)


def euler_stencils_jax(u, offset=0, stencil_size=5, boundary='transmissive'):
    """Port of FiniteVolumeMethodEuler.partU (SimulationClasses.py:212)."""
    cell_count = u.shape[0]
    radius = stencil_size // 2
    base = (
        jnp.arange(cell_count)[:, None]
        + offset
        + jnp.arange(-radius, radius + 1)[None, :]
    )
    if boundary == 'periodic':
        indices = base % cell_count
    else:
        indices = jnp.clip(base, 0, cell_count - 1)
    gathered = u[indices, :]
    return jnp.transpose(gathered, (0, 2, 1))


def euler_characteristic_state_stencils(u, boundary='transmissive'):
    """(n, 3, 5) characteristic-projected state stencils, forward direction.

    Reuses `get_euler_flux_jax`'s own Roe-basis/projection steps (its
    `u_part`/`wp`) without the rest of the flux-split machinery -- for
    callers that need characteristic stencils on their own, e.g. Phase 5's
    `L_BVD`, which needs a (..., 5)-shaped stencil array to compute an
    interface jump on, the same way it does for the scalar case.
    """
    left_basis, _ = roe_eigenbasis_jax(u, boundary=boundary)
    u_part = euler_stencils_jax(u, offset=0, boundary=boundary)
    return project_to_characteristic_jax(u_part, left_basis)


def euler_boundary_variation_jax(u, reconstruct, boundary='transmissive'):
    """True BVD boundary-variation jump at every Euler interface, in
    characteristic space: |Phi_i(x_{i+1/2}) - Phi_{i+1}(x_{i+1/2})}|, the
    Euler analogue of numerics_jax.boundary_variation_jax (per the
    professor's clarification and Sun, Inaba & Xiao 2016, Eq. 7: BVD
    compares each cell's OWN candidate reconstruction against its
    neighbor's own reconstruction, not a single stencil against its own
    mirror image). Reuses `get_euler_flux_jax`'s own forward/mirror
    characteristic-stencil construction (its `wp`/`wm`) rather than
    reinventing it -- `wp` reconstructed is cell i's own right-face
    characteristic value; `wm` reconstructed is cell (i+1)'s own left-face
    characteristic value, both already correct in that function, just not
    previously exposed on their own.
    """
    left_basis, _ = roe_eigenbasis_jax(u, boundary=boundary)
    u_part = euler_stencils_jax(u, offset=0, boundary=boundary)
    u_part_neg = jnp.flip(
        euler_stencils_jax(u, offset=1, boundary=boundary), axis=2
    )
    wp = project_to_characteristic_jax(u_part, left_basis)
    wm = project_to_characteristic_jax(u_part_neg, left_basis)
    forward = reconstruct(wp)
    mirrored = reconstruct(wm)
    return jnp.abs(forward - mirrored)


def euler_split_flux_stencils(u, boundary='transmissive', return_basis=False):
    """Exact positive/negative characteristic LF inputs used by the solver.

    Each array has shape (interfaces, 3 fields, 5 stencil values). The negative
    direction includes the solver's mirror and offset. No reconstruction occurs.
    return_basis is used internally to avoid recomputing the Roe eigenbasis.
    """
    fu = flux_jax(u)
    c = jnp.max(jnp.abs(spds_jax(u)), axis=1)
    if boundary == 'periodic':
        right_speed = jnp.roll(c, -1)
    else:
        right_speed = jnp.concatenate([c[1:], c[-1:]])
    alpha = jnp.maximum(c, right_speed)

    left_basis, right_basis = roe_eigenbasis_jax(u, boundary=boundary)

    u_part = euler_stencils_jax(u, offset=0, boundary=boundary)
    fu_part = euler_stencils_jax(fu, offset=0, boundary=boundary)
    u_part_neg = jnp.flip(
        euler_stencils_jax(u, offset=1, boundary=boundary), axis=2
    )
    fu_part_neg = jnp.flip(
        euler_stencils_jax(fu, offset=1, boundary=boundary), axis=2
    )

    wp = project_to_characteristic_jax(u_part, left_basis)
    fwp = project_to_characteristic_jax(fu_part, left_basis)
    wm = project_to_characteristic_jax(u_part_neg, left_basis)
    fwm = project_to_characteristic_jax(fu_part_neg, left_basis)

    # alpha varies only over the leading (interface) axis -- see the
    # module docstring for why this replaces the original per-field loop.
    f_pos = 0.5 * (fwp + alpha[:, None, None] * wp)
    f_neg = 0.5 * (fwm - alpha[:, None, None] * wm)

    if return_basis:
        return f_pos, f_neg, right_basis
    return f_pos, f_neg


def get_euler_flux_jax(reconstruct, boundary='transmissive', return_boundary_flux=False):
    """Port of eulerEquations.py:79 getEulerFlux.

    `reconstruct` plays the role of `FVM.evalF`/`FVM.L`: it maps a
    (n, 3, 5) characteristic-stencil array to (n, 3) reconstructed
    interface values. Pass `teno5_reconstruct_jax` for the deterministic
    scheme, or a network-driven closure once Phase 4/5 wire one in.

    With return_boundary_flux=True also expose (left, right) numerical
    boundary fluxes for conservation accounting. The default arithmetic
    and return value are unchanged.
    """

    def full_flux(u):
        if boundary == 'reflective':
            # Three ghost cells supply complete five-point stencils at both
            # physical wall interfaces. Mirror normal momentum, retain rho/E.
            sign = jnp.array([1., -1., 1.], dtype=u.dtype)
            padded = jnp.concatenate((jnp.flip(u[:3],axis=0)*sign,u,
                                      jnp.flip(u[-3:],axis=0)*sign),axis=0)
            pos, neg, basis = euler_split_flux_stencils(padded, 'transmissive', True)
            faces = project_to_conservative_jax(reconstruct(pos)+reconstruct(neg),basis)
            n = u.shape[0]
            net = faces[3:n+3]-faces[2:n+2]
            return (net, (faces[2],faces[n+2])) if return_boundary_flux else net
        f_pos, f_neg, right_basis = euler_split_flux_stencils(u, boundary, return_basis=True)

        f_half_pos = reconstruct(f_pos)
        f_half_neg = reconstruct(f_neg)
        flux_char = f_half_pos + f_half_neg
        flux_cons = project_to_conservative_jax(flux_char, right_basis)

        if boundary == 'periodic':
            net_flux = flux_cons - jnp.roll(flux_cons, 1, axis=0)
        else:
            left_edge = flux_cons[0:1, :] - flux_jax(u[0:1, :])
            interior = flux_cons[1:, :] - flux_cons[:-1, :]
            net_flux = jnp.concatenate([left_edge, interior], axis=0)
        if return_boundary_flux:
            left = flux_cons[-1] if boundary == 'periodic' else flux_jax(u[0:1, :])[0]
            return net_flux, (left, flux_cons[-1])
        return net_flux

    return full_flux


def get_euler_state_flux_jax(reconstruct, boundary='periodic', return_boundary_flux=False):
    """Experimental finite-volume state reconstruction + Rusanov flux.

    The historical split-flux solver treats F(cell-average U) as a cell-average
    flux. These differ at O(dx^2) for general nonlinear smooth Euler data.
    Reconstructing characteristic states at faces before evaluating F supplies a
    distinct finite-volume variant. No positivity guarantee or default change.
    """
    if boundary != 'periodic':
        raise ValueError('Experimental state-flux variant currently supports periodic boundaries only')

    def flux(u):
        left, right = roe_eigenbasis_jax(u, boundary)
        plus = project_to_characteristic_jax(euler_stencils_jax(u,boundary=boundary),left)
        minus = project_to_characteristic_jax(jnp.flip(euler_stencils_jax(u,offset=1,boundary=boundary),axis=2),left)
        ul = project_to_conservative_jax(reconstruct(plus),right)
        ur = project_to_conservative_jax(reconstruct(minus),right)
        bound = jnp.maximum(jnp.max(jnp.abs(spds_jax(ul)),axis=1),jnp.max(jnp.abs(spds_jax(ur)),axis=1))
        faces = .5*(flux_jax(ul)+flux_jax(ur)-bound[:,None]*(ur-ul))
        net = faces-jnp.roll(faces,1,axis=0)
        return (net,(faces[-1],faces[-1])) if return_boundary_flux else net
    return flux


def positivity_penalty_jax(u):
    """Differentiable stand-in for eulerSimulation._validate_state's raise.

    `_validate_state` (SimulationClasses.py:65) raises FloatingPointError on
    non-positive density/pressure. A `jax.lax`-traced rollout cannot raise
    mid-trace, and Phase 5's loss terms need a differentiable signal instead
    of a hard stop anyway -- this returns 0 exactly when density and
    pressure are both positive everywhere, and a smooth positive penalty
    proportional to how far into the invalid region the state has gone
    otherwise.

    Per the professor's clarification (see `euler_positivity_repair`): an
    isolated violating cell is a repairable glitch, but if several occur
    simultaneously that signals the scheme itself is unstable -- this
    penalty is what actually tells training that happened, since the
    repair function keeps the rollout numerically finite either way and
    would otherwise hide the difference. Wire this into
    `losses_jax.unrolled_physics_loss` via `lambda_positivity`; it was
    computed in Phase 5 but never actually wired into a training loss
    before this correction.

    Divides by a guarded density (not raw `u[:, 0]`) so this function is
    itself safe to call on a state that is already violating -- exactly
    the state it exists to detect -- without its own division blowing up
    the gradient the same way the unguarded rollout did.

    Also guards non-finite rows (not just negative/zero density), needed
    once `numerics_jax.rollout_with_positivity_penalty` started calling
    this on PRE-`state_guard` substages: those can already contain literal
    NaN/Inf (the state having blown up past floating-point range before
    `euler_positivity_repair` gets a chance to run), not just a negative
    density this function's original division guard alone handles. Without
    this, one non-finite substage in one rollout out of a whole training
    batch poisoned `l_positivity`, and through it the batch-mean total
    loss, to NaN outright -- the same "guard the input, not just the
    untaken branch" lesson as `teno5_jax_toy.py`'s `toy_keep_weight`
    docstring: `jax.grad` computes both `jnp.where` branches' gradients
    unconditionally, so a NaN produced downstream of the untaken branch
    still poisons the backward pass even though its forward value is
    discarded. Substituting a whole finite, clearly-violating row
    (density -1, pressure +1 after the formula below) for any non-finite
    row, before any arithmetic reads it, keeps every downstream value AND
    gradient finite while still counting that row as a real violation.

    Also caps each cell's contribution at `POSITIVITY_VIOLATION_CAP`: a
    genuinely astronomic (but finite) pre-repair excursion -- Results.md's
    own traced failure reached a max state value of 5.7e144 before finally
    going non-finite -- produces a violation depth of the same astronomic
    order, and an uncapped per-cell sum at that scale overflows the
    batch-mean total loss to `inf` on its own even though this function's
    own output was itself still finite. The cap turns "how severely did
    this blow up" into a saturating (bounded, still gradient-informative
    below the cap) rather than unbounded signal -- same scale-mismatch
    reasoning as `losses_jax.l_tv`/`l_reg`'s `normalize` options, applied
    here as a hard cap instead of a division since there's no natural
    per-example denominator to normalize a positivity violation by.
    """
    finite_row = jnp.all(jnp.isfinite(u), axis=-1)
    safe_u = jnp.where(finite_row[:, None], u, jnp.array([-1.0, 0.0, 1.0], dtype=u.dtype))
    density = safe_u[:, 0]
    safe_density = jnp.where(density > 0.0, density, 1.0)
    velocity = safe_u[:, 1] / safe_density
    pressure = (GAMMA - 1) * (safe_u[:, 2] - 0.5 * density * velocity ** 2)
    density_violation = jnp.minimum(jnp.maximum(-density, 0.0), POSITIVITY_VIOLATION_CAP)
    pressure_violation = jnp.minimum(jnp.maximum(-pressure, 0.0), POSITIVITY_VIOLATION_CAP)
    return jnp.sum(density_violation) + jnp.sum(pressure_violation)


POSITIVITY_VIOLATION_CAP = 1.0e3
# Shared with `euler_positivity_repair`'s `max_scale`: the same "far beyond
# the problem's O(1)-O(10) scale, but not astronomic" bound, used two
# places for two different reasons -- see each function's docstring.


def euler_positivity_repair(u, eps=5.0e-2, max_velocity=50.0, max_scale=POSITIVITY_VIOLATION_CAP,
                            return_diagnostics=False):
    """Repair non-positive density/pressure the way the professor
    describes doing it in practice: an ISOLATED violating cell (both
    immediate neighbors still valid) is repaired by averaging its
    neighbors' conservative state; several simultaneously violating cells
    means the scheme itself is unstable, not a repairable single-point
    glitch, and should be surfaced to training via `positivity_penalty_jax`
    (wired into the loss as `lambda_positivity`) rather than silently
    patched over.

    This function still floors any cell the neighbor-average doesn't fix
    (an isolated repair that's itself still invalid, or a genuine cluster)
    as a last resort, purely to keep `jax.lax.scan` tracing finite --
    unlike the neighbor-average repair, this floor is not claimed to be
    physically meaningful, which is exactly why `positivity_penalty_jax`
    needs to be watching separately rather than trusting this function
    alone to signal correctness.

    P0 correction: apply the last-resort path only to rows still invalid
    after neighbor averaging. Admissible rows (including positive values
    below eps) are returned unchanged. Neighbor averaging and the existing
    fallback are NOT globally conservative; evaluation must report their
    conservative-state changes. Neighbor selection remains periodic, as in
    the historical training implementation; this is not a boundary-aware
    positivity-preserving numerical scheme.

    `eps` defaults to `5e-2`, not a generic near-zero value like `1e-8`,
    and the last-resort path also caps velocity -- both for the same
    reason discovered the hard way in the function this one replaces
    (`euler_positivity_guard`, see Results.md Phase 10): local sound speed
    scales like `sqrt(pressure/density)`, so flooring density to something
    many orders of magnitude below the problem's actual O(1) scale makes
    sound speed (and, without a velocity cap, `velocity = momentum/density`
    right where density was just floored) explode instead of stabilizing
    anything. A first version of this function used `eps=1e-8` with no
    velocity cap on the last-resort path and made the K=2 curriculum stage
    diverge *earlier* than before this fix (step 37 of 60, vs. completing
    all 60 previously) -- a regression, caught by rerunning the curriculum
    experiment immediately after making the change, not assumed correct.

    Replaces an earlier version (`euler_positivity_guard`) that
    unconditionally floored density and clipped velocity everywhere,
    including at cells that were never in violation -- a blunter
    instrument than what was actually asked for, and the one this project
    measured (Results.md Phase 10/11) to only delay, not prevent, a
    training-time blowup on its own.

    Two more failure modes now feed `violates`, found while wiring
    `positivity_penalty_jax` into a real pre-repair training signal
    (`numerics_jax.rollout_with_positivity_penalty`) and discovering this
    function did not actually catch either one, contrary to its own "keep
    jax.lax.scan tracing finite" claim: (1) a row that's already NaN/Inf --
    `NaN <= 0.0` is `False`, so a fully non-finite row was never flagged as
    violating, and the last-resort floor below (`jnp.maximum(NaN, eps)` is
    still `NaN`) does not fix it either; (2) a row that's technically
    positive but has blown up to an astronomic magnitude (Results.md's own
    traced failure reached a max state value of 5.7e144 before finally
    going non-finite) -- `density <= 0.0` is `False` for a huge positive
    density, so this was never repaired either, right up until it
    overflowed. Both now count as violations via `~finite_row` and
    `max_scale`, reusing the same neighbor-average/last-resort mechanism
    already built for negativity rather than adding a new repair path.
    """
    finite_row = jnp.all(jnp.isfinite(u), axis=-1)
    safe_u = jnp.where(finite_row[:, None], u, jnp.array([-1.0, 0.0, 1.0], dtype=u.dtype))

    density = safe_u[:, 0]
    safe_density = jnp.where(density > 0.0, density, 1.0)
    velocity = safe_u[:, 1] / safe_density
    pressure = (GAMMA - 1) * (safe_u[:, 2] - 0.5 * density * velocity ** 2)
    out_of_scale = (jnp.abs(density) > max_scale) | (jnp.abs(pressure) > max_scale)
    violates = (~finite_row) | (density <= 0.0) | (pressure <= 0.0) | out_of_scale

    neighbor_average = 0.5 * (jnp.roll(safe_u, 1, axis=0) + jnp.roll(safe_u, -1, axis=0))
    left_violates = jnp.roll(violates, 1)
    right_violates = jnp.roll(violates, -1)
    isolated = violates & (~left_violates) & (~right_violates)

    repaired = jnp.where(isolated[:, None], neighbor_average, safe_u)

    rho_r = repaired[:, 0]
    rho_denominator = jnp.where(rho_r > 0.0, rho_r, 1.0)
    p_r = (GAMMA - 1) * (
        repaired[:, 2] - 0.5 * repaired[:, 1] ** 2 / rho_denominator
    )
    needs_fallback = (
        (rho_r <= 0.0) | (p_r <= 0.0) | (~jnp.all(jnp.isfinite(repaired), axis=-1))
        | (jnp.abs(rho_r) > max_scale) | (jnp.abs(p_r) > max_scale)
    )

    # Last-resort floor/cap for anything still invalid (a genuine cluster, or
    # an isolated repair whose neighbor-average was itself invalid) -- see
    # docstring: this is a numerical-finiteness safety net, not claimed to
    # be a correct repair. Now clips density/pressure from ABOVE too (not
    # just floors them from below), for failure mode (2) above.
    safe_density_r = jnp.clip(repaired[:, 0], eps, max_scale)
    velocity_r = jnp.clip(repaired[:, 1] / safe_density_r, -max_velocity, max_velocity)
    pressure_r = jnp.clip(
        (GAMMA - 1) * (repaired[:, 2] - 0.5 * safe_density_r * velocity_r ** 2), eps, max_scale
    )
    momentum_r = safe_density_r * velocity_r
    energy_r = pressure_r / (GAMMA - 1) + 0.5 * safe_density_r * velocity_r ** 2
    fallback = jnp.stack([safe_density_r, momentum_r, energy_r], axis=-1)
    result = jnp.where(needs_fallback[:, None], fallback, repaired)
    if return_diagnostics:
        return result, dict(violation_mask=violates, isolated_mask=isolated,
                            fallback_mask=needs_fallback,
                            changed_mask=jnp.any(result != u, axis=-1))
    return result
