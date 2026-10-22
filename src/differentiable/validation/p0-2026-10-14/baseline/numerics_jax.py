# -*- coding: utf-8 -*-
"""JAX port of the scalar finite-volume solver (periodic boundary only).

Lives under `src/differentiable/` rather than `src/core/` for the same
import-isolation reason as `teno5_jax.py` -- see
`src/differentiable/__init__.py`.

Ports, one function each:

- `periodic_stencils`     <- FiniteVolumeMethod.partU     (SimulationClasses.py:188)
- `lax_friedrichs_flux`   <- FluxSplittingMethod.flux      (SimulationClasses.py:161)
                             + LaxFriedrichs                (FluxSplittingMethods.py:13)
- `ssprk3_step`           <- TimeSteppingMethod.stepIt      (TimeSteppingMethods.py:29)
                             + SSPRK3's fixed (ss, cff)      (TimeSteppingMethods.py:29)

`ssprk3_step` is written as three unrolled lines instead of porting the
generic nss x nss loop from `TimeSteppingMethod.stepIt`. SSPRK3 always has
exactly 3 substeps with a fixed (ss, cff) -- the generic double loop exists
in the NumPy version to support *other* time-steppers (ExplicitEuler,
SSPRK2) through the same class. Since this port only needs SSPRK3, writing
out its Shu-Osher form directly is fewer moving parts than reproducing a
generic loop for one fixed instantiation of it. If a differentiable
ExplicitEuler/SSPRK2 is needed later, port the generic loop instead.
"""

import jax
import jax.numpy as jnp


def periodic_stencils(u, stencil_size=5):
    """Port of FiniteVolumeMethod.partU, periodic boundary only.

    NumPy version loops `u_all[:, i] = np.roll(u, floor(ss/2) - i)`. Same
    shifts here, stacked instead of column-assigned (JAX arrays are
    immutable).
    """
    radius = stencil_size // 2
    columns = [jnp.roll(u, radius - i) for i in range(stencil_size)]
    return jnp.stack(columns, axis=-1)


def two_sided_reconstruction(u, reconstruct):
    """Cell i's own right-face value and cell (i+1)'s own left-face value,
    both at interface i+1/2, each from its own stencil (periodic).

    This is the forward/mirror trick FluxSplittingMethod.flux
    (SimulationClasses.py:161) uses to build a two-sided flux from this
    codebase's one-sided reconstruction convention -- factored out here
    because it is also exactly the pair `boundary_variation_jax` needs:
    per the professor's clarification (and Sun, Inaba & Xiao 2016, Eq. 7),
    BVD compares each cell's OWN candidate reconstruction against its
    neighbor's, not a single stencil against its own mirror image. Mirroring
    the *whole array* then shifting (not mirroring one stencil in place) is
    what makes `mirrored[i]` genuinely cell (i+1)'s own left-face value
    rather than a symmetrized version of cell i's.
    """

    def eval_f(v):
        return reconstruct(periodic_stencils(v))

    forward = eval_f(u)
    mirrored = jnp.roll(jnp.flip(eval_f(jnp.flip(u))), -1)
    return forward, mirrored


def boundary_variation_jax(u, reconstruct):
    """BV(Phi)_{i+1/2} = |Phi_i(x_{i+1/2}) - Phi_{i+1}(x_{i+1/2})|, the true
    BVD boundary-variation jump (Sun, Inaba & Xiao 2016, Eq. 7) at every
    interface -- see `two_sided_reconstruction`'s docstring.
    """
    forward, mirrored = two_sided_reconstruction(u, reconstruct)
    return jnp.abs(forward - mirrored)


def lax_friedrichs_flux(u, physical_flux, alpha, reconstruct):
    """Port of FluxSplittingMethod.flux + LaxFriedrichs, periodic only.

    `reconstruct` plays the role of `FVM.L`: it maps a (nx, 5) stencil array
    to (nx,) reconstructed interface values. Passing it in as an argument
    (rather than baking in a specific scheme) is what lets Phase 4/5 swap in
    a network-driven reconstruction without touching this function.
    """

    u_int_forward, u_int_mirrored = two_sided_reconstruction(u, reconstruct)

    f_plus = 0.5 * (physical_flux(u_int_forward) + alpha * u_int_forward)
    f_minus = 0.5 * (physical_flux(u_int_mirrored) - alpha * u_int_mirrored)

    return (f_plus + f_minus) - jnp.roll(f_plus + f_minus, 1)


def ssprk3_step(u, flux_fn, dt, dx, state_guard=None):
    """Shu-Osher SSPRK3, matching TimeSteppingMethods.SSPRK3's (ss, cff).

    `state_guard`, if given, is applied after *each* of the three internal
    substages (u1, u2, u3) -- not just once at the end. This matters: u1
    and u2 are themselves fed into `flux_fn` again before the step
    finishes (`flux_fn(u1)`, `flux_fn(u2)`), so a guard applied only to the
    final return value cannot stop an invalid *intermediate* substage from
    reaching the flux function first. This was a real bug, not a
    hypothetical one: `numerics_jax.rollout`'s original `state_guard`
    parameter applied the guard only between full `ssprk3_step` calls, and
    Euler training still went to NaN at step 0 with it in place, because
    the NaN was coming from an unguarded u1/u2 inside a single step -- see
    UNROLLED_PHYSICS_LOSS_GUIDE.md's Phase 6/7 status note for the full
    diagnosis trail.
    """
    def guard(v):
        return v if state_guard is None else state_guard(v)

    u0 = u
    u1 = guard(u0 - flux_fn(u0) * dt / dx)
    u2 = guard(0.75 * u0 + 0.25 * u1 - 0.25 * flux_fn(u1) * dt / dx)
    u3 = guard((1.0 / 3.0) * u0 + (2.0 / 3.0) * u2 - (2.0 / 3.0) * flux_fn(u2) * dt / dx)
    return u3


def rollout_unrolled(u0, flux_fn, dt, dx, steps):
    """K-step SSPRK3 rollout, plain Python loop.

    Kept for reference/comparison and for cases needing per-step access
    (not compatible with `jax.lax.scan`'s fixed carry-in/carry-out
    signature). Measured cost on the Euler flux (Phase 1's Sod spike,
    K=240 substeps): `jax.jit` compiled this in ~11 minutes on CPU, because
    jit traces a plain Python loop by unrolling it -- the compiled graph
    contains 240 full copies of the (expensive: batched 3x3 matrix inverse,
    two einsums, several stencil gathers) Euler flux function, and XLA's
    optimizer has to process all of them as one flat graph. Once compiled,
    each call was fast (0.14s) -- but an 11-minute compile is a real cost
    for any workflow that changes K, changes shapes, or restarts often.
    `rollout` below (lax.scan) was built and measured specifically to fix
    this; see its docstring for the result.
    """
    u = u0
    for _ in range(steps):
        u = ssprk3_step(u, flux_fn, dt, dx)
    return u


def rollout(u0, flux_fn, dt, dx, steps, state_guard=None):
    """K-step SSPRK3 rollout via jax.lax.scan.

    `jax.lax.scan` compiles the loop body once and has XLA emit an actual
    loop construct, instead of unrolling `steps` copies into one graph --
    the fix for `rollout_unrolled`'s compile-time blowup on the Euler flux.
    Measured on the same Sod problem/K that took `rollout_unrolled` ~11
    minutes (647s) to jit-compile: `rollout` compiled in 4.4s -- about 150x
    faster -- with identical results (same loss values, same regression
    match against the NumPy solver) and the same per-call cost after
    compiling (~0.15s). This is
    now the default `rollout` -- Phase 5/6 should use this one; reach for
    `rollout_unrolled` only if something needs to break out of scan's rigid
    carry-in/carry-out shape (e.g. returning every intermediate step, which
    `lax.scan` can also do via its `ys` output, but wasn't needed here).

    `state_guard`, if given, is forwarded to `ssprk3_step` and applied
    after each of its three *internal* substages, not just once per full
    step -- see `ssprk3_step`'s docstring for why applying it only between
    full steps is not enough. Identity (None) by default -- the scalar
    advection case never needed one. Euler training did: an
    untrained-for-Euler network's soft reconstruction is not guaranteed to
    keep density/pressure positive substep-to-substep the way the
    deterministic scheme is, and once either goes negative,
    `euler_jax.py`'s `sqrt`/divisions turn it into NaN immediately and
    irrecoverably (`eulerSimulation._validate_state`, SimulationClasses.py:65,
    raises on exactly this in the NumPy path -- `jax.lax.scan` can't raise
    mid-trace, so this is the differentiable equivalent
    UNROLLED_PHYSICS_LOSS_GUIDE.md's Phase 1 anticipated). Pass
    `euler_jax.euler_positivity_guard` for Euler training.
    """

    def body(u, _):
        return ssprk3_step(u, flux_fn, dt, dx, state_guard=state_guard), None

    u_final, _ = jax.lax.scan(body, u0, xs=None, length=steps)
    return u_final


def ssprk3_step_with_penalty(u, flux_fn, dt, dx, state_guard, penalty_fn):
    """Like `ssprk3_step`, but also returns `penalty_fn` summed over each of
    the three internal substages, evaluated BEFORE `state_guard` repairs them.

    Exists to fix a real bug found by inspection (Results.md Phase 12's
    "honest limitation"): `euler_jax.positivity_penalty_jax`, wired into
    `losses_jax.unrolled_physics_loss` as `lambda_positivity`, was being
    computed on a rollout's *final*, already-`state_guard`-repaired state --
    by construction always 0, since every violation was already cleaned up
    before the penalty ever looked. This function evaluates `penalty_fn` on
    each pre-repair substage instead (the same substages `ssprk3_step`
    already guards individually, for the same "u1/u2 feed into flux_fn again
    before the step finishes" reason -- see its docstring), so a violation
    the repair fixes is no longer invisible to the loss.

    `state_guard` and `penalty_fn` are both required (not optional like
    `ssprk3_step`'s `state_guard`) -- this function only exists to serve
    that combination; call `ssprk3_step` directly when a plain rollout
    (no penalty tracking) is what's needed.
    """
    def guarded(v):
        return state_guard(v), penalty_fn(v)

    u0 = u
    v1 = u0 - flux_fn(u0) * dt / dx
    u1, p1 = guarded(v1)
    v2 = 0.75 * u0 + 0.25 * u1 - 0.25 * flux_fn(u1) * dt / dx
    u2, p2 = guarded(v2)
    v3 = (1.0 / 3.0) * u0 + (2.0 / 3.0) * u2 - (2.0 / 3.0) * flux_fn(u2) * dt / dx
    u3, p3 = guarded(v3)
    return u3, p1 + p2 + p3


def rollout_with_positivity_penalty(u0, flux_fn, dt, dx, steps, state_guard, penalty_fn):
    """`rollout`'s `jax.lax.scan` structure, but also accumulates
    `penalty_fn` (pre-repair, see `ssprk3_step_with_penalty`) across every
    substage of every step, and returns `(u_final, total_penalty)` instead
    of just `u_final`. See `ssprk3_step_with_penalty`'s docstring for why
    this is a separate function rather than an option on `rollout` itself:
    `rollout`'s single-return-value signature is relied on unchanged by
    every other caller (`DiffSpike_*`, `ValidationMatrix_ProjPDF.py`,
    the scalar training script, evaluation code in this same module) --
    this one is additive, for `Train_NNTENO5euler_UnrolledJAX.py`'s
    training loss only.
    """

    def body(carry, _):
        u, total_penalty = carry
        u_next, step_penalty = ssprk3_step_with_penalty(u, flux_fn, dt, dx, state_guard, penalty_fn)
        return (u_next, total_penalty + step_penalty), None

    (u_final, total_penalty), _ = jax.lax.scan(
        body, (u0, jnp.asarray(0.0, dtype=u0.dtype)), xs=None, length=steps
    )
    return u_final, total_penalty
