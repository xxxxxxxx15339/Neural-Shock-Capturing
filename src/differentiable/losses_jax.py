# -*- coding: utf-8 -*-
"""Phase 5: proj.pdf's Unrolled Physics Loss (Equations 5-8).

    L_total = lambda1*L_MSE + lambda2*L_TV + lambda3*L_BVD + lambda4*L_Reg

Each term operates on the Phase 1 rollout's output (a JAX-traced final
state after K SSPRK3 steps -- see `numerics_jax.rollout`), so all four are
differentiable w.r.t. whatever produced that state (network weights, via
`teno5_soft_reconstruct_jax` and `project_keep_weight`).

Two of proj.pdf's four terms have a direct formula (Eq. 6, 8); the other
two need a documented reading, same policy as feature_vector.py's caveats:

- L_TV (Eq. 7) has a formula but an implicit assumption worth stating: the
  `max(0, ...)` hinge only penalizes an *increase* in total variation
  relative to the state the rollout started from. That makes it a soft TVD
  constraint (discourage new oscillations), not a hard one (never
  guarantees TV cannot rise) -- say so if this gets reported as a result,
  since "TV <= const" (proj.pdf's own Table 1 objective for the advection
  case) is a stronger claim than this loss term can make on its own.

- L_BVD (Eq. 5) was named but never given a formula in proj.pdf itself.
  Resolved directly by the professor, who pointed to the source: Sun,
  Inaba & Xiao (2016), "Boundary Variation Diminishing (BVD)
  reconstruction." Its boundary variation is BV(Phi)_{i+1/2} =
  |Phi_i(x_{i+1/2}) - Phi_{i+1}(x_{i+1/2})| -- each cell's OWN candidate
  reconstruction compared against its NEIGHBOR's own reconstruction at
  their shared interface, purely self-referential (no exact solution
  needed). `L_BVD` here penalizes the network's own BV only when it
  *exceeds* the deterministic TENO5 scheme's BV at the same interface --
  "don't introduce spurious reconstruction disagreement beyond what the
  reference scheme already has." See `numerics_jax.boundary_variation_jax`
  / `euler_jax.euler_boundary_variation_jax` for the actual computation;
  an earlier version of this file approximated BV by reconstructing one
  stencil forwards and reversed in place, which is a *different* (and
  wrong) quantity -- it compares a stencil to its own mirror image, not to
  its neighbor's own reconstruction. Corrected once the real definition
  was available; see PROJ_IMPLEMENTATION_GUIDE.md Phase 5's status note.

L_Reg (Eq. 8, `||c_NN - c_base||^2`) is implemented as a generic pytree L2
distance via `jax.tree_util`, not hardcoded to the toy network's specific
list-of-(weight, bias)-tuples parameter shape -- so it keeps working
unchanged if Phase 6 swaps in a differently-structured network. This is
exactly the anchor-weight penalty already built and validated in
`Train_NNTENO5.py` (`--warm-start-path`/`--anchor-weight`, Results.md
Phase 9), reframed as one term in a combined loss instead of the whole
objective.
"""

import jax
import jax.numpy as jnp


def l_mse(final_state, exact_final_state):
    """Equation 6."""
    return jnp.mean((final_state - exact_final_state) ** 2)


def total_variation(v, axis=0):
    """TV(v) = sum_i |v_{i+1} - v_i}|, Equation 7's TV(.).

    For a multi-component state (e.g. Euler's (nx, 3)), sums the diff
    magnitude across all components as well as all cells -- i.e. this is
    the combined TV of every conserved variable together, not a per-field
    breakdown. A per-field TV vector is one `axis`-only reduction away if
    that turns out to be what's wanted instead.
    """
    return jnp.sum(jnp.abs(jnp.diff(v, axis=axis)))


def l_tv(final_state, initial_state, axis=0, normalize=False):
    """Equation 7: max(0, TV(u^{n+K}) - TV(u^n)).

    `normalize=False` matches Eq. 7 literally (a raw sum, growing with grid
    size). `normalize=True` divides by the number of interior differences
    instead (mean absolute jump-of-jumps), which is what Phase 6's first
    training run (see PROJ_IMPLEMENTATION_GUIDE.md Phase 6 status note)
    found was actually necessary: at CELLS=64, raw TV sits around 0.6-0.9
    while L_MSE sits around 0.01-0.02 -- a ~40-50x scale mismatch that, even
    after applying lambda_tv=0.1, still let L_TV dominate the gradient over
    L_MSE by roughly 4x, training the network to minimize total variation
    over getting the answer right. This is a real gap in proj.pdf's own
    Equation 7: it specifies TV's formula but says nothing about the scale
    mismatch between an unnormalized sum and the other (already
    small-and-bounded) loss terms, which any implementation will hit the
    moment lambda2 is chosen without accounting for it.
    """
    total_variation_change = jnp.maximum(
        total_variation(final_state, axis=axis) - total_variation(initial_state, axis=axis),
        0.0,
    )
    if normalize:
        cell_count = final_state.shape[axis]
        return total_variation_change / max(cell_count - 1, 1)
    return total_variation_change


def l_bvd(network_boundary_variation, deterministic_boundary_variation):
    """Equation 5's L_BVD, per the professor's clarification: penalize the
    network's own BVD boundary-variation jump only when it *exceeds* the
    deterministic scheme's, at the same interfaces.

    Both arguments are precomputed boundary-variation arrays -- see
    `numerics_jax.boundary_variation_jax` (scalar) or
    `euler_jax.euler_boundary_variation_jax` (Euler) -- rather than raw
    stencils/reconstruct callables, because the true BVD jump
    (Sun, Inaba & Xiao 2016, Eq. 7: |Phi_i(x_{i+1/2}) - Phi_{i+1}(x_{i+1/2})|)
    needs each cell's own neighbor's own reconstruction, which needs the
    full state array (to mirror-and-shift, not mirror one stencil in
    place) and, for Euler, the Roe eigenbasis and characteristic
    projection -- machinery that differs between the scalar and Euler
    cases and that this module has no business knowing about. Computing
    it is the caller's job; comparing it against the deterministic
    scheme's own jump is this function's.

    An earlier version of this function computed the jump by reconstructing
    a single stencil forwards and reversed in place -- not the true BVD
    definition (compares a stencil to its own mirror image, not to its
    neighbor's own reconstruction). Corrected after the professor pointed
    to Sun, Inaba & Xiao (2016) directly.
    """
    return jnp.mean(jnp.maximum(network_boundary_variation - deterministic_boundary_variation, 0.0))


def l_reg(params, anchor_params, normalize=False):
    """Equation 8: ||c_NN - c_base||^2, as a generic pytree L2 distance.

    `normalize=True` divides by the total parameter count (mean squared
    difference per weight, instead of a raw sum over every weight in the
    network) -- same scale-mismatch reasoning as `l_tv`'s `normalize`: a raw
    sum grows with network size, not with how "wrong" the drift actually is.
    """
    squared_diffs = jax.tree_util.tree_map(
        lambda p, a: jnp.sum((p - a) ** 2), params, anchor_params
    )
    total = jax.tree_util.tree_reduce(lambda x, y: x + y, squared_diffs)
    if normalize:
        sizes = jax.tree_util.tree_map(lambda p: p.size, params)
        count = jax.tree_util.tree_reduce(lambda x, y: x + y, sizes)
        return total / count
    return total


def unrolled_physics_loss(
    final_state,
    exact_final_state,
    initial_state,
    network_boundary_variation,
    deterministic_boundary_variation,
    params,
    anchor_params,
    lambda_mse=1.0,
    lambda_tv=0.0,
    lambda_bvd=0.0,
    lambda_reg=0.0,
    lambda_positivity=0.0,
    positivity_penalty=0.0,
    tv_axis=0,
    tv_normalize=False,
    reg_normalize=False,
):
    """Equation 5: the full weighted combination, plus the per-term
    breakdown (Phase 6's guide entry asks these be logged separately, not
    just the weighted sum, to diagnose which term dominates during a
    lambda sweep).

    `tv_normalize`/`reg_normalize` default to False (literal Eq. 7/8, raw
    sums) for backward compatibility with Phase 5's own tests; Phase 6
    training should set both True -- see `l_tv`/`l_reg`'s docstrings for
    why the un-normalized sums caused a real training failure.

    `network_boundary_variation`/`deterministic_boundary_variation`: see
    `l_bvd`'s docstring -- precomputed by the caller (scalar vs. Euler
    stencil-gathering differs), not recomputed here.

    `positivity_penalty`/`lambda_positivity`: not one of proj.pdf's own
    four terms -- added per the professor's clarification that an
    *isolated* positivity violation should be repaired (see
    `euler_jax.euler_positivity_repair`), while several *simultaneous*
    violations signal the scheme itself is unstable, which the loss should
    be told about rather than have silently hidden by the repair. Default
    0.0/0.0 keeps this inert for the scalar case, which has no positivity
    concept. `positivity_penalty` is itself precomputed by the caller
    (e.g. `euler_jax.positivity_penalty_jax`) for the same reason as the
    BVD arrays: it needs Euler-specific primitive-variable extraction this
    module shouldn't have to know about.
    """
    mse_term = l_mse(final_state, exact_final_state)
    tv_term = l_tv(final_state, initial_state, axis=tv_axis, normalize=tv_normalize)
    bvd_term = l_bvd(network_boundary_variation, deterministic_boundary_variation)
    reg_term = l_reg(params, anchor_params, normalize=reg_normalize)

    total = (
        lambda_mse * mse_term
        + lambda_tv * tv_term
        + lambda_bvd * bvd_term
        + lambda_reg * reg_term
        + lambda_positivity * positivity_penalty
    )

    return total, {
        'l_mse': mse_term,
        'l_tv': tv_term,
        'l_bvd': bvd_term,
        'l_reg': reg_term,
        'l_positivity': positivity_penalty,
        'l_total': total,
    }
