# -*- coding: utf-8 -*-
"""JAX port of the deterministic TENO5 reconstruction math.

Lives under `src/differentiable/`, not `src/schemes/`, so that importing it
never pulls in `src/schemes/__init__.py` -> ... -> Keras. See
`src/differentiable/__init__.py` for why that isolation matters.

This module exists for one reason: `src/schemes/teno5.py` is pure NumPy, and
NumPy has no autodiff. Everything here is a line-for-line port of that file's
math (`teno5_candidate_values`, `teno5_smoothness_indicators`,
`teno5_selection_values`) so it can be differentiated with `jax.grad` and
compiled with `jax.jit`.

Two reconstruction entry points are provided, and the split matters:

- `teno5_reconstruct_jax` reproduces the *deterministic* scheme exactly,
  including the hard `chi >= cutoff` cutoff. It exists only to check this
  port against `teno5.py` — the cutoff is a step function, so gradients
  through it are zero almost everywhere. It is not what a trained NN-TENO5
  will use.
- `teno5_soft_reconstruct_jax` takes an externally supplied keep-weight in
  [0, 1] per candidate (e.g. a network's sigmoid output, or
  `1 - troubled_probability`) instead of computing one from the cutoff rule.
  This is the function that actually needs to be differentiable end-to-end,
  because it stands in for what Phase 4/5 will wire an NN-TENO5 model into.
"""

import jax.numpy as jnp

STENCIL_SIZE = 5
CANDIDATE_COUNT = 3
CENTER_INDEX = 2

LINEAR_WEIGHTS = jnp.array([1.0 / 10.0, 3.0 / 5.0, 3.0 / 10.0])

DEFAULT_EPSILON = 1.0e-40
DEFAULT_SCALE_CONSTANT = 1.0
DEFAULT_POWER = 6.0
DEFAULT_CUTOFF = 1.0e-5


def teno5_candidate_values(values):
    """Port of teno5.py:192 teno5_candidate_values. Same coefficients."""
    value0 = values[..., 0]
    value1 = values[..., 1]
    value2 = values[..., 2]
    value3 = values[..., 3]
    value4 = values[..., 4]

    candidate0 = (1.0 / 3.0) * value0 - (7.0 / 6.0) * value1 + (11.0 / 6.0) * value2
    candidate1 = -(1.0 / 6.0) * value1 + (5.0 / 6.0) * value2 + (1.0 / 3.0) * value3
    candidate2 = (1.0 / 3.0) * value2 + (5.0 / 6.0) * value3 - (1.0 / 6.0) * value4

    return jnp.stack([candidate0, candidate1, candidate2], axis=-1)


def teno5_smoothness_indicators(values):
    """Port of teno5.py:251 teno5_smoothness_indicators (Jiang-Shu betas)."""
    value0 = values[..., 0]
    value1 = values[..., 1]
    value2 = values[..., 2]
    value3 = values[..., 3]
    value4 = values[..., 4]

    beta0 = (13.0 / 12.0) * (value0 - 2.0 * value1 + value2) ** 2 \
        + 0.25 * (value0 - 4.0 * value1 + 3.0 * value2) ** 2
    beta1 = (13.0 / 12.0) * (value1 - 2.0 * value2 + value3) ** 2 \
        + 0.25 * (value1 - value3) ** 2
    beta2 = (13.0 / 12.0) * (value2 - 2.0 * value3 + value4) ** 2 \
        + 0.25 * (3.0 * value2 - 4.0 * value3 + value4) ** 2

    beta = jnp.stack([beta0, beta1, beta2], axis=-1)
    # NumPy version uses np.maximum(beta, 0.0) to guard roundoff. jnp.where
    # keeps this a normal elementwise op (no data-dependent shape, so it is
    # jit/grad-safe) rather than jnp.maximum's slightly different NaN
    # behavior at exactly 0 -- either is fine here, this matches the NumPy
    # semantics most closely.
    return jnp.where(beta > 0.0, beta, 0.0)


def teno5_selection_values(
    values,
    epsilon=DEFAULT_EPSILON,
    scale_constant=DEFAULT_SCALE_CONSTANT,
    power=DEFAULT_POWER,
):
    """Port of teno5.py:352 teno5_selection_values (chi_k), log-domain softmax."""
    beta = teno5_smoothness_indicators(values)
    tau5 = jnp.abs(beta[..., 0] - beta[..., 2])

    ratio = tau5[..., None] / (beta + epsilon)
    log_gamma = power * jnp.log(scale_constant + ratio)
    log_gamma = log_gamma - jnp.max(log_gamma, axis=-1, keepdims=True)
    gamma = jnp.exp(log_gamma)
    chi = gamma / jnp.sum(gamma, axis=-1, keepdims=True)
    return chi


def combine_teno_candidates_jax(candidates, keep_weight, linear_weights, fallback):
    """Port of teno_common.py:47 combine_teno_candidates.

    `keep_weight` may be a hard {0,1} mask (deterministic path) or a soft
    value in [0,1] (network path) -- the arithmetic is identical either way,
    which is exactly the point: the deterministic scheme is the keep_weight
    -> {0,1} special case of the same differentiable formula.
    """
    active_weights = keep_weight * linear_weights
    denominator = jnp.sum(active_weights, axis=-1)
    numerator = jnp.sum(active_weights * candidates, axis=-1)
    # NumPy uses np.divide(..., where=denominator > 0.0) so the fallback
    # value survives untouched when every candidate is rejected. jnp has no
    # in-place `where=`; jnp.where over the two full results is the
    # jit/grad-safe equivalent (both branches are evaluated, which is fine
    # here since neither branch can produce NaN/Inf on its own).
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    blended = numerator / safe_denominator
    return jnp.where(denominator > 0.0, blended, fallback)


def teno5_reconstruct_jax(
    values,
    epsilon=DEFAULT_EPSILON,
    scale_constant=DEFAULT_SCALE_CONSTANT,
    power=DEFAULT_POWER,
    cutoff=DEFAULT_CUTOFF,
):
    """Deterministic TENO5, hard cutoff. For regression testing against teno5.py only."""
    candidates = teno5_candidate_values(values)
    chi = teno5_selection_values(values, epsilon, scale_constant, power)
    keep_mask = jnp.where(chi >= cutoff, 1.0, 0.0)
    fallback = values[..., CENTER_INDEX]
    return combine_teno_candidates_jax(candidates, keep_mask, LINEAR_WEIGHTS, fallback)


def teno5_soft_reconstruct_jax(values, keep_weight):
    """Reconstruction driven by an externally supplied [0,1] keep_weight.

    This is the function later phases hook a network into: replace
    `keep_weight` with `1 - troubled_probability` from an NN-TENO5-style
    model and this becomes the differentiable NN-TENO5 forward pass.
    """
    candidates = teno5_candidate_values(values)
    fallback = values[..., CENTER_INDEX]
    return combine_teno_candidates_jax(candidates, keep_weight, LINEAR_WEIGHTS, fallback)
