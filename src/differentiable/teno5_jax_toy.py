# -*- coding: utf-8 -*-
"""Tiny JAX MLP stand-in for the NN-TENO5 classifier, for Phase 0/1 only.

`src/networks/TenoNetworks.py` builds the real NN-TENO5 classifier
(5 -> 64 -> 32 -> 16 -> 8 -> 3, sigmoid) with standalone Keras 2.2.4, which
has no autodiff-through-a-custom-solver story. This module is not a
replacement for that network -- it is a small hand-rolled MLP with the same
input/output contract (normalized 5-stencil in, 3 troubled-probabilities
out), used only to answer Phase 0's question: "can a gradient reach network
weights through the reconstruction + SSPRK3 rollout chain?" Once that is
answered, Phase 4/5 build the real architecture in JAX (or port trained
Keras weights into it) -- this toy network is deliberately not that.

Lives under `src/differentiable/` rather than `src/networks/` for the same
import-isolation reason as `teno5_jax.py` -- see
`src/differentiable/__init__.py`.
"""

import jax
import jax.numpy as jnp


def init_toy_params(key, hidden_sizes=(16, 8)):
    """Random weights for a stencil_size(5) -> ... -> CANDIDATE_COUNT(3) MLP."""
    sizes = [5] + list(hidden_sizes) + [3]
    layer_keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for layer_key, n_in, n_out in zip(layer_keys, sizes[:-1], sizes[1:]):
        w_key, _ = jax.random.split(layer_key)
        # He initialization: matches the ReLU hidden layers below.
        w = jax.random.normal(w_key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append((w, b))
    return params


def toy_keep_weight(params, stencil):
    """Normalized (nx, 5) stencil -> (nx, 3) keep-weight in [0, 1].

    Mirrors `normalize_teno_stencils` (teno_common.py:4) for the input side
    and NN-TENO5's sigmoid troubled-probability head (TenoNetworks.py) for
    the output side, inverted to a "keep" convention so it plugs directly
    into `combine_teno_candidates_jax`'s `keep_weight` argument.
    """
    minimum = jnp.min(stencil, axis=-1, keepdims=True)
    value_range = jnp.max(stencil, axis=-1, keepdims=True) - minimum
    # NOT `jnp.where(value_range > 0, (stencil - minimum) / value_range, 0.0)`.
    # jax.grad evaluates the gradient of *both* where-branches before
    # selecting one (there is no lazy short-circuit at the VJP level), so on
    # a constant stencil (value_range == 0, exactly the case on any flat
    # region of a piecewise-constant profile like the square wave below)
    # that untaken true-branch produces a 1/0 gradient, and 0 * inf = NaN --
    # which then poisons the entire backward pass through every layer, not
    # just this one. Guarding the denominator itself (not just the forward
    # value) keeps both the value AND the gradient finite: when
    # value_range == 0 the numerator (stencil - minimum) is also exactly 0,
    # so dividing by the substituted 1.0 already gives the correct 0 with no
    # separate select needed.
    safe_value_range = jnp.where(value_range > 0.0, value_range, 1.0)
    x = (stencil - minimum) / safe_value_range

    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)

    w, b = params[-1]
    troubled_probability = jax.nn.sigmoid(x @ w + b)
    return 1.0 - troubled_probability
