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


CANDIDATE_COUNT = 3


def init_lambda_head_params(key, init_bias=-3.0, candidate_count=CANDIDATE_COUNT):
    """A small, SEPARATE head for `projection_jax.mixed_reconstruct`'s
    mixing coefficient (Results.md Phase 18), decoupled from the existing
    candidate keep-weight head.

    Phase 17 tried training through the mix while reusing the network's
    EXISTING troubled-probability output (`max` over candidates) as
    `lambda` directly, and got a consistent, three-times-repeated
    regression -- the working theory: that signal serves double duty
    (selecting which TENO5 candidates to keep AND, reused, deciding how
    much to trust the network's own reconstruction), so a gradient step
    that would improve the mix's loss also perturbs the already-good
    candidate-selection weights it shares parameters with, and vice versa.
    This head has its OWN weights, taking the existing troubled_probability
    vector as an input FEATURE (not as `lambda` itself) -- decoupled, so
    training it can no longer disturb the keep-weight head directly.

    `init_bias=-3.0` (not a generic near-zero init) is deliberate: a fresh,
    randomly-initialized head would start by mixing in a large but
    essentially random contribution from the network's own reconstruction
    at every interface, likely *worse* than either endpoint (this project
    already measured what an untrained network's contribution does to
    training stability -- Results.md Phase 6/10-12's whole Euler
    instability arc). `sigmoid(-3.0) ~= 0.047`: starts almost entirely
    deferring to TENO5 everywhere, a safe, conservative point gradient
    descent can only move away from where the loss actually justifies it,
    not one it has to first discover is safe.
    """
    w = jax.random.normal(key, (candidate_count, 1)) * 0.01
    b = jnp.array([init_bias])
    return (w, b)


def toy_lambda(lambda_params, troubled_probability):
    """(..., CANDIDATE_COUNT) troubled_probability -> (...,) mixing lambda
    in [0, 1], via the small dedicated head from `init_lambda_head_params`.
    """
    w, b = lambda_params
    logit = troubled_probability @ w + b
    return jax.nn.sigmoid(logit)[..., 0]


def init_lambda_head_params_v2(key, hidden_size=8, stencil_size=5, candidate_count=CANDIDATE_COUNT, init_bias=-3.0):
    """Results.md Phase 21: a richer mixing head than `init_lambda_head_params`,
    for the specific structural limit Phase 20 measured -- Sod's ratio got
    monotonically WORSE with any mixing coefficient tried, uniform or
    input-dependent, because `toy_lambda`'s only input (3
    troubled-probability numbers) doesn't carry enough local context to
    tell a Sod-like interface apart from a Shu-Osher-like one; both can
    show similar per-candidate troubled signals at a local discontinuity
    even though the two problems need opposite mixing behavior (Phase 20's
    sweep: any lambda > 0 already hurts Sod, while Shu-Osher wants lambda
    up to ~0.2).

    Feeds the head the RAW (normalized) stencil values on top of
    troubled_probability -- `stencil_size` (5) + `candidate_count` (3) = 8
    input features, the actual local solution shape instead of a single
    compressed per-candidate scalar -- through a small hidden layer
    (8 -> hidden_size -> 1) instead of `toy_lambda`'s single linear layer,
    since distinguishing two qualitatively different local shapes plausibly
    isn't linearly separable from 8 raw numbers. Note `stencils` (the
    reconstruction input) carries no separate candidate axis of its own --
    TENO5's 3 candidates are generated BY the network's output layer, not
    present in its input -- so this is 5 raw stencil values, not
    `candidate_count * stencil_size`.

    Same conservative-init philosophy as `init_lambda_head_params`: the
    OUTPUT layer's bias starts at `init_bias=-3.0` (`sigmoid(-3.0) ~= 0.047`,
    mostly defers to TENO5) and every weight is scaled tiny (`*0.01`), so
    training starts from the same safe point regardless of how much
    capacity the head has -- more input information should let it learn
    to lower lambda specifically on Sod-like inputs without needing to
    already be doing that at initialization.
    """
    input_size = stencil_size + candidate_count
    key1, key2 = jax.random.split(key)
    w1 = jax.random.normal(key1, (input_size, hidden_size)) * jnp.sqrt(2.0 / input_size) * 0.01
    b1 = jnp.zeros(hidden_size)
    w2 = jax.random.normal(key2, (hidden_size, 1)) * 0.01
    b2 = jnp.array([init_bias])
    return [(w1, b1), (w2, b2)]


def toy_lambda_v2(lambda_params, troubled_probability, stencils):
    """(..., CANDIDATE_COUNT) troubled_probability + (..., stencil_size)
    stencils -> (...,) mixing lambda in [0, 1], via the richer head from
    `init_lambda_head_params_v2`.

    `stencils` carries no candidate axis of its own (see
    `init_lambda_head_params_v2`'s docstring) -- its leading axes already
    match `troubled_probability`'s exactly (both are (n,) for the scalar
    case, (n, characteristic_fields) for Euler), so no reshape is needed,
    only a concatenation along the trailing feature axis.

    Stencils normalized the same way `toy_keep_weight` normalizes them
    (min-max) before being used as features -- consistent with how the
    rest of this module already treats raw stencil values, and keeps the
    feature scale bounded regardless of the problem's actual
    density/pressure magnitude.
    """
    minimum = jnp.min(stencils, axis=-1, keepdims=True)
    value_range = jnp.max(stencils, axis=-1, keepdims=True) - minimum
    safe_value_range = jnp.where(value_range > 0.0, value_range, 1.0)
    normalized_stencils = (stencils - minimum) / safe_value_range

    x = jnp.concatenate([troubled_probability, normalized_stencils], axis=-1)

    (w1, b1), (w2, b2) = lambda_params
    x = jax.nn.relu(x @ w1 + b1)
    logit = x @ w2 + b2
    return jax.nn.sigmoid(logit)[..., 0]
