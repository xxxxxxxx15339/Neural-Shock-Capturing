# -*- coding: utf-8 -*-
"""Phase 4 projection layer: network troubled-probability -> keep_weight.

UNROLLED_PHYSICS_LOSS_GUIDE.md's Phase 4 asks for a layer "guaranteeing
reconstruction consistency." The guarantee that actually matters for
NN-TENO5 -- never combining candidates in a way that produces an undefined
or unstable result -- was already built as part of Phase 1, not deferred to
here: `combine_teno_candidates_jax` (teno5_jax.py) falls back to the
stencil's center value whenever every candidate's `keep_weight` is zero
(or, for a soft/continuous `keep_weight`, whenever their sum is), for any
`keep_weight` in [0, 1], hard or soft alike. That is the whole "projection"
the Unrolled Physics Loss proposal's Etape 4 is describing, applied uniformly rather than bolted on
separately -- see `teno5_soft_reconstruct_jax`'s docstring.

What this module adds is not new math, it's a single, explicit,
tested definition of "network output -> keep_weight" with the two modes a
trained model actually needs, so both training and inference go through the
same function instead of ad hoc `1 - sigmoid(...)` scattered per caller (as
`teno5_jax_toy.py` did for the Phase 0/1 spike):

- `mode='soft'`  (training): `keep_weight = 1 - troubled_probability`,
  continuous in [0, 1] -- gradients flow through every candidate's weight.
- `mode='hard'`  (inference): `keep_weight = 1{troubled_probability < threshold}`,
  matching `teno5_neural_keep_mask`'s NumPy convention
  (`probability >= threshold` means discard -- teno5.py:662) exactly.

`test_projection_jax.py` checks the exit criterion this module exists to
satisfy: a JAX reconstruction using `mode='hard'` at threshold 0.5, given
the same troubled-probabilities a real (fake, for the test) NN-TENO5 model
would return, is a bit-exact no-op against `teno5.py`'s own
`teno5_neural_reconstruct` -- not a self-consistency check against this
module's own math, but agreement with the actual existing inference code
path.
"""

import jax.numpy as jnp


def project_keep_weight(troubled_probability, mode='soft', threshold=0.5):
    """Convert a network's troubled-probability output to a keep_weight.

    Parameters
    ----------
    troubled_probability : array, any shape, values in [0, 1]
        E.g. an NN-TENO5-style sigmoid output (0 = smooth/keep candidate,
        1 = troubled/discard candidate).
    mode : 'soft' or 'hard'
        'soft': continuous keep_weight = 1 - troubled_probability, for
        training (differentiable everywhere).
        'hard': binary keep_weight via `threshold`, for inference -- matches
        teno5_neural_keep_mask's convention exactly.
    threshold : float, only used when mode='hard'.

    Returns
    -------
    array, same shape as `troubled_probability`, values in [0, 1] ('soft')
    or {0, 1} ('hard').
    """
    if mode == 'soft':
        return 1.0 - troubled_probability
    if mode == 'hard':
        return jnp.where(troubled_probability < threshold, 1.0, 0.0)
    raise ValueError("mode must be 'soft' or 'hard'; received {!r}.".format(mode))


def mixed_reconstruct(troubled_probability, stencils, network_reconstruct_fn, teno5_reconstruct_fn, mode='soft', lam=None):
    """Convex-mix the network's own reconstruction with TENO5's deterministic
    one, weighted by the network's own worst (max) per-candidate troubled
    probability -- so the mix provably reduces to TENO5's exact reconstruction
    wherever the network sees nothing troubled, instead of trusting an
    imitation-pretrained network to have learned that on its own.

    Found via literature research (Results.md Phase 16) after every prior
    Euler/scalar attempt (K-curriculum, smooth-data mixing, gradient
    clipping) either partially or entirely failed to close the specific gap
    this project kept measuring: the network losing to TENO5 more in SMOOTH
    regions than at discontinuities. Kossaczky & Ehrhardt-style differentiable
    TVD limiter (arXiv:2503.09625) diagnose the identical failure mode for
    flux limiters and fix it with the identical trick: parameterize the
    learned limiter as `(1-lambda)*classical + lambda*learned`, `lambda` a
    sigmoid network output -- both endpoints already have the properties
    (2nd-order accuracy, TVD) that matter, so *any* convex combination
    inherits them, rather than hoping an unconstrained network learns to
    reproduce them.

    This uses the network's EXISTING troubled-probability output as
    `lambda` (`lambda = max_candidate(troubled_probability)`) instead of
    adding a new network head for it: `troubled_probability` is already
    exactly "how confident is this candidate genuinely troubled" -- the
    right quantity to gate on -- and reusing it means zero new parameters,
    so this can be dropped into every EXISTING checkpoint (`run3_pretrained`,
    the Euler curriculum checkpoints) with no retraining needed to test it,
    unlike an architecture change that adds a fresh, randomly-initialized
    head. `troubled_probability`'s shape already matches the reconstruction
    output's shape along every axis except the trailing candidate axis
    (`teno5_soft_reconstruct_jax`/`teno5_reconstruct_jax` both reduce over
    that axis) -- for the scalar case, (n, 3) -> lambda (n,); for Euler's
    characteristic-stencil case, (n, 3, 3) -> lambda (n, 3) -- so `jnp.max`
    over the last axis always lines up with the reconstruction shape
    directly, no reshaping needed for either physics.

    `mode` is forwarded to `project_keep_weight` for `network_reconstruct_fn`
    the same as any other caller -- 'soft' for training (differentiable
    everywhere), 'hard' for inference.

    Parameters
    ----------
    troubled_probability : array (..., candidates)
        Raw network output, e.g. `1 - teno5_jax_toy.toy_keep_weight(...)`.
    stencils : array (..., candidates, stencil_size)
        Same stencils both `network_reconstruct_fn`/`teno5_reconstruct_fn`
        will be called on.
    network_reconstruct_fn : stencils -> array (...)
        E.g. `lambda s: teno5_soft_reconstruct_jax(s, project_keep_weight(troubled_probability, mode=mode))`.
    teno5_reconstruct_fn : stencils -> array (...)
        The deterministic scheme's own reconstruction, e.g. `teno5_reconstruct_jax`.
    lam : array (...), optional
        Precomputed mixing coefficient, same shape as the reconstruction
        output. If not given, defaults to `max_candidate(troubled_probability)`
        (Phase 16's original, zero-new-parameter choice). Pass
        `teno5_jax_toy.toy_lambda(lambda_params, troubled_probability)` for
        Phase 18's dedicated, separately-trained head instead.
    """
    if lam is None:
        lam = jnp.max(troubled_probability, axis=-1)
    return (1.0 - lam) * teno5_reconstruct_fn(stencils) + lam * network_reconstruct_fn(stencils)
