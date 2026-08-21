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
