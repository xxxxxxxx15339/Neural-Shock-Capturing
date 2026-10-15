"""Matched inference ablations; all use the historical hard keep-mask policy."""
import math

import jax.numpy as jnp

from .model_io import CheckpointError
from .projection_jax import project_keep_weight
from .teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from .teno5_jax_toy import toy_keep_weight, toy_lambda, toy_lambda_v2

METHODS = ('teno5', 'neural-only', 'fixed-blend', 'learned-blend', 'max-blend')
INITIAL_FIXED_BLEND = 1.0 / (1.0 + math.exp(3.0))


def make_reconstruction(checkpoint, method, fixed_lambda=INITIAL_FIXED_BLEND):
    if method not in METHODS:
        raise ValueError(f'Unknown method {method!r}; expected one of {METHODS}')
    if not math.isfinite(fixed_lambda) or not 0 <= fixed_lambda <= 1:
        raise ValueError('fixed_lambda must be finite and in [0, 1]')
    if method == 'teno5':
        return teno5_reconstruct_jax
    if checkpoint is None:
        raise CheckpointError(f'{method} requires an explicit checkpoint')
    if method == 'learned-blend' and checkpoint.head is None:
        raise CheckpointError('learned-blend requires a saved lambda head; none is synthesized')
    backbone = tuple((jnp.asarray(w, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64))
                     for w, b in checkpoint.backbone)
    head = checkpoint.head
    if head is not None:
        if checkpoint.schema == 'linear':
            head = tuple(jnp.asarray(v, dtype=jnp.float64) for v in head)
        else:
            head = tuple(tuple(jnp.asarray(v, dtype=jnp.float64) for v in pair) for pair in head)

    def reconstruct(stencils):
        if stencils.shape[-1] != 5:
            raise ValueError(f'Expected stencil width 5, got {stencils.shape}')
        troubled = 1.0 - toy_keep_weight(backbone, stencils)
        neural = teno5_soft_reconstruct_jax(stencils, project_keep_weight(troubled, mode='hard'))
        if method == 'neural-only':
            return neural
        if method == 'fixed-blend':
            lam = fixed_lambda
        elif method == 'max-blend':
            lam = jnp.max(troubled, axis=-1)
        elif checkpoint.schema == 'linear':
            lam = toy_lambda(head, troubled)
        else:
            lam = toy_lambda_v2(head, troubled, stencils)
        return (1.0 - lam) * teno5_reconstruct_jax(stencils) + lam * neural

    return reconstruct
