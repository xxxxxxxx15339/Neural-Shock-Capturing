
"""V2 deterministic gate. Optional diagnostics describe characteristic flux
corrections, not a proof of state-space oscillation control or accuracy."""
import math
from numbers import Real
import jax.numpy as jnp

from src.differentiable.teno5_jax import (
    DEFAULT_CUTOFF,
    teno5_selection_values,
    teno5_reconstruct_jax,
    teno5_soft_reconstruct_jax,
    LINEAR_WEIGHTS,
)
from src.differentiable.teno5_jax_toy import toy_keep_weight
from src.differentiable.projection_jax import project_keep_weight


def gated_nn_teno5_reconstruct(
    params,
    stencils,
    mode="soft",
    strength=1.0,
    cutoff=DEFAULT_CUTOFF,
):
    if isinstance(strength, Real) and (not math.isfinite(strength) or not 0.0 <= strength <= 1.0):
        raise ValueError("strength must be in [0, 1]")

    # 1. Classical TENO5 reconstruction
    teno_flux = teno5_reconstruct_jax(
        stencils, cutoff=cutoff
    )

    if isinstance(strength, Real) and strength == 0:
        return teno_flux

    # 2. Neural reconstruction
    soft_keep = toy_keep_weight(params, stencils)

    keep = project_keep_weight(
        1.0 - soft_keep,
        mode=mode,
    )

    nn_flux = teno5_soft_reconstruct_jax(
        stencils, keep
    )

    # 3. Detect TENO-troubled stencils
    chi = teno5_selection_values(stencils)

    troubled = jnp.any(
        chi < cutoff,
        axis=-1,
    )

    # 4. Apply correction only where troubled
    corrected_flux = (
        teno_flux
        + strength * (nn_flux - teno_flux)
    )

    return jnp.where(
        troubled & (strength != 0),
        corrected_flux,
        teno_flux,
    )


def gated_reconstruction_diagnostics(params, stencils, mode='soft', strength=1., cutoff=DEFAULT_CUTOFF):
    """Side-effect-free observations; callers choose when/where to sample.

    Leading axes retain interface, characteristic field and LF sign when present.
    Effective candidate weights include TENO optimal weights and normalization.
    A zero denominator means the documented center-value fallback.
    """
    base = teno5_reconstruct_jax(stencils, cutoff=cutoff)
    chi = teno5_selection_values(stencils)
    rejected = chi < cutoff
    gate = jnp.any(rejected, axis=-1)
    soft = project_keep_weight(1. - toy_keep_weight(params, stencils), 'soft')
    hard = project_keep_weight(1. - toy_keep_weight(params, stencils), 'hard')
    keep = soft if mode == 'soft' else hard
    if mode not in ('soft', 'hard'):
        raise ValueError('mode must be soft or hard')
    raw_weights = keep * LINEAR_WEIGHTS
    denom = jnp.sum(raw_weights, axis=-1, keepdims=True)
    weights = raw_weights / jnp.where(denom > 0, denom, 1.)
    neural = teno5_soft_reconstruct_jax(stencils, keep)
    correction = jnp.where(gate & (strength != 0), strength * (neural - base), 0.)
    return dict(gate=gate, rejected=rejected, selection_values=chi,
                soft_keep=soft, hard_keep=hard, normalized_weights=weights,
                fallback=(denom[..., 0] == 0), neural_minus_classical=neural-base,
                correction=correction, correction_sign=jnp.sign(correction),
                effective_strength=jnp.where(gate, strength, 0.),
                base=base, neural=neural)
