
import jax.numpy as jnp

from src.differentiable.teno5_jax import (
    DEFAULT_CUTOFF,
    teno5_selection_values,
    teno5_reconstruct_jax,
    teno5_soft_reconstruct_jax,
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
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    # 1. Classical TENO5 reconstruction
    teno_flux = teno5_reconstruct_jax(
        stencils, cutoff=cutoff
    )

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
        troubled,
        corrected_flux,
        teno_flux,
    )
