import numpy as np


def normalize_teno_stencils(stencils):
    """Map each nonconstant stencil to [0, 1] along its final axis."""
    stencils = np.asarray(
        stencils,
        dtype=float,
    )

    if stencils.ndim < 1:
        raise ValueError(
            'TENO normalization requires a final stencil axis.'
        )

    if not np.isfinite(stencils).all():
        raise FloatingPointError(
            'TENO normalization received NaN or infinity.'
        )

    minimum = np.min(
        stencils,
        axis=-1,
        keepdims=True,
    )
    value_range = (
        np.max(
            stencils,
            axis=-1,
            keepdims=True,
        )
        - minimum
    )
    normalized = np.zeros_like(
        stencils,
        dtype=float,
    )
    np.divide(
        stencils - minimum,
        value_range,
        out=normalized,
        where=value_range > 0.0,
    )
    return normalized


def combine_teno_candidates(candidates, keep_mask, linear_weights, fallback):
    """Combine retained candidates and use fallback if all are rejected."""
    candidates = np.asarray(candidates, dtype=float)
    keep_mask = np.asarray(keep_mask, dtype=float)
    linear_weights = np.asarray(linear_weights, dtype=float)

    active_weights = keep_mask*linear_weights
    denominator = np.sum(active_weights, axis=-1)
    numerator = np.sum(active_weights*candidates, axis=-1)
    result = np.array(fallback, dtype=float, copy=True)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


# Compatibility with the name used by the first deterministic implementation.
teno_common = combine_teno_candidates
