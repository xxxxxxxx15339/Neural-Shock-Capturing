# -*- coding: utf-8 -*-
"""proj.pdf Etape 1 feature vector (Equations 1-3), built on the existing
TENO5 building blocks in src/schemes/teno5.py and src/schemes/teno_common.py.

X_j = [u_s, c~_WENO/TENO, eta_0, eta_1, eta_2, J_{j-1/2}, J_{j+1/2}, M_j, |dP|/P]

Plain NumPy, not JAX: this runs once, offline, over a stored dataset to
produce network *inputs* -- nothing here needs to be differentiable, only
the network that consumes its output does (that network lives in
src/differentiable/ once Phase 4/5 wire it up). Keeping this Keras-free and
JAX-free means it can be imported from any of this repo's four environments
(base, weno-nn, zero-to-hero, teno-jax) with no install friction, matching
the same reasoning as src/differentiable/__init__.py's package split.

One thing in proj.pdf's own spec was underspecified and has since been
resolved by the professor directly; one remains an approximation, now with
a precise (rather than vague) description of what it approximates:

1. "c~_WENO/TENO" (Equation 3) never got a formula in proj.pdf, only a
   name. Confirmed directly by the professor: it is c_k/d_k, the scheme's
   own optimal linear reconstruction weights -- exactly what
   `scheme_linear_weights` already returns here (`LINEAR_WEIGHTS`,
   `[1/10, 3/5, 3/10]` for TENO5, broadcast over every sample). No change
   needed; this was a guess when written, it no longer is one.

2. J_{j-1/2} and J_{j+1/2} (Equation 3): the professor pointed to the
   actual source, Sun, Inaba & Xiao (2016), "Boundary Variation
   Diminishing (BVD) reconstruction." Its real definition is
   BV(Phi)_{i+1/2} = |Phi_i(x_{i+1/2}) - Phi_{i+1}(x_{i+1/2})| -- cell i's
   OWN right-face reconstruction compared against cell (i+1)'s OWN
   left-face reconstruction, each from its OWN stencil. The correct,
   full-grid-aware version of this is implemented in
   `src/differentiable/numerics_jax.py` (`boundary_variation_jax`) and
   `src/differentiable/euler_jax.py` (`euler_boundary_variation_jax`), and
   is what Phase 5's `L_BVD` loss term actually uses.

   The `_interface_jump` function below is NOT that -- it reconstructs one
   stencil forwards and reversed in place, which compares a stencil to its
   own mirror image, not to a genuine neighboring cell's own reconstruction.
   That's a real approximation, not the definition, kept here only because
   this module works over *isolated* stencil samples (e.g. one row of
   `data/teno5ExactStencils.csv`) with no grid continuity -- there is no
   actual neighboring cell's data available to reconstruct from. If a
   dataset generator ever produces full-grid context instead of isolated
   stencils, use `boundary_variation_jax`/`euler_boundary_variation_jax`
   directly rather than this approximation. Widening the window from 5 to
   6 points (`WIDE_WINDOW` below) does not fix this gap -- it only makes
   J_{j-1/2} available as the *same kind* of same-stencil-mirror
   approximation, not a genuine neighbor-based BVD jump.
"""

import numpy as np

from ..schemes.teno5 import (
    DEFAULT_EPSILON,
    LINEAR_WEIGHTS,
    STENCIL_SIZE,
    teno5_global_smoothness,
    teno5_reconstruct,
    teno5_smoothness_indicators,
)
from ..schemes.teno_common import normalize_teno_stencils

WIDE_WINDOW = STENCIL_SIZE + 1  # [j-3 .. j+2]


def _interface_jump(five_point_stencil, **teno_kwargs):
    """|left-biased reconstruction - mirrored right-biased reconstruction|
    at this window's right face.

    Reuses the exact forward/mirror trick FluxSplittingMethod.flux
    (SimulationClasses.py:161) already relies on to build a two-sided flux
    from teno5_reconstruct's one-sided convention: reconstructing on the
    stencil as given yields the state approaching this face from the left;
    reconstructing on the reversed stencil yields the state a mirrored
    scheme would assign approaching from the right. Their difference is
    exactly the BVD "reconstruction jump" proj.pdf's J_{j+1/2} names.
    """
    left_state = teno5_reconstruct(five_point_stencil, **teno_kwargs)
    right_state = teno5_reconstruct(five_point_stencil[..., ::-1], **teno_kwargs)
    return np.abs(left_state - right_state)


def scalar_feature_vector(window, epsilon=DEFAULT_EPSILON):
    """proj.pdf Equation 3's X_j, scalar-PDE terms only (no Mach/pressure).

    Parameters
    ----------
    window : ndarray, shape (..., 5) or (..., 6)
        5-wide: the standard NN-TENO5 stencil [u_{j-2}, ..., u_{j+2}].
        6-wide: [u_{j-3}, ..., u_{j+2}] -- one point wider, enabling both
        interface jumps. See module docstring, point 2.

    Returns
    -------
    dict of ndarray. Deliberately a dict, not one concatenated array --
    callers that only want a subset (e.g. skip `scheme_linear_weights`
    under interpretation 1's caveat) can do so without re-deriving offsets
    into a flat vector. Use `flatten_feature_vector` for the concatenated
    form.
    """
    window = np.asarray(window, dtype=float)
    width = window.shape[-1]
    if width not in (STENCIL_SIZE, WIDE_WINDOW):
        raise ValueError(
            'scalar_feature_vector expects a trailing window of size {} or {}; '
            'received {}.'.format(STENCIL_SIZE, WIDE_WINDOW, window.shape)
        )

    stencil_right = window[..., -STENCIL_SIZE:]  # always [j-2 .. j+2]

    normalized_stencil = normalize_teno_stencils(stencil_right)  # u_s, Eq. 1

    beta = teno5_smoothness_indicators(stencil_right)
    tau5 = teno5_global_smoothness(stencil_right)
    eta = tau5[..., None] / (beta + epsilon)  # Eq. 2

    scheme_identity = np.broadcast_to(
        LINEAR_WEIGHTS, stencil_right.shape[:-1] + (3,)
    ).copy()

    features = {
        'u_normalized': normalized_stencil,       # (..., 5)  Eq. 1
        'scheme_linear_weights': scheme_identity, # (..., 3)  "c~_WENO/TENO", see caveat 1
        'eta': eta,                               # (..., 3)  Eq. 2
        'j_right': _interface_jump(stencil_right, epsilon=epsilon),  # (...,) J_{j+1/2}
    }

    if width == WIDE_WINDOW:
        stencil_left = window[..., :STENCIL_SIZE]  # [j-3 .. j+1]
        features['j_left'] = _interface_jump(stencil_left, epsilon=epsilon)  # J_{j-1/2}

    return features


def euler_feature_vector(window, mach_number, pressure_gradient, epsilon=DEFAULT_EPSILON):
    """Extends scalar_feature_vector with the two Euler-only terms (Eq. 3).

    mach_number, pressure_gradient : array-like, shape matching window's
    leading dims. Computed by the caller from the conservative-variable
    stencil -- deriving these needs the Euler primitive-variable formulas
    (density/velocity/pressure/sound-speed from src/core/eulerEquations.py),
    which are system-specific and belong with whatever generates the Euler
    dataset (Phase 3), not duplicated in this scalar-PDE-agnostic module.
    """
    features = scalar_feature_vector(window, epsilon=epsilon)
    features['mach_number'] = np.asarray(mach_number, dtype=float)
    features['pressure_gradient'] = np.asarray(pressure_gradient, dtype=float)
    return features


_FLATTEN_ORDER = (
    'u_normalized', 'scheme_linear_weights', 'eta', 'j_left', 'j_right',
    'mach_number', 'pressure_gradient',
)


def flatten_feature_vector(features):
    """Concatenate a feature dict into one flat trailing axis, in the fixed
    order above, for feeding a network that expects a single input vector.
    Keys absent from `features` (e.g. `j_left` on a 5-wide window, or the
    Euler-only keys for a scalar problem) are skipped, not padded.
    """
    reference = features['u_normalized']
    pieces = []
    for name in _FLATTEN_ORDER:
        if name not in features:
            continue
        value = features[name]
        if value.ndim == reference.ndim - 1:
            value = value[..., None]
        pieces.append(value)
    return np.concatenate(pieces, axis=-1)
