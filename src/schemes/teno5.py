# -*- coding: utf-8 -*-
"""Deterministic TENO5 and neural-selector NN-TENO5 schemes.

The five-point stencil is ordered as::

    [f_(i-2), f_(i-1), f_i, f_(i+1), f_(i+2)]

The three third-order candidate reconstructions use:

    candidate 0: [f_(i-2), f_(i-1), f_i]
    candidate 1: [f_(i-1), f_i,     f_(i+1)]
    candidate 2: [f_i,     f_(i+1), f_(i+2)]

The optimal fifth-order linear weights are:

    [1/10, 3/5, 3/10]

For deterministic TENO5, candidate stencils are selected through the
TENO scale-separation procedure.

For NN-TENO5, the neural network predicts three independent troubled
probabilities:

    0 = smooth candidate, keep it
    1 = troubled candidate, discard it

The network selects candidate stencils only. Candidate polynomials,
optimal linear weights, weight renormalization, fallback behavior and
the final flux reconstruction remain deterministic.
"""

import numpy as np

from ..core.SimulationClasses import (
    FiniteVolumeMethod,
    FiniteVolumeMethodEuler,
)
from .teno_common import (
    combine_teno_candidates,
    normalize_teno_stencils,
)


# ---------------------------------------------------------------------------
# Scheme configuration
# ---------------------------------------------------------------------------

STENCIL_SIZE = 5
CANDIDATE_COUNT = 3
CENTER_INDEX = 2

LINEAR_WEIGHTS = np.array(
    [
        1.0 / 10.0,
        3.0 / 5.0,
        3.0 / 10.0,
    ],
    dtype=float,
)

# Standard fixed TENO5 parameters.
DEFAULT_EPSILON = 1.0e-40
DEFAULT_SCALE_CONSTANT = 1.0
DEFAULT_POWER = 6.0
DEFAULT_CUTOFF = 1.0e-5
DEFAULT_NEURAL_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _as_stencil_array(values):
    """Validate and return an array whose final axis is a TENO5 stencil."""

    values = np.asarray(values, dtype=float)

    if values.ndim < 1:
        raise ValueError(
            'TENO5 requires an array with a final stencil axis.'
        )

    if values.shape[-1] != STENCIL_SIZE:
        raise ValueError(
            'TENO5 expects a final stencil dimension of {}; received {}.'
            .format(
                STENCIL_SIZE,
                values.shape,
            )
        )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            'TENO5 received a stencil containing NaN or infinity.'
        )

    return values


def _validate_teno_parameters(
    epsilon,
    scale_constant,
    power,
    cutoff,
):
    """Validate deterministic TENO5 selector parameters."""

    epsilon = float(epsilon)
    scale_constant = float(scale_constant)
    power = float(power)
    cutoff = float(cutoff)

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            'epsilon must be finite and positive.'
        )

    if (
        not np.isfinite(scale_constant)
        or scale_constant <= 0.0
    ):
        raise ValueError(
            'scale_constant must be finite and positive.'
        )

    if not np.isfinite(power) or power <= 0.0:
        raise ValueError(
            'power must be finite and positive.'
        )

    if (
        not np.isfinite(cutoff)
        or cutoff < 0.0
        or cutoff > 1.0
    ):
        raise ValueError(
            'cutoff must belong to [0, 1].'
        )

    return (
        epsilon,
        scale_constant,
        power,
        cutoff,
    )


def _validate_thresholds(threshold):
    """Convert one scalar or three values into candidate thresholds."""

    thresholds = np.asarray(
        threshold,
        dtype=float,
    )

    if thresholds.ndim == 0:
        thresholds = np.full(
            CANDIDATE_COUNT,
            float(thresholds),
            dtype=float,
        )
    elif thresholds.shape != (CANDIDATE_COUNT,):
        raise ValueError(
            'NN-TENO5 threshold must be a scalar or have shape ({},); '
            'received {}.'
            .format(
                CANDIDATE_COUNT,
                thresholds.shape,
            )
        )

    if not np.isfinite(thresholds).all():
        raise ValueError(
            'NN-TENO5 thresholds must be finite.'
        )

    if np.any(
        (thresholds < 0.0)
        | (thresholds > 1.0)
    ):
        raise ValueError(
            'NN-TENO5 thresholds must belong to [0, 1].'
        )

    return thresholds


# ---------------------------------------------------------------------------
# Candidate reconstructions
# ---------------------------------------------------------------------------

def teno5_candidate_values(values):
    """Calculate the three third-order TENO5 candidate reconstructions.

    Parameters
    ----------
    values : ndarray
        Array with final dimension five.

    Returns
    -------
    ndarray
        Candidate values with shape ``values.shape[:-1] + (3,)``.
    """

    values = _as_stencil_array(values)

    value0 = values[..., 0]
    value1 = values[..., 1]
    value2 = values[..., 2]
    value3 = values[..., 3]
    value4 = values[..., 4]

    candidate0 = (
        (1.0 / 3.0) * value0
        - (7.0 / 6.0) * value1
        + (11.0 / 6.0) * value2
    )

    candidate1 = (
        -(1.0 / 6.0) * value1
        + (5.0 / 6.0) * value2
        + (1.0 / 3.0) * value3
    )

    candidate2 = (
        (1.0 / 3.0) * value2
        + (5.0 / 6.0) * value3
        - (1.0 / 6.0) * value4
    )

    return np.stack(
        [
            candidate0,
            candidate1,
            candidate2,
        ],
        axis=-1,
    )


# Preserve the private function name used by the existing tests.
def _candidate_values(values):
    return teno5_candidate_values(values)


# ---------------------------------------------------------------------------
# Smoothness indicators
# ---------------------------------------------------------------------------

def teno5_smoothness_indicators(values):
    """Calculate the three Jiang--Shu candidate smoothness indicators.

    Parameters
    ----------
    values : ndarray
        Array with final dimension five.

    Returns
    -------
    ndarray
        Smoothness indicators with shape
        ``values.shape[:-1] + (3,)``.
    """

    values = _as_stencil_array(values)

    value0 = values[..., 0]
    value1 = values[..., 1]
    value2 = values[..., 2]
    value3 = values[..., 3]
    value4 = values[..., 4]

    beta0 = (
        (13.0 / 12.0)
        * (
            value0
            - 2.0 * value1
            + value2
        )**2
        + 0.25
        * (
            value0
            - 4.0 * value1
            + 3.0 * value2
        )**2
    )

    beta1 = (
        (13.0 / 12.0)
        * (
            value1
            - 2.0 * value2
            + value3
        )**2
        + 0.25
        * (
            value1
            - value3
        )**2
    )

    beta2 = (
        (13.0 / 12.0)
        * (
            value2
            - 2.0 * value3
            + value4
        )**2
        + 0.25
        * (
            3.0 * value2
            - 4.0 * value3
            + value4
        )**2
    )

    beta = np.stack(
        [
            beta0,
            beta1,
            beta2,
        ],
        axis=-1,
    )

    # The analytical formulas are sums of squares, but a maximum with zero
    # prevents very small negative values caused by roundoff.
    return np.maximum(
        beta,
        0.0,
    )


def teno5_global_smoothness(values):
    """Calculate the fifth-order global smoothness measure tau_5."""

    beta = teno5_smoothness_indicators(
        values
    )

    return np.abs(
        beta[..., 0]
        - beta[..., 2]
    )


# ---------------------------------------------------------------------------
# Deterministic TENO5 selector
# ---------------------------------------------------------------------------

def teno5_selection_values(
    values,
    epsilon=DEFAULT_EPSILON,
    scale_constant=DEFAULT_SCALE_CONSTANT,
    power=DEFAULT_POWER,
):
    """Calculate normalized TENO5 scale-separation values ``chi_k``.

    This function does not apply the cutoff. It is useful for:

    - deterministic reconstruction;
    - dataset generation;
    - debugging selection decisions;
    - studying alternative cutoff values.

    Parameters
    ----------
    values : ndarray
        Input stencils with final dimension five.
    epsilon : float
        Small denominator protection.
    scale_constant : float
        Positive TENO scale-separation constant.
    power : float
        Positive exponent applied to the scale-separation measure.

    Returns
    -------
    ndarray
        Normalized selection values with final dimension three.
    """

    epsilon, scale_constant, power, _ = _validate_teno_parameters(
        epsilon,
        scale_constant,
        power,
        DEFAULT_CUTOFF,
    )

    beta = teno5_smoothness_indicators(
        values
    )

    tau5 = np.abs(
        beta[..., 0]
        - beta[..., 2]
    )

    # Original TENO scale-separation measure:
    #
    # gamma_k = (C + tau_5 / (beta_k + epsilon))**power
    #
    # It is evaluated in logarithmic form to prevent overflow when one
    # candidate is much smoother than the others.
    with np.errstate(
        divide='raise',
        invalid='raise',
        over='raise',
    ):
        ratio = tau5[..., None] / (beta + epsilon)

        if scale_constant == 1.0: 
            log_gamma = power * np.log1p(ratio)
        else:
            log_gamma = power * np.log(scale_constant + ratio)

    # Softmax-style normalization in the logarithmic domain.
    log_gamma = (
        log_gamma
        - np.max(
            log_gamma,
            axis=-1,
            keepdims=True,
        )
    )

    gamma = np.exp(
        log_gamma
    )

    gamma_sum = np.sum(
        gamma,
        axis=-1,
        keepdims=True,
    )

    if (
        not np.isfinite(gamma).all()
        or not np.isfinite(gamma_sum).all()
        or np.any(gamma_sum <= 0.0)
    ):
        raise FloatingPointError(
            'TENO5 produced invalid scale-separation values.'
        )

    chi = gamma / gamma_sum

    return chi


def teno5_keep_mask(
    values,
    epsilon=DEFAULT_EPSILON,
    scale_constant=DEFAULT_SCALE_CONSTANT,
    power=DEFAULT_POWER,
    cutoff=DEFAULT_CUTOFF,
):
    """Return the deterministic TENO5 candidate keep mask.

    ``True`` means that the candidate is considered smooth and retained.
    ``False`` means that the candidate is considered troubled and discarded.
    """

    (
        epsilon,
        scale_constant,
        power,
        cutoff,
    ) = _validate_teno_parameters(
        epsilon,
        scale_constant,
        power,
        cutoff,
    )

    chi = teno5_selection_values(
        values,
        epsilon=epsilon,
        scale_constant=scale_constant,
        power=power,
    )

    return chi >= cutoff


def teno5_troubled_labels(
    values,
    epsilon=DEFAULT_EPSILON,
    scale_constant=DEFAULT_SCALE_CONSTANT,
    power=DEFAULT_POWER,
    cutoff=DEFAULT_CUTOFF,
):
    """Generate NN-TENO5 training labels.

    The returned convention is:

    - ``0``: smooth candidate, keep it;
    - ``1``: troubled candidate, discard it.

    Returns
    -------
    ndarray
        Float32 binary labels with final dimension three.
    """

    keep_mask = teno5_keep_mask(
        values,
        epsilon=epsilon,
        scale_constant=scale_constant,
        power=power,
        cutoff=cutoff,
    )

    return np.logical_not(
        keep_mask
    ).astype(
        np.float32
    )


# Preserve the old private function name.
def _deterministic_keep(values):
    return teno5_keep_mask(values)


# ---------------------------------------------------------------------------
# Deterministic reconstruction
# ---------------------------------------------------------------------------

def teno5_reconstruct(
    values,
    epsilon=DEFAULT_EPSILON,
    scale_constant=DEFAULT_SCALE_CONSTANT,
    power=DEFAULT_POWER,
    cutoff=DEFAULT_CUTOFF,
):
    """Reconstruct the interface flux using deterministic TENO5."""

    values = _as_stencil_array(values)

    candidates = teno5_candidate_values(
        values
    )

    keep_mask = teno5_keep_mask(
        values,
        epsilon=epsilon,
        scale_constant=scale_constant,
        power=power,
        cutoff=cutoff,
    )

    # If every candidate is rejected, use the central/upwind value.
    fallback = values[..., CENTER_INDEX]

    return combine_teno_candidates(
        candidates,
        keep_mask,
        LINEAR_WEIGHTS,
        fallback,
    )


def _reconstruct(values):
    """Default deterministic reconstruction used by the scheme wrapper."""

    return teno5_reconstruct(values)


# ---------------------------------------------------------------------------
# Neural candidate selection
# ---------------------------------------------------------------------------

def teno5_neural_probabilities(
    model,
    values,
):
    """Predict one troubled probability for each TENO5 candidate.

    The entire leading shape is flattened into one model batch. This allows
    both scalar stencils ``(nx, 5)`` and Euler characteristic stencils
    ``(nx, nchar, 5)`` to use a single neural-network prediction call.
    """

    values = _as_stencil_array(values)

    normalized = normalize_teno_stencils(
        values
    )

    flat_inputs = normalized.reshape(
        -1,
        STENCIL_SIZE,
    )

    try:
        prediction = model.predict(
            flat_inputs,
            verbose=0,
        )
    except TypeError:
        # Compatibility with lightweight test doubles or older Keras APIs.
        prediction = model.predict(
            flat_inputs
        )

    probabilities = np.asarray(
        prediction,
        dtype=float,
    )

    expected_shape = (
        flat_inputs.shape[0],
        CANDIDATE_COUNT,
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            'NN-TENO5 model returned shape {}; expected {}.'
            .format(
                probabilities.shape,
                expected_shape,
            )
        )

    if not np.isfinite(probabilities).all():
        raise FloatingPointError(
            'NN-TENO5 model produced NaN or infinite probabilities.'
        )

    # Sigmoid models should naturally remain inside [0, 1]. A strict check
    # exposes an incorrectly configured or incompatible model.
    probability_tolerance = 1.0e-7

    if np.any(
        probabilities < -probability_tolerance
    ) or np.any(
        probabilities > 1.0 + probability_tolerance
    ):
        raise ValueError(
            'NN-TENO5 model outputs must represent probabilities in [0, 1].'
        )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    return probabilities.reshape(
        values.shape[:-1]
        + (CANDIDATE_COUNT,)
    )


def teno5_neural_keep_mask(
    model,
    values,
    threshold=DEFAULT_NEURAL_THRESHOLD,
):
    """Return the neural candidate keep mask.

    ``probability >= threshold`` means troubled/discard.
    """

    thresholds = _validate_thresholds(
        threshold
    )

    probabilities = teno5_neural_probabilities(
        model,
        values,
    )

    troubled_mask = (
        probabilities
        >= thresholds
    )

    return np.logical_not(
        troubled_mask
    )


# Preserve the private helper expected by older code.
def _neural_keep(
    model,
    values,
    threshold,
):
    return teno5_neural_keep_mask(
        model,
        values,
        threshold=threshold,
    )


def teno5_neural_reconstruct(
    model,
    values,
    threshold=DEFAULT_NEURAL_THRESHOLD,
):
    """Reconstruct an interface flux using neural candidate selection."""

    values = _as_stencil_array(values)

    candidates = teno5_candidate_values(
        values
    )

    keep_mask = teno5_neural_keep_mask(
        model,
        values,
        threshold=threshold,
    )

    fallback = values[..., CENTER_INDEX]

    return combine_teno_candidates(
        candidates,
        keep_mask,
        LINEAR_WEIGHTS,
        fallback,
    )


# ---------------------------------------------------------------------------
# Finite-volume scheme builders
# ---------------------------------------------------------------------------

def TENO5(
    epsilon=DEFAULT_EPSILON,
    scale_constant=DEFAULT_SCALE_CONSTANT,
    power=DEFAULT_POWER,
    cutoff=DEFAULT_CUTOFF,
    boundary='periodic',
):
    """Build the scalar deterministic TENO5 reconstruction scheme."""

    _validate_teno_parameters(
        epsilon,
        scale_constant,
        power,
        cutoff,
    )

    def scheme(values):
        return teno5_reconstruct(
            values,
            epsilon=epsilon,
            scale_constant=scale_constant,
            power=power,
            cutoff=cutoff,
        )

    return FiniteVolumeMethod(
        STENCIL_SIZE,
        scheme,
        boundary=boundary,
    )


def TENO5euler(
    epsilon=DEFAULT_EPSILON,
    scale_constant=DEFAULT_SCALE_CONSTANT,
    power=DEFAULT_POWER,
    cutoff=DEFAULT_CUTOFF,
    boundary='transmissive',
):
    """Build the characteristic deterministic Euler TENO5 scheme."""

    _validate_teno_parameters(
        epsilon,
        scale_constant,
        power,
        cutoff,
    )

    def scheme(values):
        return teno5_reconstruct(
            values,
            epsilon=epsilon,
            scale_constant=scale_constant,
            power=power,
            cutoff=cutoff,
        )

    return FiniteVolumeMethodEuler(
        STENCIL_SIZE,
        scheme,
        boundary=boundary,
    )


def NNMethod(
    model,
    threshold=DEFAULT_NEURAL_THRESHOLD,
    boundary='periodic',
):
    """Build the scalar NN-TENO5 reconstruction scheme.

    Parameters
    ----------
    model
        Neural model mapping normalized arrays of shape ``(batch, 5)`` to
        troubled probabilities of shape ``(batch, 3)``.
    threshold : float or array-like
        One shared threshold or three per-candidate thresholds.
    boundary : str
        Scalar finite-volume boundary condition.
    """

    thresholds = _validate_thresholds(
        threshold
    )

    def scheme(values):
        values = np.asarray(
            values,
            dtype=float,
        )

        if (
            values.ndim != 2
            or values.shape[1] != STENCIL_SIZE
        ):
            raise ValueError(
                'Scalar NN-TENO5 expects input shape (nx, 5); '
                'received {}.'
                .format(values.shape)
            )

        return teno5_neural_reconstruct(
            model,
            values,
            threshold=thresholds,
        )

    return FiniteVolumeMethod(
        STENCIL_SIZE,
        scheme,
        boundary=boundary,
    )


def NNEuler(
    model,
    threshold=DEFAULT_NEURAL_THRESHOLD,
    boundary='transmissive',
):
    """Build the characteristic Euler NN-TENO5 reconstruction scheme.

    All cells and characteristic fields are evaluated in one neural-network
    batch.
    """

    thresholds = _validate_thresholds(
        threshold
    )

    def scheme(values):
        values = np.asarray(
            values,
            dtype=float,
        )

        if (
            values.ndim != 3
            or values.shape[2] != STENCIL_SIZE
        ):
            raise ValueError(
                'Euler NN-TENO5 expects input shape (nx, nchar, 5); '
                'received {}.'
                .format(values.shape)
            )

        return teno5_neural_reconstruct(
            model,
            values,
            threshold=thresholds,
        )

    return FiniteVolumeMethodEuler(
        STENCIL_SIZE,
        scheme,
        boundary=boundary,
    )


__all__ = [
    'STENCIL_SIZE',
    'CANDIDATE_COUNT',
    'CENTER_INDEX',
    'LINEAR_WEIGHTS',
    'DEFAULT_EPSILON',
    'DEFAULT_SCALE_CONSTANT',
    'DEFAULT_POWER',
    'DEFAULT_CUTOFF',
    'DEFAULT_NEURAL_THRESHOLD',
    'teno5_candidate_values',
    'teno5_smoothness_indicators',
    'teno5_global_smoothness',
    'teno5_selection_values',
    'teno5_keep_mask',
    'teno5_troubled_labels',
    'teno5_reconstruct',
    'teno5_neural_probabilities',
    'teno5_neural_keep_mask',
    'teno5_neural_reconstruct',
    'TENO5',
    'TENO5euler',
    'NNMethod',
    'NNEuler',
]
