# -*- coding: utf-8 -*-
"""Generate an oracle-relabeled NN-TENO5 dataset from synthetic functions.

Each sample is a five-point stencil of exact cell averages of a randomly
drawn analytic test function (sinusoid, double sinusoid, polynomial, step,
or absolute-value kink), sampled at a random cell width and interface
location. Cell averages are computed from closed-form antiderivatives, not
quadrature, so generation is fully vectorized and exact up to floating
point.

Unlike ``teno5_troubled_labels`` (which reproduces TENO5's own
scale-separation threshold decision), the label here is the joint
keep/discard mask -- searched exhaustively over all eight combinations --
that minimizes reconstruction error against the *true* interface point
value of the underlying function, using the same candidate formulas and
``combine_teno_candidates`` weighting the real scheme and NN-TENO5 use at
inference time. This is a per-stencil-local oracle: it is only meaningful
for static, non-interacting test functions where the correct interface
value is fully determined by the stencil itself, not by upstream wave
history (see Results.md Phase 6 for why history-dependent problems like
Riemann/shock-oscillation targets do not admit this kind of local oracle).
"""

import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import itertools
import json
import os

import numpy as np

from src.schemes.teno5 import (
    CENTER_INDEX,
    LINEAR_WEIGHTS,
    teno5_candidate_values,
    teno5_troubled_labels,
)
from src.schemes.teno_common import combine_teno_candidates

STENCIL_OFFSETS = np.array([-2, -1, 0, 1, 2], dtype=float)

# Ordered with the most-kept mask first (all-True, i.e. "keep everything")
# down to the fewest-kept, so that the tie-break in oracle_labels (which
# keeps the first mask reaching a given error, via strict "<") lands on
# keeping more candidates rather than discarding them. This matters:
# whenever the underlying function is locally smooth (all three candidates
# already agree), every mask is an exact tie -- with no ordering bias, the
# tie silently resolves to whatever mask happens to be visited first, which
# would otherwise make the oracle mislabel huge numbers of ordinary smooth
# stencils as "discard everything" for no real reason.
ALL_MASKS = np.array(
    sorted(
        itertools.product([False, True], repeat=3),
        key=lambda mask: -sum(mask),
    ),
    dtype=bool,
)

# Reconstruction differences below this are treated as numerical noise, not
# a genuine improvement, so near-ties don't get counted as disagreements.
TIE_TOLERANCE = 1.0e-10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate an oracle-relabeled five-point-stencil dataset for '
            'NN-TENO5, using true local reconstruction error instead of '
            "TENO5's own scale-separation decision."
        )
    )
    parser.add_argument('--samples', type=int, default=75000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--stencils-path',
        default='data/teno5ExactStencils.csv',
    )
    parser.add_argument(
        '--labels-path',
        default='data/teno5ExactLabels.csv',
    )
    parser.add_argument(
        '--metadata-path',
        default='data/teno5ExactLabels_metadata.json',
    )
    parser.add_argument('--min-cell-width', type=float, default=0.01)
    parser.add_argument('--max-cell-width', type=float, default=0.3)
    parser.add_argument(
        '--family-weights',
        nargs=5,
        type=float,
        default=[0.30, 0.20, 0.20, 0.20, 0.10],
        metavar=('SINE', 'DOUBLE_SINE', 'POLY', 'STEP', 'KINK'),
    )
    return parser.parse_args()


def _cell_bounds(center, width):
    return center - 0.5 * width, center + 0.5 * width


def sample_sine(rng, count, width):
    amplitude = rng.uniform(0.2, 1.0, count)
    # Tie wavelength to the local cell width instead of sampling it
    # independently, so "smooth" samples stay physically well-posed --
    # otherwise a high wavenumber paired with a wide cell packs many
    # oscillations into one 5-point stencil, which neither TENO5 nor any
    # oracle can meaningfully reconstruct (aliasing, not a real test of
    # candidate selection). 2 points/wavelength is the Nyquist limit
    # (genuinely hard/marginal); 20 is comfortably well-resolved.
    points_per_wavelength = rng.uniform(2.0, 20.0, count)
    wavelength = width * points_per_wavelength
    wavenumber = 1.0 / wavelength
    phase = rng.uniform(0.0, 2.0 * np.pi, count)
    offset = rng.uniform(-0.3, 0.3, count)
    omega = 2.0 * np.pi * wavenumber

    def point_value(x):
        return amplitude * np.sin(omega * x + phase) + offset

    def cell_average(center, width):
        left, right = _cell_bounds(center, width)
        omega_col = omega[:, None]
        phase_col = phase[:, None]
        return offset[:, None] + amplitude[:, None] * (
            np.cos(omega_col * left + phase_col)
            - np.cos(omega_col * right + phase_col)
        ) / (omega_col * width)

    return point_value, cell_average


def sample_double_sine(rng, count, width):
    point_a, average_a = sample_sine(rng, count, width)
    point_b, average_b = sample_sine(rng, count, width)

    def point_value(x):
        return 0.5 * (point_a(x) + point_b(x))

    def cell_average(center, width):
        return 0.5 * (
            average_a(center, width) + average_b(center, width)
        )

    return point_value, cell_average


def sample_polynomial(rng, count, width):
    degree = 3
    coefficients = rng.uniform(-1.0, 1.0, (count, degree + 1))

    def point_value(x):
        value = np.zeros_like(x)
        for power in range(degree + 1):
            value = value + coefficients[:, power] * x ** power
        return value

    def cell_average(center, width):
        left, right = _cell_bounds(center, width)
        total = np.zeros_like(center)
        for power in range(degree + 1):
            total = total + coefficients[:, power][:, None] * (
                right ** (power + 1) - left ** (power + 1)
            ) / ((power + 1) * width)
        return total

    return point_value, cell_average


def sample_step(rng, count, width):
    low = rng.uniform(-1.0, 0.5, count)
    high = low + rng.uniform(0.3, 1.5, count)
    jump = rng.uniform(-2.0, 2.0, count)

    def point_value(x):
        return np.where(x < jump, low, high)

    def cell_average(center, width):
        left, right = _cell_bounds(center, width)
        left_fraction = np.clip(
            (jump[:, None] - left) / width,
            0.0,
            1.0,
        )
        return (
            low[:, None] * left_fraction
            + high[:, None] * (1.0 - left_fraction)
        )

    return point_value, cell_average


def sample_kink(rng, count, width):
    # |x - x0| scaled/offset: continuous, but with a slope discontinuity --
    # a milder feature than a jump, between smooth and fully discontinuous.
    slope = rng.uniform(0.5, 3.0, count)
    kink = rng.uniform(-2.0, 2.0, count)
    offset = rng.uniform(-0.5, 0.5, count)

    def point_value(x):
        return slope * np.abs(x - kink) + offset

    def cell_average(center, width):
        left, right = _cell_bounds(center, width)
        kink_col = kink[:, None]

        def antiderivative(x):
            # Integral of |x - kink| is 0.5*(x-kink)*|x-kink|.
            shifted = x - kink_col
            return 0.5 * shifted * np.abs(shifted)

        return offset[:, None] + slope[:, None] * (
            antiderivative(right) - antiderivative(left)
        ) / width

    return point_value, cell_average


FAMILIES = (
    ('sine', sample_sine),
    ('double_sine', sample_double_sine),
    ('polynomial', sample_polynomial),
    ('step', sample_step),
    ('kink', sample_kink),
)


def generate_family_batch(name, sampler, rng, count, min_width, max_width):
    interface_center = rng.uniform(-1.0, 1.0, count)
    width = rng.uniform(min_width, max_width, count)

    point_value, cell_average = sampler(rng, count, width)

    # Cell i is centered on (interface_center - 0.5*width), so its right
    # face -- the reconstructed interface -- lands exactly on
    # interface_center.
    cell_i_center = interface_center - 0.5 * width

    cell_centers = (
        cell_i_center[:, None]
        + STENCIL_OFFSETS[None, :] * width[:, None]
    )
    stencils = cell_average(cell_centers, width[:, None])
    true_interface_value = point_value(interface_center)

    return name, stencils, true_interface_value


def oracle_labels(stencils, true_values):
    candidates = teno5_candidate_values(stencils)
    fallback = stencils[..., CENTER_INDEX]

    best_error = None
    best_mask_index = np.zeros(stencils.shape[0], dtype=int)

    for mask_index, mask in enumerate(ALL_MASKS):
        combined = combine_teno_candidates(
            candidates,
            np.broadcast_to(mask, candidates.shape),
            LINEAR_WEIGHTS,
            fallback,
        )
        error = np.abs(combined - true_values)
        if best_error is None:
            best_error = error
            best_mask_index[:] = mask_index
        else:
            improved = error < best_error - TIE_TOLERANCE
            best_error = np.where(improved, error, best_error)
            best_mask_index = np.where(
                improved,
                mask_index,
                best_mask_index,
            )

    best_keep_mask = ALL_MASKS[best_mask_index]
    oracle_troubled = np.logical_not(best_keep_mask).astype(np.float32)
    return oracle_troubled, best_error


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    weights = np.array(args.family_weights, dtype=float)
    weights = weights / weights.sum()
    family_counts = np.random.default_rng(args.seed).multinomial(
        args.samples,
        weights,
    )

    all_stencils = []
    all_names = []
    for (name, sampler), count in zip(FAMILIES, family_counts):
        if count == 0:
            continue
        _, stencils, true_values = generate_family_batch(
            name,
            sampler,
            rng,
            int(count),
            args.min_cell_width,
            args.max_cell_width,
        )
        all_stencils.append((stencils, true_values))
        all_names.extend([name] * int(count))

    stencils = np.concatenate([s for s, _ in all_stencils], axis=0)
    true_values = np.concatenate([t for _, t in all_stencils], axis=0)
    family_names = np.array(all_names)

    permutation = rng.permutation(len(stencils))
    stencils = stencils[permutation]
    true_values = true_values[permutation]
    family_names = family_names[permutation]

    if not np.isfinite(stencils).all() or not np.isfinite(true_values).all():
        raise FloatingPointError(
            'Generated non-finite stencil or target values.'
        )

    labels, oracle_error = oracle_labels(stencils, true_values)
    teno5_labels = teno5_troubled_labels(stencils)

    agreement = np.all(labels == teno5_labels, axis=-1)
    per_candidate_agreement = np.mean(
        labels == teno5_labels,
        axis=0,
    )

    parent = os.path.dirname(args.stencils_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    np.savetxt(args.stencils_path, stencils.T, delimiter=',')
    np.savetxt(args.labels_path, labels, delimiter=',')

    diagnostics = {
        'samples': int(len(stencils)),
        'family_weights': dict(zip(
            [name for name, _ in FAMILIES],
            weights.tolist(),
        )),
        'family_sample_counts': {
            name: int(np.sum(family_names == name))
            for name, _ in FAMILIES
        },
        'oracle_vs_teno5': {
            'exact_mask_agreement': float(np.mean(agreement)),
            'per_candidate_agreement': per_candidate_agreement.tolist(),
        },
        'oracle_error': {
            'mean': float(np.mean(oracle_error)),
            'max': float(np.max(oracle_error)),
        },
        'per_family_disagreement': {
            name: float(
                1.0
                - np.mean(agreement[family_names == name])
            )
            for name, _ in FAMILIES
            if np.any(family_names == name)
        },
    }

    with open(args.metadata_path, 'w', encoding='utf-8') as stream:
        json.dump(diagnostics, stream, indent=2, sort_keys=True)

    print('Samples generated: {}'.format(diagnostics['samples']))
    print(
        'Oracle vs. TENO5 exact-mask agreement: {:.4%}'.format(
            diagnostics['oracle_vs_teno5']['exact_mask_agreement']
        )
    )
    print(
        'Per-candidate agreement: {}'.format(
            per_candidate_agreement
        )
    )
    print('Per-family disagreement rate:')
    for name, rate in diagnostics['per_family_disagreement'].items():
        print('  {}: {:.4%}'.format(name, rate))
    print('Stencils written to {}'.format(
        os.path.abspath(args.stencils_path)
    ))
    print('Oracle labels written to {}'.format(
        os.path.abspath(args.labels_path)
    ))
    print('Metadata written to {}'.format(
        os.path.abspath(args.metadata_path)
    ))


if __name__ == '__main__':
    main()
