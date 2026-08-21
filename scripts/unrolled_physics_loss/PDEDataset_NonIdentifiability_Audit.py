# -*- coding: utf-8 -*-
"""Phase 3 go/no-go audit for the Unrolled Physics Loss proposal's 60% real-PDE-projected training data.

UNROLLED_PHYSICS_LOSS_GUIDE.md's Phase 3 flags a known risk before building
this data bucket at scale: this project already tried "train on real
high-resolution PDE solutions projected to a coarse grid" once, for
WENO7-NN (Results.md Phase 2), and lost to non-identifiability -- stencils
that look nearly identical locally required genuinely different correct
answers depending on upstream wave history the network never sees. This
script measures that phenomenon directly, on real trajectory data, rather
than taking the earlier failure on faith.

Method: reuse GenerateTrainingData_EulerExact.py's own data-generation
pipeline (Riemann-problem and Shu-Osher-shock-interaction categories --
the two identifiability-risk categories; 'smooth'/'constant' samples are
skipped since they carry no history-dependence to audit) completely
unmodified. Normalize each generated input stencil the same way the network
would see it (min-max to [0,1], matching normalize_teno_stencils), quantize
to a coarse grid, and bin. Any bin containing samples from more than one
trajectory ('group_id') is a set of stencils that look alike locally but
came from different physical histories; if their targets disagree by more
than noise, that is a directly measured non-identifiability conflict, not
an inference from the earlier failure.

Run (pure NumPy + matplotlib, no Keras/JAX needed -- see feature_vector.py's
docstring for why this repo keeps data-generation code framework-free):

    conda run -n base python scripts/unrolled_physics_loss/PDEDataset_NonIdentifiability_Audit.py
"""

import os
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
_sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts', 'legacy_imitation_training'))

import argparse
from collections import defaultdict

import numpy as np

from GenerateTrainingData_EulerExact import (
    CATEGORY_PROPORTIONS,
    build_samples,
    generated_states,
)
from GenerateTrainingData_TENOExact import oracle_labels as teno5_oracle_labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems', type=int, default=300)
    parser.add_argument('--smooth-waves', type=int, default=0)
    parser.add_argument('--constant-states', type=int, default=0)
    parser.add_argument('--shu-variants', type=int, default=6)
    parser.add_argument('--shu-fine-cells', type=int, default=400)
    parser.add_argument('--shu-snapshots', type=int, default=20)
    parser.add_argument('--cells', type=int, default=64)
    parser.add_argument('--quadrature-points', type=int, default=9)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--decimals', type=int, nargs='+', default=[1, 2],
        help='Quantization precision(s) to audit at (coarser = more binning).',
    )
    return parser.parse_args()


def normalize_rows(stencils):
    minimum = stencils.min(axis=1, keepdims=True)
    value_range = stencils.max(axis=1, keepdims=True) - minimum
    safe_range = np.where(value_range > 0.0, value_range, 1.0)
    return (stencils - minimum) / safe_range


def audit_at_precision(normalized, targets, groups, decimals):
    quantized = np.round(normalized, decimals)
    bins = defaultdict(list)
    for index, key in enumerate(map(tuple, quantized)):
        bins[key].append(index)

    target_scale = float(np.std(targets)) + 1e-12
    conflict_bins = 0
    multi_group_bins = 0
    multi_group_samples = 0
    examples = []

    for key, indices in bins.items():
        indices = np.array(indices)
        group_subset = groups[indices]
        if len(np.unique(group_subset)) < 2:
            continue
        multi_group_bins += 1
        multi_group_samples += len(indices)
        target_subset = targets[indices]
        spread = float(target_subset.max() - target_subset.min())
        relative_spread = spread / target_scale
        if relative_spread > 0.1:
            conflict_bins += 1
            if len(examples) < 5:
                examples.append({
                    'quantized_shape': key,
                    'groups': group_subset[:6].tolist(),
                    'targets': target_subset[:6].tolist(),
                    'relative_spread': relative_spread,
                })

    return {
        'decimals': decimals,
        'total_samples': len(normalized),
        'total_bins': len(bins),
        'multi_group_bins': multi_group_bins,
        'multi_group_samples': multi_group_samples,
        'conflict_bins': conflict_bins,
        'conflict_rate_among_multi_group_bins': (
            conflict_bins / multi_group_bins if multi_group_bins else 0.0
        ),
        'examples': examples,
    }


def label_conflict_at_precision(normalized, oracle_labels_bits, groups, decimals):
    """Same audit as audit_at_precision, but on discrete keep/discard labels
    instead of continuous targets -- tests the Phase 3 mitigation this
    project's guide proposes: classification labels should be far less
    sensitive to non-identifiability than direct value regression, because
    many different discard patterns can be equally "not wrong" for a given
    stencil, where only one continuous value can be exactly right.
    """
    quantized = np.round(normalized, decimals)
    bins = defaultdict(list)
    for index, key in enumerate(map(tuple, quantized)):
        bins[key].append(index)

    conflict_bins = 0
    multi_group_bins = 0
    multi_group_samples = 0

    for key, indices in bins.items():
        indices = np.array(indices)
        group_subset = groups[indices]
        if len(np.unique(group_subset)) < 2:
            continue
        multi_group_bins += 1
        multi_group_samples += len(indices)
        label_subset = oracle_labels_bits[indices]
        if len(np.unique(label_subset, axis=0)) > 1:
            conflict_bins += 1

    return {
        'decimals': decimals,
        'multi_group_bins': multi_group_bins,
        'multi_group_samples': multi_group_samples,
        'conflict_bins': conflict_bins,
        'conflict_rate_among_multi_group_bins': (
            conflict_bins / multi_group_bins if multi_group_bins else 0.0
        ),
    }


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    risk_categories = {'riemann', 'shock_oscillation'}
    all_inputs, all_targets, all_groups, all_categories = [], [], [], []

    print('Generating real-PDE-projected samples (Riemann + Shu-Osher '
          'categories only)...', flush=True)
    problem_count = 0
    for category, group_id, state, exact_at_interface in generated_states(args, rng):
        if category not in risk_categories:
            continue
        inputs, targets, is_boundary = build_samples(
            state, exact_at_interface, args.cells, radius=args.radius,
        )
        keep = ~is_boundary
        if not np.any(keep):
            continue
        all_inputs.append(inputs[keep])
        all_targets.append(targets[keep])
        all_groups.append(np.full(int(keep.sum()), group_id, dtype=object))
        all_categories.append(np.full(int(keep.sum()), category, dtype=object))
        problem_count += 1

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    categories = np.concatenate(all_categories, axis=0)

    print('Problems/trajectories processed: {}'.format(problem_count))
    print('Total (interior, non-boundary) samples audited: {}'.format(len(inputs)))
    print('  riemann samples         : {}'.format(int(np.sum(categories == 'riemann'))))
    print('  shock_oscillation samples: {}'.format(int(np.sum(categories == 'shock_oscillation'))))

    normalized = normalize_rows(inputs)

    print('\nComputing oracle keep/discard labels for the mitigation test '
          '(exhaustive 8-mask search vs. these same targets)...', flush=True)
    oracle_bits, _ = teno5_oracle_labels(inputs, targets)

    print('\n' + '#' * 70)
    print('# VALUE-REGRESSION conflict rate (the risk the Unrolled Physics Loss proposal\'s plan as')
    print('# written would hit -- direct continuous target per stencil)')
    print('#' * 70)

    value_results = {}
    for decimals in args.decimals:
        result = audit_at_precision(normalized, targets, groups, decimals)
        value_results[decimals] = result
        print('\n' + '=' * 70)
        print('Quantization precision: {} decimal places'.format(decimals))
        print('=' * 70)
        print('Total stencil-shape bins            : {}'.format(result['total_bins']))
        print('Bins with samples from >1 trajectory : {}'.format(result['multi_group_bins']))
        print('  (samples in those bins)            : {}'.format(result['multi_group_samples']))
        print('Bins with a genuine target conflict  : {}'.format(result['conflict_bins']))
        print('Conflict rate among multi-trajectory bins: {:.2%}'.format(
            result['conflict_rate_among_multi_group_bins']
        ))
        if result['examples']:
            print('\nExample conflicts (same quantized shape, different targets):')
            for example in result['examples']:
                print('  shape={} groups={} targets={}'.format(
                    tuple(round(v, 2) for v in example['quantized_shape']),
                    example['groups'],
                    [round(t, 4) for t in example['targets']],
                ))

    print('\n' + '#' * 70)
    print('# CLASSIFICATION-LABEL conflict rate (the Phase 3 mitigation:')
    print('# oracle keep/discard mask instead of the raw continuous target)')
    print('#' * 70)

    label_results = {}
    for decimals in args.decimals:
        result = label_conflict_at_precision(normalized, oracle_bits, groups, decimals)
        label_results[decimals] = result
        print('\n' + '=' * 70)
        print('Quantization precision: {} decimal places'.format(decimals))
        print('=' * 70)
        print('Bins with samples from >1 trajectory : {}'.format(result['multi_group_bins']))
        print('Bins with a genuine label conflict   : {}'.format(result['conflict_bins']))
        print('Conflict rate among multi-trajectory bins: {:.2%}'.format(
            result['conflict_rate_among_multi_group_bins']
        ))

    print('\n' + '#' * 70)
    print('# SUMMARY: does the classification-label mitigation actually help?')
    print('#' * 70)
    for decimals in args.decimals:
        value_rate = value_results[decimals]['conflict_rate_among_multi_group_bins']
        label_rate = label_results[decimals]['conflict_rate_among_multi_group_bins']
        print(
            '  {} decimals: value-regression conflict {:.2%} -> '
            'classification-label conflict {:.2%}  ({})'.format(
                decimals, value_rate, label_rate,
                'IMPROVED' if label_rate < value_rate else 'NOT IMPROVED',
            )
        )


if __name__ == '__main__':
    main()
