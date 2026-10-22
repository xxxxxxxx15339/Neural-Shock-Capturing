"""Unified, non-overwriting P0 evaluation CLI.

From the repository root, in the isolated JAX environment:
    python -m src.differentiable.evaluate benchmark --suite canonical --output NEW_DIRECTORY
    python -m src.differentiable.evaluate reference --suite all --output NEW_DIRECTORY
    python -m src.differentiable.evaluate smooth --output NEW_DIRECTORY
"""
import argparse
import hashlib
from pathlib import Path
import sys

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

from .artifacts import create_output, finish_manifest, provenance, write_json, write_metrics_csv
from .benchmark_cases import MANIFEST_PATH, grid, initial_state, read_manifest
from .evaluation import build_reference, compare_methods, errors, execute, plot_case
from .model_io import CheckpointError, load_checkpoint, sha256_file
from .reconstruction import make_reconstruction


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('benchmark', 'reference', 'smooth'))
    p.add_argument('--manifest', type=Path, default=MANIFEST_PATH)
    p.add_argument('--suite', choices=('canonical', 'heldout', 'all'), default='canonical')
    p.add_argument('--problem', choices=('advection', 'burgers', 'sod', 'shu_osher', 'all'), default='all')
    p.add_argument('--case', action='append', help='Select predeclared case IDs; repeat to select several')
    p.add_argument('--output', type=Path, required=True, help='A new directory; existing directories are rejected')
    p.add_argument('--repair-policy', choices=('none', 'historical-repair'), default='none')
    return p


def select_cases(manifest, args):
    suites = list(manifest['suites']) if args.suite == 'all' else [args.suite]
    cases = [case for suite in suites for case in manifest['suites'][suite]]
    if args.problem != 'all':
        cases = [c for c in cases if c['problem'] == args.problem]
    if args.case:
        available = {c['id'] for c in cases}
        if not set(args.case) <= available:
            raise ValueError(f'Unknown or filtered case IDs: {set(args.case)-available}')
        cases = [c for c in cases if c['id'] in args.case]
    if args.action == 'reference':
        cases = [c for c in cases if c['problem'] == 'shu_osher']
    if not cases:
        raise ValueError('No cases match the selection')
    return cases


def run_benchmarks(cases, models, manifest, args, output):
    all_records, results = [], []
    any_invalid = False
    for case in cases:
        print(f"Evaluating {case['id']}", flush=True)
        destination = output / case['id']
        destination.mkdir()
        write_json(destination / 'case.json', case)
        reference, reference_info, reference_arrays = build_reference(case)
        reference_info['sample_array_sha256'] = hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest()
        write_json(destination / 'reference.json', reference_info)
        np.savez_compressed(destination / 'reference.npz', x=grid(case), state=reference, **reference_arrays)
        model = models['scalar' if case['problem'] in ('advection', 'burgers') else 'euler']
        records, finals = [], {}
        for method in manifest['methods']:
            policy = args.repair_policy if case['problem'] in ('sod', 'shu_osher') else 'none'
            final, diagnostics, history, timing = execute(case, make_reconstruction(model, method, manifest['fixed_lambda']), repair_policy=policy)
            record = dict(case=case['id'], method=method, time_grid=timing,
                          metric_definition='mean component L1' if final.ndim == 2 else 'root mean square error',
                          **errors(final, reference), **diagnostics)
            record['total_variation'] = float(np.sum(np.abs(np.diff(final[:, 0])))) if final.ndim == 2 else float(np.sum(np.abs(final - np.roll(final, 1))))
            if final.ndim == 1:
                initial = initial_state(case)
                record['final_minimum'] = float(np.min(final))
                record['final_maximum'] = float(np.max(final))
                record['initial_range_violation'] = float(max(0., np.min(initial)-np.min(final), np.max(final)-np.max(initial)))
            write_json(destination / f'{method}.json', record)
            np.savez_compressed(destination / f'{method}.npz', x=grid(case), initial=initial_state(case), state=final,
                                **{'history_' + k: v for k, v in history.items()})
            records.append(record)
            finals[method] = final
            any_invalid |= not diagnostics['valid_for_accuracy_claim']
            print(f"  {method:14} error={record['primary_error']:.9g} valid={record['valid_for_accuracy_claim']}", flush=True)
        comparison = compare_methods(records, reference_info.get('trustworthy_for_defined_samples', False))
        result = dict(case=case, methods=records, comparisons=comparison)
        write_json(destination / 'comparison.json', comparison)
        plot_case(case, grid(case), reference, finals, destination / 'profiles.png')
        results.append(result)
        all_records.extend(records)
    write_json(output / 'metrics.json', results)
    write_metrics_csv(output / 'metrics.csv', all_records)
    return int(any_invalid)


def main(argv=None):
    args = parser().parse_args(argv)
    output = None
    try:
        manifest = read_manifest(args.manifest)
        cases = select_cases(manifest, args)
        if set(manifest['methods']) != {'teno5', 'neural-only', 'fixed-blend', 'learned-blend'}:
            raise ValueError('P0 manifests must declare the four matched ablation methods')
        models = {physics: load_checkpoint(info['path'], info['sha256']) for physics, info in manifest['models'].items()}
        if set(models) != {'scalar', 'euler'}:
            raise ValueError('Explicit scalar and Euler model identities are required')
        environment = provenance(args.manifest)
        output = create_output(args.output)
        write_json(output / 'provenance.json', environment)
        write_json(output / 'configuration.json', dict(manifest=manifest, manifest_sha256=sha256_file(args.manifest),
                   action=args.action, suite=args.suite, selected_cases=[c['id'] for c in cases],
                   repair_policy=args.repair_policy, gamma=1.4, integrator='SSPRK3',
                   checkpoint_metadata={k: v.metadata() for k, v in models.items()}))
        if args.action == 'benchmark':
            code = run_benchmarks(cases, models, manifest, args, output)
        elif args.action == 'reference':
            from .reference_study import run_reference_study
            records = [run_reference_study(c, models['euler'], manifest['methods'], manifest['fixed_lambda'],
                         manifest['reference_study'], output / c['id']) for c in cases]
            write_json(output / 'reference_summary.json', records)
            code = int(any(not all(row['coarse_valid'].values()) for row in records))
        else:
            from .smooth_study import run_smooth_study
            records = run_smooth_study(models, manifest['methods'], manifest['fixed_lambda'], output)
            write_json(output / 'smooth_summary.json', records)
            code = int(any(not row['diagnostics']['valid_for_accuracy_claim']
                           for methods in records['rollout'].values() for rows in methods.values() for row in rows))
        write_json(output / 'status.json', dict(completed=True, exit_code=code,
                   all_rollouts_admissible=code == 0,
                   reference_win_claims='See per-comparison validity and reference-convergence findings'))
        finish_manifest(output)
        return code
    except (ValueError, OSError, FloatingPointError, CheckpointError) as exc:
        print(f'EVALUATION FAILED: {exc}', file=sys.stderr)
        if output is not None:
            write_json(output / 'failure.json', dict(type=type(exc).__name__, message=str(exc)))
            finish_manifest(output)
        return 2


if __name__ == '__main__':
    sys.exit(main())
