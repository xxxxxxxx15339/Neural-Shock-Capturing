"""Evaluation-only primitives shared by the CLI and reference studies."""
import time

import numpy as np

from .benchmark_cases import analytic_reference, grid, initial_state, reference_case, time_grid
from .diagnostics import diagnostic_rollout
from .reconstruction import make_reconstruction


def execute(case, reconstruct, refinement=1, repair_policy='none'):
    initial = initial_state(case)
    timing = time_grid(case, initial, refinement)
    started = time.perf_counter()
    final, summary, history = diagnostic_rollout(
        initial, reconstruct, timing['dt'], timing['dx'], timing['steps'],
        system='euler' if case['problem'] in ('sod', 'shu_osher') else case['problem'],
        boundary=case['boundary'], wave_speed=case.get('wave_speed', 1.0),
        alpha=case.get('alpha', 1.0), repair_policy=repair_policy)
    timing['wall_seconds_including_compilation'] = time.perf_counter() - started
    return final, summary, history, timing


def errors(final, reference):
    difference = np.asarray(final) - reference
    absolute = np.abs(difference)
    component_l1 = np.atleast_1d(np.mean(absolute, axis=0))
    return dict(
        primary_error=float(np.mean(absolute)) if final.ndim == 2 else float(np.sqrt(np.mean(difference ** 2))),
        l1=float(np.mean(absolute)), l2=float(np.sqrt(np.mean(difference ** 2))),
        linf=float(np.max(absolute)), component_l1=component_l1.tolist(),
        component_l2=np.atleast_1d(np.sqrt(np.mean(difference ** 2, axis=0))).tolist(),
        component_linf=np.atleast_1d(np.max(absolute, axis=0)).tolist())


def build_reference(case):
    if case['problem'] != 'shu_osher':
        reference = analytic_reference(case)
        label = 'exact Riemann solution' if case['problem'] == 'sod' else 'exact pre-shock characteristic solution' if case['problem'] == 'burgers' else 'exact translated periodic profile'
        return reference, dict(kind='analytic', definition=label, sampling=case['sampling'],
                               trustworthy_for_defined_samples=True), {}
    ref_case = reference_case(case, case['reference_cells'])
    final, diagnostic, history, timing = execute(ref_case, make_reconstruction(None, 'teno5'))
    if not diagnostic['valid_for_accuracy_claim']:
        raise FloatingPointError(f"Reference {case['id']} is inadmissible: {diagnostic}")
    ref_x = grid(ref_case)
    target = grid(case)
    reference = np.column_stack([np.interp(target, ref_x, final[:, k]) for k in range(3)])
    metadata = dict(kind='numerical', scheme='TENO5 JAX', case=ref_case,
                    time_grid=timing, diagnostics=diagnostic,
                    interpolation='componentwise np.interp at historical point samples',
                    convergence_status='uncertified; consult the separate reference study',
                    trustworthy_for_win_claim=False)
    return reference, metadata, dict(reference_grid=ref_x, reference_state=final,
                                      **{'reference_history_' + k: v for k, v in history.items()})


def compare_methods(records, reference_trustworthy):
    baseline = next(r for r in records if r['method'] == 'teno5')
    comparisons = []
    for row in records:
        if row['method'] == 'teno5':
            continue
        a, b = baseline['primary_error'], row['primary_error']
        valid = baseline['valid_for_accuracy_claim'] and row['valid_for_accuracy_claim']
        status = 'invalid' if not valid else 'reference-unresolved' if not reference_trustworthy else 'tie' if np.isclose(a, b, rtol=1e-10, atol=1e-14) else 'lower-error' if b < a else 'higher-error'
        comparisons.append(dict(method=row['method'], teno5_error=a, method_error=b,
                                ratio=b / a if np.isfinite(a) and a > 0 else None,
                                status=status, trustworthy_reference=reference_trustworthy))
    return comparisons


def plot_case(case, x, reference, finals, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    euler = reference.ndim == 2
    components = 3 if euler else 1
    figure, axes = plt.subplots(components, 2, figsize=(11, 3.2 * components), squeeze=False)
    labels = ('density', 'momentum', 'energy') if euler else ('u',)
    for k, label in enumerate(labels):
        target = reference[:, k] if euler else reference
        axes[k, 0].plot(x, target, 'k:', label='shared reference')
        for method, final in finals.items():
            value = final[:, k] if euler else final
            axes[k, 0].plot(x, value, label=method, linewidth=1)
            axes[k, 1].plot(x, value - target, label=method, linewidth=1)
        axes[k, 0].set_ylabel(label)
        axes[k, 1].set_ylabel(label + ' error')
        for axis in axes[k]:
            axis.set_xlabel('x')
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(case['id'])
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    figure.savefig(path.with_suffix('.pdf'))
    plt.close(figure)
