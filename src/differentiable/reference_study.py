"""Cauchy/reference-sensitivity studies for historical and held-out Shu–Osher cases."""
import math

import numpy as np

from .artifacts import write_json
from .benchmark_cases import grid, reference_case
from .evaluation import errors, execute
from .reconstruction import make_reconstruction


def _restrict(x, state, target):
    return np.column_stack([np.interp(target, x, state[:, k]) for k in range(3)])


def run_reference_study(case, checkpoint, methods, fixed_lambda, settings, output):
    output.mkdir()
    target = grid(case)
    coarse, coarse_valid = {}, {}
    for method in methods:
        final, diagnostic, history, timing = execute(case, make_reconstruction(checkpoint, method, fixed_lambda))
        coarse[method] = final
        coarse_valid[method] = diagnostic['valid_for_accuracy_claim']
        write_json(output / f'{method}-coarse.json', dict(diagnostics=diagnostic, time_grid=timing))
        np.savez_compressed(output / f'{method}-coarse.npz', x=target, state=final)

    spatial, temporal, arrays, fine_results = [], [], {}, {}
    previous, previous_x, previous_difference = None, None, None
    for cells in settings['resolutions']:
        fine_case = reference_case(case, cells)
        fine, diagnostic, history, timing = execute(fine_case, make_reconstruction(None, 'teno5'))
        if not diagnostic['valid_for_accuracy_claim']:
            raise FloatingPointError(f"Inadmissible {cells}-cell reference for {case['id']}")
        x = grid(fine_case)
        projected = _restrict(x, fine, target)
        fine_results[cells] = (fine, projected, diagnostic, timing)
        np.savez_compressed(output / f'reference-{cells}-dt1.npz', x=x, state=fine,
                            **{'history_' + k: v for k, v in history.items()})
        row = dict(cells=cells, time_grid=timing, diagnostics=diagnostic,
                   errors={method: errors(final, projected) for method, final in coarse.items()})
        if previous is not None:
            difference = errors(projected, previous)
            previous_grid_difference = errors(_restrict(x, fine, previous_x), arrays['previous_fine'])
            region = (target >= settings['regional_window'][0]) & (target <= settings['regional_window'][1])
            row.update(target_grid_cauchy=difference, previous_grid_cauchy=previous_grid_difference,
                       regional_target_cauchy=errors(projected[region], previous[region]),
                       observed_target_cauchy_order=None if previous_difference is None or difference['l1'] == 0 else math.log(previous_difference / difference['l1'], 2))
            previous_difference = difference['l1']
        previous, previous_x, arrays['previous_fine'] = projected, x, fine
        spatial.append(row)
        print(f"  {case['id']} reference N={cells}: learned/TENO5={row['errors']['learned-blend']['primary_error']/row['errors']['teno5']['primary_error']:.7f}", flush=True)

    cells = settings['resolutions'][-1]
    finest_case = reference_case(case, cells)
    previous = None
    temporal_projected = {}
    for refinement in settings['temporal_refinements']:
        if refinement == 1:
            fine, projected, diagnostic, timing = fine_results[cells]
        else:
            fine, diagnostic, history, timing = execute(finest_case, make_reconstruction(None, 'teno5'), refinement=refinement)
            projected = _restrict(grid(finest_case), fine, target)
            np.savez_compressed(output / f'reference-{cells}-dt{refinement}.npz', x=grid(finest_case), state=fine,
                                **{'history_' + k: v for k, v in history.items()})
        if not diagnostic['valid_for_accuracy_claim']:
            raise FloatingPointError('Inadmissible temporal reference')
        row = dict(refinement=refinement, time_grid=timing, diagnostics=diagnostic,
                   errors={method: errors(final, projected) for method, final in coarse.items()})
        if previous is not None:
            row['target_grid_cauchy'] = errors(projected, previous)
            row['full_grid_cauchy'] = errors(fine, previous_fine)
        temporal.append(row)
        temporal_projected[refinement] = projected
        previous, previous_fine = projected, fine
        print(f"  {case['id']} temporal dt/{refinement}: learned/TENO5={row['errors']['learned-blend']['primary_error']/row['errors']['teno5']['primary_error']:.7f}", flush=True)

    spatial_uncertainty = spatial[-1]['target_grid_cauchy']['l1']
    # Include the full standard-dt -> finest-dt shift, not just the last increment.
    temporal_uncertainty = errors(temporal_projected[settings['temporal_refinements'][-1]], temporal_projected[1])['l1']
    uncertainty = spatial_uncertainty + temporal_uncertainty
    final_errors = temporal[-1]['errors']
    findings = {}
    for method in methods:
        if method == 'teno5':
            continue
        margin = final_errors['teno5']['primary_error'] - final_errors[method]['primary_error']
        recent_margins = [row['errors']['teno5']['primary_error'] - row['errors'][method]['primary_error'] for row in spatial[-3:] + temporal]
        consistent = all(m * margin > 0 for m in recent_margins)
        resolved = (coarse_valid['teno5'] and coarse_valid[method] and consistent
                    and 2 * uncertainty < settings['reference_uncertainty_fraction'] * abs(margin))
        findings[method] = dict(
            final_margin=margin, ratio=final_errors[method]['primary_error'] / final_errors['teno5']['primary_error'],
            recent_ranking_consistent=consistent, empirical_margin_uncertainty_bound=2 * uncertainty,
            uncertainty_fraction_limit=settings['reference_uncertainty_fraction'],
            status='reliable-lower-error' if resolved and margin > 0 else 'reliable-higher-error' if resolved else 'reference-unresolved',
            caveat='Empirical Cauchy criterion, not a rigorous error bound or an independent-scheme reference')
    temporal[-1]['observed_target_cauchy_order'] = (
        math.log(temporal[-2]['target_grid_cauchy']['l1'] / temporal[-1]['target_grid_cauchy']['l1'], 2)
        if temporal[-2]['target_grid_cauchy']['l1'] > 0 and temporal[-1]['target_grid_cauchy']['l1'] > 0 else None)
    result = dict(case=case, settings=settings, coarse_valid=coarse_valid,
                  interpolation='componentwise linear point interpolation',
                  spatial=spatial, temporal=temporal, spatial_target_uncertainty=spatial_uncertainty,
                  temporal_target_uncertainty=temporal_uncertainty, findings=findings)
    write_json(output / 'reference_convergence.json', result)
    _plot(case['id'], spatial, temporal, output)
    return result


def _plot(name, spatial, temporal, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rows = spatial[1:]
    axes[0].loglog([r['cells'] for r in rows], [r['target_grid_cauchy']['l1'] for r in rows], 'o-', label='target grid')
    axes[0].loglog([r['cells'] for r in rows], [r['previous_grid_cauchy']['l1'] for r in rows], 's-', label='previous fine grid')
    axes[0].set(xlabel='reference cells', ylabel='Cauchy mean state L1')
    for method in spatial[0]['errors']:
        if method != 'teno5':
            axes[1].semilogx([r['cells'] for r in spatial], [r['errors'][method]['primary_error']/r['errors']['teno5']['primary_error'] for r in spatial], 'o-', label=method)
    axes[1].axhline(1, color='black', linestyle=':')
    axes[1].set(xlabel='reference cells', ylabel='method/TENO5 error')
    axes[2].loglog([r['refinement'] for r in temporal[1:]], [r['target_grid_cauchy']['l1'] for r in temporal[1:]], 'o-')
    axes[2].set(xlabel='dt refinement factor', ylabel='temporal target Cauchy L1')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(output / 'reference_convergence.png', dpi=160)
    fig.savefig(output / 'reference_convergence.pdf')
    plt.close(fig)
