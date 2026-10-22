import copy
import json

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from src.differentiable.artifacts import create_output, json_safe
from src.differentiable.benchmark_cases import analytic_reference, initial_state, read_manifest, time_grid, validate_case
from src.differentiable.diagnostics import diagnostic_rollout
from src.differentiable.euler_jax import euler_positivity_repair, get_euler_flux_jax
from src.differentiable.evaluation import compare_methods
from src.differentiable.model_io import load_checkpoint
from src.differentiable.numerics_jax import lax_friedrichs_flux, rollout
from src.differentiable.reconstruction import make_reconstruction


@pytest.fixture(scope='module')
def manifest():
    return read_manifest()


def test_canonical_time_grids_are_preserved(manifest):
    assert [time_grid(c)['steps'] for c in manifest['suites']['canonical']] == [128, 64, 93, 321]


def test_burgers_reference_and_pre_shock_validation(manifest):
    case = copy.deepcopy(manifest['suites']['canonical'][1])
    case['final_time'] = 1e-12
    np.testing.assert_allclose(analytic_reference(case), initial_state(case), atol=2e-12)
    case['final_time'] = 2 / np.pi
    with pytest.raises(ValueError, match='pre-shock'):
        analytic_reference(case)
    case = copy.deepcopy(manifest['suites']['canonical'][1])
    case['initial']['modes'][0][1] = 1.25
    with pytest.raises(ValueError, match='integer'):
        validate_case(case)


@pytest.mark.parametrize('system', ['advection', 'burgers', 'euler'])
@pytest.mark.parametrize('method', ['teno5', 'neural-only', 'fixed-blend', 'learned-blend'])
def test_final_models_conserve_periodic_states(manifest, system, method):
    info = manifest['models']['euler' if system == 'euler' else 'scalar']
    model = load_checkpoint(info['path'], info['sha256'])
    reconstruction = make_reconstruction(model, method, manifest['fixed_lambda'])
    x = np.arange(32) / 32
    if system == 'euler':
        rho = 1 + .1 * np.sin(2 * np.pi * x)
        initial = np.column_stack((rho, .7 * rho, 1 / .4 + .5 * .7**2 * rho))
    else:
        initial = .5 + .1 * np.sin(2 * np.pi * x)
    final, summary, _ = diagnostic_rollout(initial, reconstruction, .001, 1/32, 4, system)
    assert summary['valid_for_accuracy_claim']
    assert summary['repaired_stage_rows'] == 0
    np.testing.assert_allclose(np.sum(final, axis=0), np.sum(initial, axis=0), atol=5e-13)
    assert np.max(np.abs(summary['conservation_residual'])) < 2e-14


@pytest.mark.parametrize('system', ['advection', 'euler'])
def test_diagnostics_match_existing_rollout_and_boundary_budget(manifest, system):
    model = load_checkpoint(**dict(path=manifest['models']['euler' if system == 'euler' else 'scalar']['path']))
    reconstruction = make_reconstruction(model, 'learned-blend')
    x = np.arange(32) / 32
    if system == 'euler':
        rho = 1 + .1 * np.sin(2 * np.pi * x)
        initial = np.column_stack((rho, .7*rho, 2.5 + .5*.7**2*rho))
        flux = get_euler_flux_jax(reconstruction, boundary='transmissive')
        boundary = 'transmissive'
    else:
        initial = .5 + .1*np.sin(2*np.pi*x)
        flux = lambda u: lax_friedrichs_flux(u, lambda v: v, 1., reconstruction)
        boundary = 'periodic'
    expected = np.asarray(rollout(jnp.asarray(initial), flux, .001, 1/32, 5))
    final, summary, _ = diagnostic_rollout(initial, reconstruction, .001, 1/32, 5, system, boundary)
    np.testing.assert_allclose(final, expected, atol=1e-13, rtol=0)
    assert np.max(np.abs(summary['conservation_residual'])) < 1e-13
    if system == 'euler':
        assert np.max(np.abs(summary['integrated_boundary_exchange'])) > 1e-5


def test_uniform_low_state_repair_is_stable_conservative_and_accurate():
    initial = np.tile([.01, .002, .025 + .5 * .01 * .2**2], (16, 1))
    final, summary, _ = diagnostic_rollout(initial, make_reconstruction(None, 'teno5'), .0001, 1/16, 10,
                                         'euler', boundary='periodic', repair_policy='historical-repair')
    np.testing.assert_allclose(final, initial, atol=1e-17, rtol=0)
    assert summary['valid_for_accuracy_claim']
    assert summary['repaired_stage_rows'] == 0
    np.testing.assert_array_equal(summary['integrated_repair_exchange'], [0., 0., 0.])


def test_repair_counts_and_nonconservation_are_explicit():
    initial = np.tile([1., 0., 2.5], (7, 1))
    initial[3] = [-1., 0., -2.5]
    final, info = euler_positivity_repair(jnp.asarray(initial), return_diagnostics=True)
    assert np.sum(info['changed_mask']) == 1
    assert np.sum(info['isolated_mask']) == 1
    assert np.sum(info['fallback_mask']) == 0
    np.testing.assert_array_equal(np.sum(np.asarray(final)-initial, axis=0), [2., 0., 5.])
    initial[2:5] = [-1., 0., -2.5]
    final, info = euler_positivity_repair(jnp.asarray(initial), return_diagnostics=True)
    assert np.sum(info['fallback_mask']) == 3
    assert np.all(np.asarray(final)[:, 0] > 0)


def test_ssprk_weighted_repair_budget_is_accounted_for():
    initial = np.tile([1., 0., 2.5], (8, 1))

    def manufactured_repair(value):
        # A known artificial increment tests the RK bookkeeping, not a
        # proposed physical repair or a modification of the production path.
        after = value.at[:,0].add(1e-5)
        mask = jnp.ones(value.shape[0], dtype=bool)
        return after, dict(changed_mask=mask, isolated_mask=~mask, fallback_mask=mask)

    final, summary, _ = diagnostic_rollout(initial, make_reconstruction(None,'teno5'), .001, 1/8, 2,
                     'euler', repair_policy='historical-repair', repair_function=manufactured_repair)
    assert summary['repaired_stage_rows'] == 2*3*8
    assert not summary['valid_for_accuracy_claim']
    np.testing.assert_allclose(summary['integrated_repair_exchange'], [2*(1/6+2/3+1)*1e-5,0,0],atol=1e-15)
    np.testing.assert_allclose(summary['conservation_residual'],0,atol=1e-14)


def test_valid_repair_gradient_is_finite():
    initial = jnp.tile(jnp.array([.01, 0., .025]), (5, 1))
    gradient = jax.grad(lambda u: jnp.sum(euler_positivity_repair(u)))(initial)
    np.testing.assert_array_equal(np.asarray(gradient), np.ones((5, 3)))


def test_invalid_raw_stages_are_not_promoted_to_success():
    initial = np.tile([-1., 0., -2.5], (8, 1))
    _, summary, _ = diagnostic_rollout(initial, make_reconstruction(None, 'teno5'), .001, 1/8, 1, 'euler',
                                       repair_policy='historical-repair')
    assert not summary['valid_for_accuracy_claim']
    assert summary['invalid_stage_rows'] > 0
    assert summary['repaired_stage_rows'] > 0


def test_existing_outputs_are_protected_and_json_is_strict(tmp_path):
    create_output(tmp_path/'run')
    with pytest.raises(FileExistsError):
        create_output(tmp_path/'run')
    assert json.dumps(json_safe({'nan':np.nan, 'inf':np.inf}), allow_nan=False) == '{"nan": null, "inf": null}'
    assert compare_methods([dict(method='teno5', primary_error=1., valid_for_accuracy_claim=True),
                            dict(method='learned-blend', primary_error=1., valid_for_accuracy_claim=True)], True)[0]['status'] == 'tie'
