"""Scientific contracts for explicit V2 policies and solver-input extraction."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from src.differentiable.artifacts import create_output
from src.differentiable.diagnostics import diagnostic_rollout
from src.differentiable.euler_jax import (euler_split_flux_stencils, get_euler_flux_jax,
                                         project_to_conservative_jax)
from src.differentiable.gated_teno5_jax import gated_reconstruction_diagnostics
from src.differentiable.model_io import REPO_ROOT, load_checkpoint
from src.differentiable.reconstruction import make_reconstruction


@pytest.fixture(scope='module', params=['scalar', 'euler'])
def model(request):
    return load_checkpoint(f'models/v2/{request.param}_gated_dataset_v2_seed2026.pkl')


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_zero_strength_exact_flux_and_rollout(model, mode):
    rec = make_reconstruction(model, 'gated-'+mode, 0.)
    base = make_reconstruction(None, 'teno5')
    stencils = jnp.asarray(np.random.default_rng(17).normal(size=(32, 3, 5)))
    np.testing.assert_array_equal(rec(stencils), base(stencils))
    rho = np.where(np.arange(24) < 12, 1., .3)
    initial = np.column_stack((rho, .2*rho, 2.5+.02*rho))
    a, da, _ = diagnostic_rollout(initial, rec, .001, 1/24, 4, 'euler', 'periodic')
    b, db, _ = diagnostic_rollout(initial, base, .001, 1/24, 4, 'euler', 'periodic')
    np.testing.assert_array_equal(a, b)
    assert da['valid_for_accuracy_claim'] and db['valid_for_accuracy_claim']


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_diagnostics_preserve_reconstruction(model, mode):
    s = jnp.asarray(np.vstack((np.tile(2., (3,5)), np.random.default_rng(5).normal(size=(20,5)))))
    info = gated_reconstruction_diagnostics(model.backbone, s, mode, .25)
    actual = make_reconstruction(model, 'gated-'+mode, .25)(s)
    np.testing.assert_allclose(info['base']+info['correction'], actual, rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(info['correction'])[~np.asarray(info['gate'])], 0.)
    assert np.all(np.asarray(info['normalized_weights']) >= 0)


@pytest.mark.parametrize('boundary', ['periodic', 'transmissive'])
def test_split_flux_extraction_is_actual_solver_input(boundary):
    x = (np.arange(24)+.5)/24
    rho = 1.+.1*np.sin(2*np.pi*x)
    u = jnp.asarray(np.column_stack((rho,.4*rho,2.5+.08*rho)))
    pos, neg, basis = euler_split_flux_stencils(u, boundary, True)
    rec = make_reconstruction(None, 'teno5')
    interfaces = project_to_conservative_jax(rec(pos)+rec(neg),basis)
    expected = interfaces-jnp.roll(interfaces,1,axis=0)
    if boundary == 'transmissive':
        from src.differentiable.euler_jax import flux_jax
        expected = expected.at[0].set(interfaces[0]-flux_jax(u[:1])[0])
    np.testing.assert_allclose(get_euler_flux_jax(rec,boundary)(u), expected, rtol=0, atol=1e-14)


def test_frozen_outputs_cannot_be_extended():
    with pytest.raises(ValueError, match='Frozen P0'):
        create_output(REPO_ROOT/'src/differentiable/validation/new-unwanted-study')


@pytest.mark.parametrize('physics,family', [('scalar','square'),('scalar','burgers_top_hat'),('euler','sod_like')])
def test_unified_prediction_matches_v2_training_evaluator(physics, family):
    from scripts.unrolled_physics_loss.Train_Gated_NNTENO5_Dataset_V2 import prediction_fn
    from src.differentiable.v2_evaluate import load_config, load_validation, make_predictor
    config=load_config(REPO_ROOT/'configs/v2/smoke.json')
    bucket=load_validation(config,physics)[0]
    j=np.flatnonzero(bucket['family']==family)[0]
    model=load_checkpoint(config['models'][physics]['path'])
    strength=.75 if physics=='scalar' else .25
    args=(bucket['initial'][j:j+1],bucket['dt'][j:j+1],bucket['speed'][j:j+1])
    expected=prediction_fn(physics,64,family,strength,True)(model.backbone,*args)
    system='euler' if physics=='euler' else 'burgers' if family.startswith('burgers') else 'advection'
    actual,_=make_predictor(model,64,system,'soft',1)(*args,jnp.array(strength))
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-13)


def test_reflective_wall_mass_and_energy_exchange():
    u=jnp.asarray(np.tile([1.,0.,2.5],(24,1)))
    net,(left,right)=get_euler_flux_jax(make_reconstruction(None,'teno5'),'reflective',True)(u)
    np.testing.assert_allclose(net,0,atol=2e-14)
    np.testing.assert_allclose(np.asarray(left)[[0,2]],0,atol=2e-14)
    np.testing.assert_allclose(np.asarray(right)[[0,2]],0,atol=2e-14)


def test_experimental_finite_volume_state_flux_nonlinear_spatial_order():
    from src.differentiable.v2_nonlinear import nonlinear_spatial
    from src.differentiable.euler_jax import get_euler_state_flux_jax
    rec=make_reconstruction(None,'teno5')
    coarse=nonlinear_spatial(40,rec,get_euler_state_flux_jax)['l1']
    fine=nonlinear_spatial(80,rec,get_euler_state_flux_jax)['l1']
    assert np.log2(coarse/fine)>4.7


def test_soft_neural_closed_wall_has_zero_mass_energy_flux():
    model=load_checkpoint('models/v2/euler_gated_dataset_v2_seed2026.pkl')
    rng=np.random.default_rng(1);rho=1.+.2*rng.random(24);v=.5*rng.normal(size=24);p=1.+.2*rng.random(24)
    initial=np.column_stack((rho,rho*v,p/.4+.5*rho*v*v))
    rec=make_reconstruction(model,'gated-soft',.25)
    net,(left,right)=get_euler_flux_jax(rec,'reflective-wall',True)(jnp.asarray(initial))
    np.testing.assert_array_equal(np.asarray(left)[[0,2]],0.)
    np.testing.assert_array_equal(np.asarray(right)[[0,2]],0.)
    np.testing.assert_allclose(np.asarray(net).sum(axis=0),np.asarray(right-left),atol=1e-14)
    final,diag,_=diagnostic_rollout(initial,rec,.0001,1/24,3,'euler','reflective-wall')
    np.testing.assert_allclose(np.sum(final,axis=0)[[0,2]],np.sum(initial,axis=0)[[0,2]],atol=3e-14)


def test_experimental_state_flux_preserves_periodic_conservation_and_constants():
    from src.differentiable.euler_jax import get_euler_state_flux_jax
    from src.differentiable.v2_nonlinear import nonlinear_state
    rec=make_reconstruction(None,'teno5')
    flux=get_euler_state_flux_jax(rec,'periodic')
    initial=jnp.asarray(nonlinear_state((np.arange(40)+.5)/40))
    np.testing.assert_allclose(np.sum(np.asarray(flux(initial)),axis=0),0,atol=2e-14)
    np.testing.assert_allclose(flux(jnp.tile(jnp.array([1.,.4,2.58]),(40,1))),0,atol=2e-14)
