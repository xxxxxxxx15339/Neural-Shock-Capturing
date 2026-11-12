"""Independent contracts for the opt-in FV and conservative safeguard paths."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from src.differentiable.accuracy_upgrade import (
    Admissibility, admissible, flux_variant, state_faces, state_stencils,
    low_faces, pressure, safeguarded_step, retry_step,
)
from src.differentiable.euler_jax import get_euler_flux_jax, get_euler_state_flux_jax
from src.differentiable.teno5_jax import teno5_reconstruct_jax as rec
from src.differentiable.v2_nonlinear import nonlinear_state, nonlinear_spatial
from src.differentiable.gated_teno5_jax import gated_nn_teno5_reconstruct
from src.differentiable.teno5_jax_toy import init_toy_params


def smooth(n=40):
    z, w = np.polynomial.legendre.leggauss(16)
    return jnp.asarray(.5*np.einsum('j,ijk->ik', w,
        nonlinear_state((np.arange(n)[:, None]+.5+.5*z)/n)))


@pytest.mark.parametrize('boundary', ['periodic', 'transmissive', 'reflective-wall'])
def test_shared_flux_budget(boundary):
    u = smooth()
    faces, _ = state_faces(u, rec, boundary, True)
    if boundary == 'periodic':
        np.testing.assert_array_equal(faces[0], faces[-1])
    if boundary == 'reflective-wall':
        np.testing.assert_array_equal(np.asarray(faces)[[0, -1]][:, [0, 2]], 0.)
    result, d = jax.jit(lambda v: safeguarded_step(v, .001, 1/len(v), rec, boundary))(u)
    assert d['success']
    assert jnp.all(admissible(result))
    np.testing.assert_allclose((result-u).sum(axis=0)/len(u)+d['exchange'], 0, atol=2e-15)


@pytest.mark.parametrize('boundary', ['periodic', 'transmissive'])
def test_constant_and_baseline_identity(boundary):
    u = jnp.tile(jnp.array([1., .2, 2.52]), (24, 1))
    np.testing.assert_allclose(flux_variant(rec, boundary, 'fv_state_flux')(u), 0, atol=2e-15)
    np.testing.assert_array_equal(flux_variant(rec, boundary)(u), get_euler_flux_jax(rec, boundary)(u))
    params = init_toy_params(jax.random.PRNGKey(9))
    zero = lambda s: gated_nn_teno5_reconstruct(params, s, strength=0.)
    np.testing.assert_array_equal(flux_variant(rec, boundary, 'fv_state_flux')(smooth()),
                                 flux_variant(zero, boundary, 'fv_state_flux')(smooth()))


def test_prototype_parity_and_nonlinear_order():
    u = smooth()
    np.testing.assert_allclose(flux_variant(rec, 'periodic', 'fv_state_flux')(u),
                               get_euler_state_flux_jax(rec)(u), atol=1e-14, rtol=1e-12)
    f = lambda r, b: flux_variant(r, b, 'fv_state_flux')
    e = [nonlinear_spatial(n, rec, f)['l1'] for n in (40, 80, 160)]
    assert np.log2(e[-2]/e[-1]) > 4.8


def test_canonical_stencil_order_matches_inference():
    # Identity basis projection/back-projection is checked indirectly against
    # the independent existing prototype and exact cell-boundary polynomials.
    u = smooth(24)
    plus, minus = state_stencils(u, 'periodic')
    captured = []
    def observe(s):
        captured.append(np.asarray(s))
        return rec(s)
    state_faces(u, observe, 'periodic')
    assert plus.shape == minus.shape == (25, 3, 5)
    np.testing.assert_array_equal(captured[0], plus)
    np.testing.assert_array_equal(captured[1], minus)


def test_retry_and_explicit_rejection():
    u = smooth(16)
    cfg = Admissibility(max_retries=8)
    v, d = jax.jit(lambda u: retry_step(u, 1., 1/16, rec, config=cfg))(u)
    assert d['success'] and d['retries'] > 0 and d['accepted_dt'] < 1.
    np.testing.assert_allclose(jnp.sum(v-u, axis=0)/16+d['exchange'], 0, atol=1e-14)
    v, d = retry_step(u, 1., 1/16, rec, config=Admissibility(max_retries=0))
    assert not d['success'] and d['accepted_dt'] == 0
    np.testing.assert_array_equal(v, u)
    invalid = u.at[0, 0].set(-1.)
    v, d = jax.jit(lambda x: retry_step(x, .001, 1/16, rec))(invalid)
    assert not d['success']
    np.testing.assert_array_equal(v, invalid)


def test_low_order_cfl_and_stress_face_scaling():
    rho = jnp.where(jnp.arange(32) < 16, 1., .01)
    p = jnp.where(jnp.arange(32) < 16, 1000., .01)
    u = jnp.stack((rho, jnp.zeros(32), p/.4), axis=-1)
    low, alpha = low_faces(u)
    dt = .8/(32*alpha)
    assert jnp.all(admissible(u-dt*32*jnp.diff(low, axis=0)))
    # Deliberately nonphysical reconstruction must be diagnosed and yield
    # physical flux evaluation, with no modification to input averages.
    bad = lambda s: 10.*rec(s)-9.*s[..., 0]
    _, d = state_faces(u, bad, face_guard=True)
    assert not d['raw_faces_valid'] and d['face_theta'] < 1
    # The initial low-order CFL is valid, but intermediate sound speeds rise.
    # The whole SSP step must therefore use the documented retry policy.
    result, d = retry_step(u, dt, 1/32, bad)
    assert d['success'] and jnp.all(admissible(result))
    np.testing.assert_allclose((result-u).sum(axis=0)/32+d['exchange'], 0, atol=1e-12)


def test_finite_gradient_in_inactive_region():
    # Vary reconstruction smoothly; bisection coefficients have frozen gradients.
    u = smooth(16)
    def loss(a):
        r = lambda s: rec(s)+a*(s[..., 3]-s[..., 2])
        final, d = safeguarded_step(u, .001, 1/16, r, 'periodic')
        return jnp.mean(final**2)
    derivative = jax.jit(jax.grad(loss))(jnp.array(.01))
    assert jnp.isfinite(derivative)
    h = 1e-5
    np.testing.assert_allclose(derivative, (loss(.01+h)-loss(.01-h))/(2*h), atol=2e-9, rtol=1e-3)


def test_active_flux_limiter_preserves_periodic_balance():
    rng=np.random.default_rng(2)
    rho=np.exp(rng.uniform(-2,2,16));v=rng.uniform(-3,3,16)
    p=np.exp(rng.uniform(-3,3,16))
    u=jnp.asarray(np.stack((rho,rho*v,p/.4+.5*rho*v*v),axis=-1))
    from src.differentiable.euler_jax import spds_jax
    dt=.4/(16*jnp.max(jnp.abs(spds_jax(u))))
    result,d=jax.jit(lambda x:safeguarded_step(x,dt,1/16,rec,'periodic'))(u)
    assert d['success'] and jnp.min(d['stages']['theta'])<.5
    assert jnp.all(admissible(result))
    np.testing.assert_allclose(jnp.sum(result-u,axis=0)/16,0,atol=2e-15)


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        Admissibility(max_low_cfl=1.1)
    with pytest.raises(ValueError):
        flux_variant(rec, variant='unknown')


def test_state_faces_agree_with_numpy_reference():
    from src.core.eulerEquations import roe_eigenbasis,flux,spds
    from src.schemes.teno5 import teno5_reconstruct
    u=np.asarray(smooth(24));padded=np.pad(u,((3,3),(0,0)),mode='edge')
    left,right=roe_eigenbasis(padded)
    index=np.arange(25)[:,None]+2;offset=np.arange(-2,3)[None,:]
    plus=np.einsum('nij,njk->nik',left[2:27],padded[index+offset].transpose(0,2,1))
    minus=np.einsum('nij,njk->nik',left[2:27],padded[index+1-offset].transpose(0,2,1))
    ul=np.einsum('nij,nj->ni',right[2:27],teno5_reconstruct(plus))
    ur=np.einsum('nij,nj->ni',right[2:27],teno5_reconstruct(minus))
    alpha=np.maximum(np.max(abs(spds(ul)),axis=1),np.max(abs(spds(ur)),axis=1))
    expected=.5*(flux(ul)+flux(ur)-alpha[:,None]*(ur-ul))
    actual,_=state_faces(jnp.asarray(u),rec)
    np.testing.assert_allclose(actual,expected,atol=5e-15,rtol=5e-14)


def test_benchmark_wrapper_matches_original_ssprk_stages():
    from experiments.accuracy_upgrade.study import step_kernel,reconstruction,formulation
    from src.differentiable.numerics_jax import ssprk3_step
    u=smooth(24);dt=.001;dx=1/24
    for variant in ['historical_teno','historical_v2_soft25','fv_teno']:
        actual,d=step_kernel(variant,'transmissive')(u,jnp.array(dt),jnp.array(dx))
        f=flux_variant(reconstruction(variant),'transmissive',formulation(variant))
        expected=jax.jit(lambda u:ssprk3_step(u,f,dt,dx))(u)
        assert d['success']
        np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=2e-15)
