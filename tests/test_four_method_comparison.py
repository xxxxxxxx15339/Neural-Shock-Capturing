"""Independent parity and integrity checks for the evaluation-only adapters."""
import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from experiments.four_method_comparison.scripts import methods as m
from src.differentiable.numerics_jax import lax_friedrichs_flux,ssprk3_step
from src.differentiable.euler_jax import get_euler_flux_jax

class ComparisonTests(unittest.TestCase):
    def test_lax_contact_is_distinct_from_stronger_nearby_shock(self):
        from experiments.four_method_comparison.scripts.run_matched_benchmarks import exact_module
        from experiments.four_method_comparison.scripts.calculate_metrics import euler_wave_metrics
        ex=exact_module().ExactEuler([.445,.698,3.528],[.5,0.,.571],.5)
        n=4096;x=np.arange(n)/n;end=.1
        metrics=euler_wave_metrics(x,ex(x,end)[:,0],ex,.5,end,1.)
        self.assertLess(abs(metrics['contact_position']-(.5+ex.vel*end)),.5/n)
        self.assertLess(abs(metrics['shock_position']-(.5+ex.R[-1]*end)),.5/n)
        self.assertGreater(metrics['shock_position']-metrics['contact_position'],.08)

    def test_entropy_reference_before_and_after_shock(self):
        from experiments.four_method_comparison.scripts.run_matched_benchmarks import sine_entropy
        x=(np.arange(4096)+.5)/4096
        for t in (.2,.5,.9):
            u=sine_entropy(x,t)
            np.testing.assert_allclose(u,.5+.25*np.sin(2*np.pi*(x-t*u)),atol=8e-15)
            # Symmetric moving-frame quadrature pairs cancel exactly. Arbitrary
            # fixed point samples of a discontinuity do not integrate mass exactly.
            moving_x=.5*t+(np.arange(4096)+.5)/4096
            self.assertAlmostEqual(float(np.mean(sine_entropy(moving_x,t))),.5,places=13)
        center=.5+.5*.9
        left,right=sine_entropy(np.array([center-1e-9,center+1e-9]),.9)
        self.assertGreater(left,right)
        self.assertAlmostEqual(float((.5*left**2-.5*right**2)/(left-right)),.5,places=12)

    def test_exact_sod_reference_agrees_with_existing_solver(self):
        from experiments.four_method_comparison.scripts.run_matched_benchmarks import exact_module
        from src.analysis.euler_regression import exact_sod_solution
        x=(np.arange(127)+.37)/127
        exact=exact_module().ExactEuler([1.,0.,1.],[.125,0.,.1],.5)(x,.2)
        np.testing.assert_allclose(exact,exact_sod_solution(x,.2),atol=2e-9,rtol=2e-9)

    def test_original_keras_prediction_and_mask_parity(self):
        with np.load(m.OUT/'results/raw/original_keras_export.npz') as z:
            p=np.asarray(m.original_probabilities(jnp.asarray(z['raw'])))
            np.testing.assert_allclose(p,z['probabilities'],atol=1e-6,rtol=2e-5)
            np.testing.assert_array_equal(p<np.asarray(m.THRESHOLDS),z['probabilities']<np.asarray(m.THRESHOLDS))

    def test_author_neno_network_and_polynomial_parity(self):
        with np.load(m.OUT/'results/raw/neno6_author_export.npz') as z:
            s=jnp.asarray(z['raw'])
            np.testing.assert_allclose(m.neno_probabilities(s),z['probabilities'],atol=3e-14,rtol=3e-13)
            np.testing.assert_allclose(m.neno_reconstruct(s),z['flux'],atol=3e-14,rtol=3e-13)

    def test_author_neno_advection_trajectory_parity(self):
        with np.load(m.OUT/'results/raw/neno6_author_export.npz') as z:
            step=m.checked_step('neno6','scalar','periodic','advection')
            u=jnp.asarray(z['smoke_initial'])
            np.testing.assert_array_equal(m.gather(u,(-2,-1,0,1,2,3),'periodic'),z['left_stencils'])
            np.testing.assert_array_equal(m.gather(u,(3,2,1,0,-1,-2),'periodic'),z['right_stencils'])
            for _ in range(8):
                u,d=step(u,jnp.asarray(z['smoke_dt']),jnp.asarray(z['smoke_dx']))
                self.assertTrue(d['success'])
            np.testing.assert_allclose(u,z['smoke_final'],rtol=2e-13,atol=3e-14)

    def test_scalar_existing_flux_and_ssprk_parity(self):
        x=jnp.arange(24)/24;u=.5+.15*jnp.sin(2*jnp.pi*x)
        for method in ('teno5','nn_teno5','learned_head'):
            rec=m.reconstruction(method,'scalar')
            old=lambda v:lax_friedrichs_flux(v,lambda a:.5*a*a,1.,rec)
            np.testing.assert_allclose(m.scalar_flux(u,method,'burgers')[0],old(u),atol=3e-15)
            actual,d=m.checked_step(method,'scalar','periodic','burgers')(u,jnp.array(.001),jnp.array(1/24))
            np.testing.assert_allclose(actual,ssprk3_step(u,old,.001,1/24),atol=3e-15)
            self.assertTrue(d['success'])

    def test_euler_existing_flux_and_ssprk_parity(self):
        x=jnp.arange(24)/24;rho=1+.1*jnp.sin(2*jnp.pi*x)
        u=jnp.stack((rho,.4*rho,2.5+.08*rho),axis=-1)
        for method in ('teno5','nn_teno5','learned_head'):
            rec=m.reconstruction(method,'euler');old=get_euler_flux_jax(rec,'periodic')
            new=m.make_flux(method,'euler','periodic','euler')
            np.testing.assert_array_equal(new(u)[0],old(u))
            actual,d=m.checked_step(method,'euler','periodic','euler')(u,jnp.array(.001),jnp.array(1/24))
            np.testing.assert_allclose(actual,ssprk3_step(u,old,.001,1/24),atol=4e-15)
            self.assertTrue(d['success'])

    def test_all_constants_and_periodic_conservation(self):
        for method in m.MAIN:
            for bc in ('periodic','transmissive'):
                u=jnp.tile(jnp.array([1.,.3,2.545]),(24,1))
                v,d=m.checked_step(method,'euler',bc,'euler')(u,jnp.array(.001),jnp.array(1/24))
                self.assertTrue(d['success']);np.testing.assert_allclose(v,u,atol=3e-15)
            x=jnp.arange(24)/24;rho=1+.1*jnp.sin(2*jnp.pi*x)
            u=jnp.stack((rho,.4*rho,2.5+.08*rho),axis=-1)
            v,d=m.checked_step(method,'euler','periodic','euler')(u,jnp.array(.001),jnp.array(1/24))
            np.testing.assert_allclose(jnp.sum(v-u,axis=0)/24+d['boundary_exchange'],0.,atol=2e-15)

    def test_reflecting_wall_mass_energy_and_budget(self):
        u=jnp.tile(jnp.array([1.,.2,2.52]),(24,1))
        for method in m.MAIN:
            net,(left,right)=m.make_flux(method,'euler','reflective-wall','euler')(u)
            np.testing.assert_array_equal(np.asarray(left)[[0,2]],0.)
            np.testing.assert_array_equal(np.asarray(right)[[0,2]],0.)
            np.testing.assert_allclose(jnp.sum(net,axis=0),right-left,atol=3e-15)

    def test_head_architecture_bounds_and_gradients(self):
        for physics in ('euler','scalar'):
            s=jnp.array([[.1,.2,.9,.3,.15],[.7,.7,.7,.7,.7]])
            f=m.lambda_function(physics);lam=f(s)
            self.assertTrue(jnp.all((lam>=0)&(lam<=1)))
            self.assertTrue(jnp.all(jnp.isfinite(jax.grad(lambda a:jnp.sum(f(a)))(s))))
            np.testing.assert_array_equal(f(s),f(s))
            c=m.load_checkpoint(m.HEADS[physics])
            self.assertEqual(c.schema,'rich' if physics=='euler' else 'linear')

    def test_zero_blending_and_same_backbone_contract(self):
        s=jnp.array([[.2,.3,.9,.8,.1],[1.,1.,1.,1.,1.]])
        for physics in ('euler','scalar'):
            c=m.load_checkpoint(m.HEADS[physics])
            zero=m.make_reconstruction(c,'fixed-blend',0.)
            np.testing.assert_array_equal(zero(s),m.teno5_reconstruct_jax(s))
            pure=m.reconstruction('head_backbone_only',physics)(s)
            fixed=m.reconstruction('head_fixed_blend',physics)(s)
            np.testing.assert_allclose(fixed,(1-m.INITIAL_FIXED_BLEND)*m.teno5_reconstruct_jax(s)+m.INITIAL_FIXED_BLEND*pure,atol=0)

    def test_rejection_reports_raw_invalid_stage_without_repair(self):
        u=jnp.tile(jnp.array([1.,0.,2.5]),(24,1)).at[12:,:].set(jnp.array([.125,0.,.25]))
        v,d=m.checked_step('teno5','euler','transmissive','euler')(u,jnp.array(1.),jnp.array(1/24))
        self.assertFalse(d['success']);self.assertEqual(int(d['first_invalid_stage']),1)
        np.testing.assert_array_equal(v,u)
        self.assertFalse(m.admissible(d['invalid_candidate'],'euler'))
        np.testing.assert_array_equal(d['last_admissible_stage'],u)

if __name__=='__main__':unittest.main()
