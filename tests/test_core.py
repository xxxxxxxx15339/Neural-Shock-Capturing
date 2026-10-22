import unittest

import numpy as np

from src.data.preprocessing import scale_training_data, split_dataset
from src.analysis import (
    exact_sod_solution, interpolate_reference, periodic_advection_convergence,
    regression_metrics, run_euler_benchmark,
)
from src.core.Equations import adv
from src.core.eulerEquations import getEulerFlux, roe_eigenbasis
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation, eulerSimulation
from src.core.TimeSteppingMethods import SSPRK3
from src.initial_conditions.InitialConditions import shuOsher, sod, step1
from src.schemes import (
    WENO3, WENO3euler, WENO5, WENO5euler, WENO7, WENO7euler, TENO5
)


class WENO5Tests(unittest.TestCase):
    def test_constant_reconstruction(self):
        values = np.full(32, 2.5)
        np.testing.assert_allclose(WENO5().evalF(values), values)

    def test_short_periodic_advection_is_finite(self):
        simulation = Simulation(
            40, 11, 2.0, 0.02, SSPRK3(), LaxFriedrichs(adv(), 1),
            WENO5(), step1(), max_cfl=1.0,
        )
        solution = simulation.run()
        self.assertTrue(np.isfinite(solution).all())

    def test_cfl_violation_is_rejected(self):
        simulation = Simulation(
            20, 2, 2.0, 1.0, SSPRK3(), LaxFriedrichs(adv(), 1),
            WENO5(), step1(), max_cfl=1.0,
        )
        with self.assertRaisesRegex(ValueError, 'CFL'):
            simulation.run()

    def test_unsupported_scalar_boundary_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            WENO5().__class__(5, lambda values: values, boundary='edge')


class DataPipelineTests(unittest.TestCase):
    def test_scaling_round_trip_and_constants(self):
        inputs = np.array([[2., 4., 6.], [3., 3., 3.]])
        targets = np.array([[5.], [3.]])
        scaled_inputs, scaled_targets = scale_training_data(inputs, targets)
        np.testing.assert_allclose(scaled_inputs[0], [0., 0.5, 1.])
        np.testing.assert_allclose(scaled_targets[0], [0.75])
        np.testing.assert_allclose(scaled_inputs[1], 0.)
        np.testing.assert_allclose(scaled_targets[1], 0.)

    def test_contiguous_train_validation_test_split(self):
        inputs = np.arange(100).reshape(20, 5)
        targets = np.arange(20).reshape(20, 1)
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(
            inputs, targets
        )
        self.assertEqual((len(x_train), len(x_val), len(x_test)), (14, 3, 3))
        self.assertLess(y_train[-1, 0], y_val[0, 0])
        self.assertLess(y_val[-1, 0], y_test[0, 0])


class SpatialConvergenceTests(unittest.TestCase):
    CASES = (
        (WENO3, (320, 640, 1280), 3.0),
        (WENO5, (80, 160, 320), 4.8),
        (WENO7, (160, 320, 640), 6.8),
        (TENO5, (80, 160, 320), 4.8),
    )

    def test_smooth_periodic_advection_orders(self):
        for scheme_builder, resolutions, minimum_order in self.CASES:
            with self.subTest(scheme=scheme_builder.__name__):
                rows = periodic_advection_convergence(
                    scheme_builder, resolutions
                )
                self.assertLess(rows[-1]['l1_error'], rows[-2]['l1_error'])
                self.assertGreater(rows[-1]['order'], minimum_order)


class EulerValidationTests(unittest.TestCase):
    EULER_SCHEMES = (WENO3euler, WENO5euler, WENO7euler)

    @staticmethod
    def constant_state(density=1.0, velocity=0.0, pressure=1.0):
        def initial_condition(x):
            energy = pressure/(1.4-1) + 0.5*density*velocity**2
            return np.tile(
                [density, density*velocity, energy],
                (len(x), 1),
            )
        return initial_condition

    def run_euler(self, scheme_builder, initial_condition, length=1.0):
        simulation = eulerSimulation(
            48,
            5,
            length,
            0.001,
            SSPRK3(),
            getEulerFlux(scheme_builder()),
            initial_condition,
            3,
        )
        solution = simulation.runEuler()
        self.assertTrue(np.isfinite(solution).all())
        self.assertTrue(np.all(solution[:,-1,0] > 0))
        density = solution[:,-1,0]
        velocity = solution[:,-1,1]/density
        pressure = (1.4-1)*(solution[:,-1,2]-0.5*density*velocity**2)
        self.assertTrue(np.all(pressure > 0))
        return solution

    def test_constant_state_all_weno_orders(self):
        for scheme_builder in self.EULER_SCHEMES:
            with self.subTest(scheme=scheme_builder.__name__):
                solution = self.run_euler(
                    scheme_builder, self.constant_state()
                )
                np.testing.assert_allclose(
                    solution[:,-1,:], solution[:,0,:], atol=1e-12
                )

    def test_uniform_moving_state_all_weno_orders(self):
        for scheme_builder in self.EULER_SCHEMES:
            with self.subTest(scheme=scheme_builder.__name__):
                solution = self.run_euler(
                    scheme_builder,
                    self.constant_state(velocity=0.75),
                )
                np.testing.assert_allclose(
                    solution[:,-1,:], solution[:,0,:], atol=1e-12
                )

    def test_sod_all_weno_orders(self):
        for scheme_builder in self.EULER_SCHEMES:
            with self.subTest(scheme=scheme_builder.__name__):
                self.run_euler(scheme_builder, sod())

    def test_shu_osher_all_weno_orders(self):
        for scheme_builder in self.EULER_SCHEMES:
            with self.subTest(scheme=scheme_builder.__name__):
                self.run_euler(scheme_builder, shuOsher(), length=10.0)

    def test_nonpositive_density_is_rejected(self):
        simulation = eulerSimulation(
            5, 2, 1.0, 0.01, SSPRK3(), lambda state: state,
            sod(), 3,
        )
        invalid = np.ones((5, 3))
        invalid[0, 0] = 0
        with self.assertRaisesRegex(FloatingPointError, 'density'):
            simulation._validate_state(invalid, 0)

    def test_roe_left_and_right_bases_are_inverses(self):
        state = sod()(np.arange(32, dtype=float)/32)
        left, right = roe_eigenbasis(state)
        identity = np.einsum('nij,njk->nik', left, right)
        np.testing.assert_allclose(
            identity, np.broadcast_to(np.eye(3), identity.shape), atol=1e-12
        )

    def test_adaptive_cfl_reaches_exact_final_time(self):
        from src.core.eulerEquations import spds
        simulation = eulerSimulation(
            40, 2, 1.0, 0.02, SSPRK3(), getEulerFlux(WENO5euler()),
            sod(), 3, max_cfl=0.35,
            wave_speed_function=lambda state: np.max(np.abs(spds(state))),
        )
        simulation.runEuler()
        self.assertAlmostEqual(simulation.last_times[-1], 0.02, places=14)
        self.assertLessEqual(simulation.max_observed_cfl, 0.35+1e-12)

    def test_invalid_intermediate_runge_kutta_stage_is_rejected(self):
        state = self.constant_state()(np.arange(8))
        with self.assertRaisesRegex(FloatingPointError, 'density'):
            SSPRK3().stepItEuler(
                state, lambda value: np.ones_like(value)*100,
                1.0, 1.0, 3,
                validator=lambda stage: eulerSimulation(
                    8, 2, 1, 1, SSPRK3(), lambda value: value,
                    self.constant_state(), 3,
                )._validate_state(stage, 0),
            )


class EulerPhysicalRegressionTests(unittest.TestCase):
    EULER_SCHEMES = (WENO3euler, WENO5euler, WENO7euler)

    def test_sod_at_t_point_2_against_exact_riemann_solution(self):
        for scheme_builder in self.EULER_SCHEMES:
            with self.subTest(scheme=scheme_builder.__name__):
                x, solution, initial = run_euler_benchmark(
                    scheme_builder, sod(), 120, 1.0, 0.2
                )
                metrics = regression_metrics(
                    x, solution, initial, 0.2,
                    exact_sod_solution(x, 0.2),
                )
                self.assertLess(metrics['density_l1'], 0.012)
                self.assertLess(metrics['momentum_l1'], 0.012)
                self.assertLess(metrics['energy_l1'], 0.025)
                self.assertGreater(metrics['minimum_density'], 0.12)
                self.assertGreater(metrics['minimum_pressure'], 0.095)
                self.assertLess(metrics['density_overshoot'], 2e-4)
                self.assertLess(metrics['pressure_overshoot'], 2e-4)
                self.assertLess(metrics['shock_location_error'], 0.02)
                self.assertLess(metrics['contact_location_error'], 0.03)
                self.assertLess(max(metrics['conservation_error']), 1e-8)

    def test_shu_osher_at_t_1_point_8_against_refined_weno7(self):
        reference_x, reference, _ = run_euler_benchmark(
            WENO7euler, shuOsher(), 320, 10.0, 1.8
        )
        for scheme_builder in self.EULER_SCHEMES:
            with self.subTest(scheme=scheme_builder.__name__):
                x, solution, initial = run_euler_benchmark(
                    scheme_builder, shuOsher(), 120, 10.0, 1.8
                )
                metrics = regression_metrics(
                    x, solution, initial, 1.8,
                    interpolate_reference(reference_x, reference, x),
                )
                self.assertLess(metrics['density_l1'], 0.16)
                self.assertLess(metrics['momentum_l1'], 0.25)
                self.assertLess(metrics['energy_l1'], 0.95)
                self.assertGreater(metrics['minimum_density'], 0.75)
                self.assertGreater(metrics['minimum_pressure'], 0.99)
                self.assertLess(metrics['density_overshoot'], 0.05)
                self.assertLess(metrics['pressure_overshoot'], 1e-3)
                self.assertLess(metrics['shock_location_error'], 0.15)
                self.assertLess(metrics['contact_location_error'], 0.5)
                self.assertLess(max(metrics['conservation_error']), 0.003)


if __name__ == '__main__':
    unittest.main()
