"""Numerical verification helpers."""

from .convergence import periodic_advection_convergence
from .euler_regression import (
    exact_riemann_solution, exact_sod_solution, interpolate_reference,
    regression_metrics, run_euler_benchmark,
)

__all__ = [
    'periodic_advection_convergence', 'exact_riemann_solution',
    'exact_sod_solution',
    'interpolate_reference', 'regression_metrics', 'run_euler_benchmark',
]
