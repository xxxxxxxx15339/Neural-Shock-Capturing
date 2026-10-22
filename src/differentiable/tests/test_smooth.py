import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
import pytest

from src.differentiable.benchmark_cases import read_manifest
from src.differentiable.model_io import load_checkpoint
from src.differentiable.reconstruction import make_reconstruction
from src.differentiable.smooth_study import spatial_error


@pytest.mark.parametrize('system', ['advection', 'euler'])
def test_classical_smooth_spatial_order(system):
    reconstruct = make_reconstruction(None, 'teno5')
    coarse = spatial_error(64, system, reconstruct)['l1']
    fine = spatial_error(128, system, reconstruct)['l1']
    assert np.log2(coarse/fine) > 4.7


@pytest.mark.parametrize('system', ['advection', 'euler'])
def test_final_blend_converges_on_cell_average_smooth_data(system):
    info = read_manifest()['models']['euler' if system == 'euler' else 'scalar']
    reconstruct = make_reconstruction(load_checkpoint(info['path'], info['sha256']), 'learned-blend')
    coarse = spatial_error(64, system, reconstruct)['l1']
    fine = spatial_error(128, system, reconstruct)['l1']
    # Neural fifth order has no proof. Verify convergence and export the
    # achieved orders in the independent study instead of assuming it.
    assert np.isfinite([coarse, fine]).all()
    assert fine < coarse
