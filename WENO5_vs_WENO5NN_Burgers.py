# -*- coding: utf-8 -*-
"""Compare WENO5-JS and WENO5-NN on the inviscid Burgers' equation.

Matches Stevens & Colonius Sec. 4.2 / Fig. 10: a Gaussian initial condition
u(0, x) = exp(-k(x - L/2)^2) with k=20 on a periodic domain of length L=2,
run to T=4. The paper's reference is a fine WENO5-JS run at dx=3.125e-4,
dt=1.5625e-4 (6400 cells, 25,601 timesteps). The scalar `Simulation` class
uses a fixed uniform time grid rather than the Euler solver's adaptive
stepping, so a 6400-cell run takes on the order of 10+ minutes; the default
reference resolution here is reduced to 1600 cells for tractable runtime.
Pass `--reference-cells 6400` for the paper's exact reference. WENO5-JS and
WENO5-NN are then compared against that reference at several coarser grid
spacings.
"""

import argparse
import os

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

from src.config import BOUNDARY_CONDITION, MAX_CFL, MODEL_PATH, USE_SCALING
from src.core.Equations import invBurg
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.initial_conditions.InitialConditions import gaussian
from src.schemes import NNMethod, NNMethod_noScale, WENO5


L = 2.0
T = 4.0
K = 20.0
CFL = 0.5
ALPHA = 1.0  # max|f'(u)| = max|u| for this IC, since 0 <= u <= 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run the WENO5-JS/WENO5-NN Burgers comparison, matching '
            'Stevens & Colonius Sec. 4.2.'
        )
    )
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument(
        '--cells', type=int, nargs='+',
        default=[50, 100, 200, 400],
        help='Grid sizes to compare against the fine reference.',
    )
    parser.add_argument(
        '--reference-cells', type=int, default=1600,
        help=(
            'Cells for the fine WENO5-JS reference. The paper uses 6400 '
            '(dx=3.125e-4); the default here is reduced for runtime.'
        ),
    )
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument(
        '--save-dir', default='plots/burgers',
        help=(
            'Directory to save the figure as a PNG. Pass an empty string '
            "('') to disable saving."
        ),
    )
    return parser.parse_args()


def run_burgers(cells, fvm):
    dx = L/cells
    nt = int(np.ceil(T/(CFL*dx)))+1
    IC = gaussian(K, L/2)
    FS = LaxFriedrichs(invBurg(), ALPHA)
    if fvm.boundary != BOUNDARY_CONDITION or FS is None:
        raise ValueError('Configured boundary does not match the selected method.')
    x = np.linspace(0, L, cells, endpoint=False)
    simulation = Simulation(cells, nt, L, T, SSPRK3(), FS, fvm, IC, max_cfl=MAX_CFL)
    result = simulation.run()
    return x, result[:, -1]


def l2_error(x_coarse, u_coarse, x_fine, u_fine):
    reference = np.interp(x_coarse, x_fine, u_fine, period=L)
    return float(np.sqrt(np.mean((u_coarse-reference)**2)))


def main():
    args = parse_args()

    model = load_model(args.model_path)
    fvm_nn = NNMethod(model) if USE_SCALING else NNMethod_noScale(model)

    print('Computing fine WENO5-JS reference ({} cells)...'.format(args.reference_cells))
    x_fine, u_fine = run_burgers(args.reference_cells, WENO5())

    errors_weno = []
    errors_nn = []
    dxs = []
    coarsest = None

    for cells in sorted(args.cells):
        x_weno, u_weno = run_burgers(cells, WENO5())
        x_nn, u_nn = run_burgers(cells, fvm_nn)

        if coarsest is None:
            coarsest = (cells, x_weno, u_weno, x_nn, u_nn)

        e_weno = l2_error(x_weno, u_weno, x_fine, u_fine)
        e_nn = l2_error(x_nn, u_nn, x_fine, u_fine)

        dxs.append(L/cells)
        errors_weno.append(e_weno)
        errors_nn.append(e_nn)

        print('cells={:5d}  dx={:.6e}  E_WENO5-JS={:.6e}  E_WENO-NN={:.6e}'.format(
            cells, L/cells, e_weno, e_nn))

    cells_coarse, x_weno_coarse, u_weno_coarse, x_nn_coarse, u_nn_coarse = coarsest
    plt.figure()
    plt.plot(x_fine, u_fine, 'k:', linewidth=2, label='Reference ({} cells)'.format(args.reference_cells))
    plt.plot(x_weno_coarse, u_weno_coarse, label='WENO5-JS')
    plt.plot(x_nn_coarse, u_nn_coarse, '--', label='WENO-NN')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Solution at t={:g}, {} cells'.format(T, cells_coarse))
    plt.legend()

    plt.figure()
    plt.loglog(dxs, errors_weno, 'o-', label='WENO5-JS')
    plt.loglog(dxs, errors_nn, 's-', label='WENO-NN')
    plt.xlabel('$\\Delta x$')
    plt.ylabel('$E$')
    plt.title("Burgers' equation: error vs. grid spacing")
    plt.legend()
    plt.grid(True, which='both', alpha=0.25)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        figure_names = ('solution_profile', 'error_vs_grid_spacing')
        for num, name in zip(plt.get_fignums(), figure_names):
            path = os.path.join(args.save_dir, name + '.png')
            plt.figure(num).savefig(path, dpi=130)
            print('saved {}'.format(path))

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
