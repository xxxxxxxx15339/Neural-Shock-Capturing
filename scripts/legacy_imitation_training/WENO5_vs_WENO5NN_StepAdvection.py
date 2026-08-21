# -*- coding: utf-8 -*-
"""Run the basic WENO-NN advection experiment."""

import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import os

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from src.config import BOUNDARY_CONDITION, MAX_CFL, MODEL_PATH, USE_SCALING
from src.core.Equations import adv
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.initial_conditions.InitialConditions import step1
from src.schemes import NNMethod, NNMethod_noScale, WENO5
from src.viz.VisualizationFunctions import (
    discTrackStep,
    intError,
    plotDiscWidth,
    totalVariation,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the WENO5-NN comparison.')
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument(
        '--save-dir', default='plots/step_advection',
        help=(
            'Directory to save each figure as a PNG. Pass an empty string '
            "('') to disable saving."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Match Stevens & Colonius Sec. 4.1 / Fig. 3: L=2, c=1, 100 cells,
    # CFL=2/3, run for 50 periods (T=100).
    CFL = 2/3
    P = 50
    nx = 100
    L = 2
    T = 2*P
    dx = L/nx
    nt = int(np.ceil(T/(CFL*dx)))+1
    x = np.linspace(0,L,nx,endpoint=False)
    t = np.linspace(0,T,nt)
    dx = x[1]-x[0]
    dt = t[1]-t[0]

    model = load_model(args.model_path)

    FVM1 = NNMethod(model) if USE_SCALING else NNMethod_noScale(model)
    FVM2 = WENO5()
    if FVM1.boundary != BOUNDARY_CONDITION or FVM2.boundary != BOUNDARY_CONDITION:
        raise ValueError('Configured boundary does not match the selected methods.')
    EQ = adv()
    FS = LaxFriedrichs(EQ, 1)
    IC = step1()
    RK = SSPRK3()

    testSim = Simulation(nx, nt, L, T, RK, FS, FVM1, IC, max_cfl=MAX_CFL)
    WENO5Sim = Simulation(nx, nt, L, T, RK, FS, FVM2, IC, max_cfl=MAX_CFL)
    uv = testSim.run()
    uv_WENO5 = WENO5Sim.run()

    plotDiscWidth(x,t,P,uv,uv_WENO5)

    delt = 0.125
    discTrackStep(1,x,t,uv_WENO5,T/2,'WENO5', -1-delt,-1+delt,True)
    discTrackStep(1,x,t,uv,T/2,'Neural Network', -1-delt,-1+delt,True)
    discTrackStep(1,x,t,uv_WENO5,T/2,'WENO5', -2-delt,-2+delt,True)
    discTrackStep(1,x,t,uv,T/2,'Neural Network', -2-delt,-2+delt,True)

    intError(1,x,t,uv,'Neural Network')
    intError(1,x,t,uv_WENO5,'WENO5')
    totalVariation(t,uv_WENO5,'WENO5')
    totalVariation(t,uv,'Neural Network')

    plt.figure()
    # T is an exact multiple of the advection period (T = 2*P, wave speed 1,
    # domain length L=2), so the exact solution at t=T equals the initial
    # condition exactly.
    plt.plot(x,IC(x),'k:',linewidth=2,label='Exact')
    plt.plot(x,uv_WENO5[:,-1],label='WENO5')
    plt.plot(x,uv[:,-1],'--',label='Neural Network')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Final state at t={:g}'.format(T))
    plt.legend()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        figure_names = (
            'discontinuity_width_vs_time',
            'disc_track_weno5_x-1',
            'disc_track_nn_x-1',
            'disc_track_weno5_x-2',
            'disc_track_nn_x-2',
            'l2_error_nn',
            'l2_error_weno5',
            'total_variation_weno5',
            'total_variation_nn',
            'final_snapshot',
        )
        for num, name in zip(plt.get_fignums(), figure_names):
            path = os.path.join(args.save_dir, name + '.png')
            plt.figure(num).savefig(path, dpi=130)
            print('saved {}'.format(path))

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')

    print("Simulation finished! Here is the final array:")
    print(uv)


if __name__ == '__main__':
    main()
