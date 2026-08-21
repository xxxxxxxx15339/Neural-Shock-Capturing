"""Evaluate one WENO5-NN model on the canonical step-advection benchmark."""

import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from keras.models import load_model

from Train_WENO5NN_Scalar import make_problem
from src.config import MAX_CFL, MODEL_PATH, USE_SCALING, WENO_ORDER
from src.core.Equations import adv
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.initial_conditions.InitialConditions import step1
from src.schemes import NNMethod, NNMethod_noScale
from src.viz.VisualizationFunctions import evalPerf


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a WENO5-NN model without opening plots.'
    )
    parser.add_argument('--model-path', default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.logging.set_verbosity(tf.logging.ERROR)
    if WENO_ORDER != 5:
        raise ValueError('The canonical benchmark currently supports WENO_ORDER=5.')

    x, t, periods, length, final_time, nx, nt, exact = make_problem()
    model = load_model(args.model_path)
    method = NNMethod(model) if USE_SCALING else NNMethod_noScale(model)
    simulation = Simulation(
        nx,
        nt,
        length,
        final_time,
        SSPRK3(),
        LaxFriedrichs(adv(), 1),
        method,
        step1(),
        max_cfl=MAX_CFL,
    )
    solution = simulation.run()
    max_tv, shock_width = evalPerf(
        x, t, periods, solution, exact, verbose=False
    )
    result = {
        'model_path': args.model_path,
        'weno_order': WENO_ORDER,
        'scaling': USE_SCALING,
        'nx': nx,
        'nt': nt,
        'final_time': final_time,
        'max_tv': float(max_tv),
        'shock_width': int(shock_width),
        'solution_min': float(np.min(solution)),
        'solution_max': float(np.max(solution)),
        'accepted': bool(max_tv < 2.016 and shock_width <= 20),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
