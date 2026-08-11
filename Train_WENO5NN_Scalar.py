# -*- coding: utf-8 -*-
"""Train and evaluate a WENO3-NN, WENO5-NN, or WENO7-NN model."""

import argparse
import datetime
import os
import random

# Suppress TensorFlow INFO/WARNING device logs before TensorFlow is imported.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from keras import backend as be
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping

from src.config import (
    BOUNDARY_CONDITION,
    INPUT_DATA_PATH,
    MAX_CFL,
    MODEL_PATH,
    OUTPUT_DATA_PATH,
    RANDOM_SEED,
    USE_SCALING,
    WENO_ORDER,
)
from src.core.Equations import adv
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.data.preprocessing import scale_training_data, split_dataset
from src.initial_conditions.InitialConditions import step1
from src.networks.LoadDataMethods import loadInputData, loadOutputData
from src.networks.WenoNetworks import (
    WENO31stOrder,
    WENO51stOrder,
    WENO71stOrder,
)
from src.schemes import (
    NNMethod3,
    NNMethod3_noScale,
    NNMethod,
    NNMethod_noScale,
    NNMethod7,
    NNMethod7_noScale,
)
from src.viz.VisualizationFunctions import evalPerf


MODEL_BUILDERS = {
    3: WENO31stOrder,
    5: WENO51stOrder,
    7: WENO71stOrder,
}

SCALED_METHODS = {
    3: NNMethod3,
    5: NNMethod,
    7: NNMethod7,
}

UNSCALED_METHODS = {
    3: NNMethod3_noScale,
    5: NNMethod_noScale,
    7: NNMethod7_noScale,
}


class CompactTrainingLogger(Callback):
    """Print one stable, narrow line per epoch instead of a progress bar."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            '  Epoch {:03d} | loss {:.6f} | val_loss {:.6f}'.format(
                epoch + 1,
                logs.get('loss', float('nan')),
                logs.get('val_loss', float('nan')),
            ),
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a neural reconstruction for WENO3, WENO5, or WENO7.',
    )
    parser.add_argument('--input-data', default=INPUT_DATA_PATH)
    parser.add_argument('--output-data', default=OUTPUT_DATA_PATH)
    parser.add_argument('--model-path', default=MODEL_PATH)
    parser.add_argument('--regularization', type=float, default=0.15)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=150)
    parser.add_argument('--max-tv', type=float, default=2.05)
    parser.add_argument('--max-shock-width', type=int, default=30)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--max-attempts', type=int, default=10)
    parser.add_argument('--best-model-path', default=None)
    return parser.parse_args()


def select_stencil(raw_inputs, order):
    """Select a centered order-wide stencil from samples stored by columns."""
    available = raw_inputs.shape[0]
    if available < order:
        raise ValueError(
            'WENO{} requires {} input rows per sample, but {} contains only {}. '
            'Provide a dataset with at least {} centered stencil values using '
            '--input-data.'.format(order, order, 'the input dataset', available, order)
        )
    start = (available - order)//2
    return np.transpose(raw_inputs[start:start + order, :])


def make_problem():
    grid_factor = 4
    cfl = 2.0/3.0
    periods = 3
    nx = 25*grid_factor
    nt = int(25*periods*grid_factor/cfl) + 1
    length = 2
    final_time = 2*periods
    x = np.linspace(0, length, nx, endpoint=False)
    t = np.linspace(0, final_time, nt)

    x_grid, t_grid = np.meshgrid(x, t)
    exact = np.greater((x_grid - t_grid) % length, np.ones_like(x_grid))
    return x, t, periods, length, final_time, nx, nt, exact


def build_compiled_model(order, regularization, learning_rate):
    model = MODEL_BUILDERS[order](regularization)
    optimizer = optimizers.adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def save_model(model, path):
    model.save(path)
    print('Saved: {}'.format(path), flush=True)


def main():
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError('--max-attempts must be at least one.')
    best_model_path = args.best_model_path or args.model_path+'.best.h5'
    tf.logging.set_verbosity(tf.logging.ERROR)
    if WENO_ORDER not in MODEL_BUILDERS:
        raise ValueError('WENO_ORDER must be 3, 5, or 7; got {}.'.format(WENO_ORDER))
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    raw_inputs = loadInputData(args.input_data, stencil_size=None)
    inputs = select_stencil(raw_inputs, WENO_ORDER)
    targets = loadOutputData(args.output_data)
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            'Input/output sample mismatch: {} inputs and {} targets.'.format(
                inputs.shape[0], targets.shape[0]
            )
        )

    if USE_SCALING:
        inputs, targets = scale_training_data(inputs, targets)

    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(inputs, targets)

    x, t, periods, length, final_time, nx, nt, exact = make_problem()
    flux_splitting = LaxFriedrichs(adv(), 1)
    initial_condition = step1()
    time_stepper = SSPRK3()
    method_builder = SCALED_METHODS[WENO_ORDER] if USE_SCALING else UNSCALED_METHODS[WENO_ORDER]
    print(
        'Training WENO{}-NN with train/validation/test sizes {}/{}/{} '
        '(scaling={}).'.format(
            WENO_ORDER,
            x_train.shape[0],
            x_val.shape[0],
            x_test.shape[0],
            USE_SCALING,
        ),
        flush=True,
    )
    best = None
    for attempt in range(1, args.max_attempts+1):
        started = datetime.datetime.now()
        print('\n=== Attempt {} ==='.format(attempt), flush=True)

        attempt_seed = args.seed + attempt - 1
        random.seed(attempt_seed)
        np.random.seed(attempt_seed)
        tf.set_random_seed(attempt_seed)
        model = build_compiled_model(
            WENO_ORDER,
            args.regularization,
            args.learning_rate,
        )
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=1e-5,
            restore_best_weights=True,
        )
        compact_logger = CompactTrainingLogger()
        history = model.fit(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop, compact_logger],
            validation_data=(x_val, y_val),
            verbose=0,
        )

        finite_volume_method = method_builder(model)
        if finite_volume_method.boundary != BOUNDARY_CONDITION:
            raise ValueError(
                'Configured boundary {} does not match method boundary {}.'.format(
                    BOUNDARY_CONDITION, finite_volume_method.boundary
                )
            )
        simulation = Simulation(
            nx,
            nt,
            length,
            final_time,
            time_stepper,
            flux_splitting,
            finite_volume_method,
            initial_condition,
            max_cfl=MAX_CFL,
        )
        solution = simulation.run()
        max_tv, max_shock_width = evalPerf(
            x, t, periods, solution, exact, verbose=False
        )
        print(
            '  Result    | max_tv {:.6f} | shock_width {}'.format(
                max_tv, max_shock_width
            ),
            flush=True,
        )
        print('  Duration  | {}'.format(datetime.datetime.now() - started), flush=True)

        validation_loss = min(history.history['val_loss'])
        # Lexicographic score: numerical acceptance violations first, then
        # validation loss. This prevents a low ML loss hiding a poor solution.
        score = (
            max(0.0, max_tv-args.max_tv),
            max(0, max_shock_width-args.max_shock_width),
            validation_loss,
        )
        if best is None or score < best['score']:
            model.save(best_model_path)
            best = {
                'attempt': attempt, 'score': score, 'max_tv': max_tv,
                'shock_width': max_shock_width,
                'validation_loss': validation_loss,
            }
            print('  Best checkpoint: {}'.format(best_model_path), flush=True)

        if max_tv < args.max_tv and max_shock_width <= args.max_shock_width:
            save_model(model, args.model_path)
            test_loss = model.evaluate(x_test, y_test, verbose=0)
            print('Untouched test loss: {}'.format(test_loss))
            return
        be.clear_session()

    print('No candidate met the acceptance limits after {} attempts.'.format(
        args.max_attempts
    ))
    print('Best unsuccessful candidate: {}'.format(best))
    print('Best checkpoint saved at: {}'.format(best_model_path))
    raise SystemExit(1)

if __name__ == '__main__':
    main()
