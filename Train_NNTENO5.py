# -*- coding: utf-8 -*-
"""Train an NN-TENO5 candidate-stencil classifier using the original data.

The original data contain five-point stencils and scalar flux targets.
NN-TENO5 is a classifier rather than a flux regressor, so the scalar flux
targets are used only to validate dataset alignment. The three classifier
labels are generated using deterministic TENO5 selection:

    0 = smooth candidate, keep it
    1 = troubled candidate, discard it
"""

import argparse
import datetime
import hashlib
import json
import os
import random

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.models import load_model

from src.networks.TenoNetworks import TENO5Network
from src.schemes.teno5 import (
    CANDIDATE_COUNT,
    DEFAULT_CUTOFF,
    DEFAULT_EPSILON,
    DEFAULT_POWER,
    DEFAULT_SCALE_CONSTANT,
    STENCIL_SIZE,
    teno5_troubled_labels,
)
from src.schemes.teno_common import normalize_teno_stencils


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train NN-TENO5 using the original five-point stencil dataset.'
        )
    )

    parser.add_argument(
        '--input-data',
        default='data/2ndNewAvgs.csv',
        help='Original five-point stencil CSV.',
    )

    parser.add_argument(
        '--original-flux-data',
        default='data/2ndNewFlux.csv',
        help=(
            'Original scalar flux-target CSV. It is used only to validate '
            'sample alignment because NN-TENO5 predicts three labels.'
        ),
    )

    parser.add_argument(
        '--labels-data',
        default=None,
        help=(
            'Precomputed three-candidate label CSV (shape samples x 3), '
            'e.g. an oracle-relabeled dataset from '
            'GenerateTrainingData_TENOExact.py. When given, these labels '
            'are used instead of generating deterministic TENO5 labels '
            'from --input-data.'
        ),
    )

    parser.add_argument(
        '--warm-start-path',
        default=None,
        help=(
            'Optional path to an existing model (.h5) to load weights '
            'from before training, instead of a fresh random init. '
            'Architecture must match exactly; load_weights ignores the '
            "donor file's optimizer state."
        ),
    )
    parser.add_argument(
        '--anchor-weight',
        type=float,
        default=0.0,
        help=(
            'L2-to-origin penalty coefficient for --warm-start-path '
            'fine-tuning: penalizes weights for moving away from the '
            'donor checkpoint values, so training toward a different '
            'label source (e.g. --labels-data) cannot drift arbitrarily '
            'far from a known-safe starting point. 0.0 disables it.'
        ),
    )

    parser.add_argument(
        '--model-path',
        default='models/nn_teno5.h5',
    )

    parser.add_argument(
        '--metadata-path',
        default=None,
    )

    parser.add_argument(
        '--history-path',
        default=None,
    )

    parser.add_argument(
        '--split-path',
        default=None,
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=175,
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=20,
    )

    parser.add_argument(
        '--lr-patience',
        type=int,
        default=8,
    )

    parser.add_argument(
        '--min-learning-rate',
        type=float,
        default=1.0e-7,
    )

    parser.add_argument(
        '--train-fraction',
        type=float,
        default=0.70,
    )

    parser.add_argument(
        '--validation-fraction',
        type=float,
        default=0.15,
    )

    parser.add_argument(
        '--split-mode',
        choices=('stratified', 'random'),
        default='stratified',
        help=(
            'Stratified mode preserves the distribution of the eight '
            'possible three-candidate label masks.'
        ),
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    parser.add_argument(
        '--device',
        choices=('auto', 'gpu', 'cpu'),
        default='auto',
        help=(
            'Training device. "gpu" fails instead of silently falling back '
            'to the CPU when CUDA is unavailable.'
        ),
    )

    parser.add_argument(
        '--gpu-index',
        type=int,
        default=0,
        help='Zero-based CUDA GPU index used by --device gpu/auto.',
    )

    # Deterministic TENO teacher parameters
    parser.add_argument(
        '--teno-cutoff',
        type=float,
        default=DEFAULT_CUTOFF,
    )

    parser.add_argument(
        '--teno-epsilon',
        type=float,
        default=DEFAULT_EPSILON,
    )

    parser.add_argument(
        '--teno-scale-constant',
        type=float,
        default=DEFAULT_SCALE_CONSTANT,
    )

    parser.add_argument(
        '--teno-power',
        type=float,
        default=DEFAULT_POWER,
    )

    # Threshold calibration
    parser.add_argument(
        '--threshold-min',
        type=float,
        default=0.10,
    )

    parser.add_argument(
        '--threshold-max',
        type=float,
        default=0.90,
    )

    parser.add_argument(
        '--threshold-step',
        type=float,
        default=0.01,
    )

    # Imbalance handling
    parser.add_argument(
        '--disable-sample-weights',
        action='store_true',
        help='Disable increased weighting of troubled-stencil samples.',
    )

    parser.add_argument(
        '--maximum-positive-weight',
        type=float,
        default=20.0,
    )

    return parser.parse_args()


def validate_args(args):
    if args.learning_rate <= 0.0:
        raise ValueError('--learning-rate must be positive.')

    if args.min_learning_rate <= 0.0:
        raise ValueError('--min-learning-rate must be positive.')

    if args.min_learning_rate > args.learning_rate:
        raise ValueError(
            '--min-learning-rate cannot exceed --learning-rate.'
        )

    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive.')

    if args.epochs < 1:
        raise ValueError('--epochs must be positive.')

    if args.patience < 1:
        raise ValueError('--patience must be positive.')

    if args.lr_patience < 1:
        raise ValueError('--lr-patience must be positive.')

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError('--train-fraction must belong to (0, 1).')

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError(
            '--validation-fraction must belong to (0, 1).'
        )

    if args.train_fraction + args.validation_fraction >= 1.0:
        raise ValueError(
            'Training and validation fractions must sum to less than one.'
        )

    if not 0.0 <= args.teno_cutoff <= 1.0:
        raise ValueError('--teno-cutoff must belong to [0, 1].')

    if args.teno_epsilon <= 0.0:
        raise ValueError('--teno-epsilon must be positive.')

    if args.teno_scale_constant <= 0.0:
        raise ValueError('--teno-scale-constant must be positive.')

    if args.teno_power <= 0.0:
        raise ValueError('--teno-power must be positive.')

    if not 0.0 <= args.threshold_min <= 1.0:
        raise ValueError('--threshold-min must belong to [0, 1].')

    if not 0.0 <= args.threshold_max <= 1.0:
        raise ValueError('--threshold-max must belong to [0, 1].')

    if args.threshold_min >= args.threshold_max:
        raise ValueError(
            '--threshold-min must be smaller than --threshold-max.'
        )

    if args.threshold_step <= 0.0:
        raise ValueError('--threshold-step must be positive.')

    if args.maximum_positive_weight < 1.0:
        raise ValueError(
            '--maximum-positive-weight must be at least one.'
        )

    if args.gpu_index < 0:
        raise ValueError('--gpu-index cannot be negative.')


# ---------------------------------------------------------------------------
# Compatibility and reproducibility
# ---------------------------------------------------------------------------

def configure_tensorflow_logging():
    if hasattr(tf, 'logging'):
        tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        tf.get_logger().setLevel('ERROR')


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    if hasattr(tf, 'set_random_seed'):
        tf.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)


def configure_training_device(device, gpu_index):
    """Create the TF1/Keras session and report the device used for training."""

    if not hasattr(tf, 'ConfigProto') or not hasattr(tf, 'Session'):
        raise RuntimeError(
            'This trainer requires the TensorFlow 1.x session API.'
        )

    config = tf.ConfigProto()

    if device == 'cpu':
        config.device_count['GPU'] = 0
    else:
        config.gpu_options.visible_device_list = str(gpu_index)
        # Avoid reserving all VRAM on small and display-attached GPUs.
        config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    K.set_session(session)

    try:
        local_devices = session.list_devices()
    except AttributeError:
        from tensorflow.python.client import device_lib
        local_devices = device_lib.list_local_devices(
            session_config=config
        )

    gpu_devices = [
        found.name
        for found in local_devices
        if found.device_type == 'GPU'
    ]

    if device == 'gpu' and not gpu_devices:
        session.close()
        raise RuntimeError(
            'GPU training was requested, but TensorFlow found no GPU. '
            'Run nvidia-smi and verify that the NVIDIA device nodes, '
            'CUDA 10.0, and cuDNN 7 are available.'
        )

    selected_device = (
        'gpu'
        if gpu_devices
        else 'cpu'
    )

    print(
        'Training device: {}{}'.format(
            selected_device.upper(),
            (
                ' ({})'.format(', '.join(gpu_devices))
                if gpu_devices
                else ''
            ),
        ),
        flush=True,
    )

    return {
        'requested': device,
        'selected': selected_device,
        'gpu_index': (
            gpu_index
            if selected_device == 'gpu'
            else None
        ),
        'tensorflow_devices': [
            {
                'name': found.name,
                'type': found.device_type,
            }
            for found in local_devices
        ],
    }


def predict_model(model, inputs):
    try:
        return model.predict(
            inputs,
            verbose=0,
        )
    except TypeError:
        return model.predict(inputs)


# ---------------------------------------------------------------------------
# Paths and metadata
# ---------------------------------------------------------------------------

def ensure_parent(path):
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(
            parent,
            exist_ok=True,
        )


def resolve_output_paths(args):
    model_base = os.path.splitext(
        args.model_path
    )[0]

    metadata_path = (
        args.metadata_path
        if args.metadata_path
        else model_base + '_metadata.json'
    )

    history_path = (
        args.history_path
        if args.history_path
        else model_base + '_history.csv'
    )

    split_path = (
        args.split_path
        if args.split_path
        else model_base + '_split.npz'
    )

    for path in (
        args.model_path,
        metadata_path,
        history_path,
        split_path,
    ):
        ensure_parent(path)

    return (
        metadata_path,
        history_path,
        split_path,
    )


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, 'rb') as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Original data loading
# ---------------------------------------------------------------------------

def load_original_stencils(path):
    """Load original data as a matrix with shape (samples, 5)."""

    if not os.path.isfile(path):
        raise IOError(
            'Input stencil file does not exist: {}'.format(path)
        )

    values = np.loadtxt(
        path,
        delimiter=',',
    )

    if values.ndim != 2:
        raise ValueError(
            'Original stencil data must be two-dimensional.'
        )

    if values.shape[0] == STENCIL_SIZE:
        values = values.T

    elif values.shape[1] == STENCIL_SIZE:
        values = values

    else:
        raise ValueError(
            'Expected original stencil shape (5, samples) or '
            '(samples, 5); received {}.'
            .format(values.shape)
        )

    if len(values) < 10:
        raise ValueError(
            'The original dataset is too small for training.'
        )

    if not np.isfinite(values).all():
        raise ValueError(
            'Original stencil data contain NaN or infinity.'
        )

    return values.astype(
        np.float64,
        copy=False,
    )


def validate_original_flux_file(path, sample_count):
    """Validate alignment with the original scalar target file."""

    if not path:
        return None

    if not os.path.isfile(path):
        raise IOError(
            'Original flux-target file does not exist: {}'.format(path)
        )

    targets = np.loadtxt(
        path,
        delimiter=',',
    )

    targets = np.asarray(
        targets,
    ).reshape(-1)

    if len(targets) != sample_count:
        raise ValueError(
            'Original stencil and flux-target sample counts differ: '
            '{} versus {}.'
            .format(
                sample_count,
                len(targets),
            )
        )

    if not np.isfinite(targets).all():
        raise ValueError(
            'Original flux targets contain NaN or infinity.'
        )

    return {
        'sample_count': int(len(targets)),
        'minimum': float(np.min(targets)),
        'maximum': float(np.max(targets)),
        'sha256': file_sha256(path),
    }


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _validate_label_array(labels, sample_count, source):
    labels = np.asarray(
        labels,
        dtype=np.float32,
    )

    expected_shape = (
        sample_count,
        CANDIDATE_COUNT,
    )

    if labels.shape != expected_shape:
        raise ValueError(
            '{} label shape {} does not match expected shape {}.'
            .format(
                source,
                labels.shape,
                expected_shape,
            )
        )

    if np.any(
        (labels != 0.0)
        & (labels != 1.0)
    ):
        raise ValueError(
            '{} labels must contain only zero and one.'.format(source)
        )

    if not np.any(labels == 1.0):
        raise ValueError(
            'The {} produced no troubled-candidate labels. '
            'Inspect the deterministic TENO cutoff and input data.'
            .format(source)
        )

    return labels


def generate_labels(stencils, args):
    """Generate three deterministic TENO5 troubled labels per stencil."""

    labels = teno5_troubled_labels(
        stencils,
        epsilon=args.teno_epsilon,
        scale_constant=args.teno_scale_constant,
        power=args.teno_power,
        cutoff=args.teno_cutoff,
    )

    return _validate_label_array(
        labels,
        len(stencils),
        'Generated TENO5',
    )


def load_labels(path, sample_count):
    """Load precomputed three-candidate labels from a CSV file."""

    if not os.path.isfile(path):
        raise IOError(
            'Labels file does not exist: {}'.format(path)
        )

    labels = np.loadtxt(
        path,
        delimiter=',',
    )

    return _validate_label_array(
        labels,
        sample_count,
        'Precomputed ({})'.format(path),
    )


def label_mask_codes(labels):
    """Encode each three-candidate mask as an integer from zero to seven."""

    bits = (
        2 ** np.arange(
            labels.shape[1],
            dtype=int,
        )
    )

    return np.sum(
        labels.astype(int) * bits,
        axis=1,
    )


def mask_string(code, candidate_count=CANDIDATE_COUNT):
    return ''.join(
        str((int(code) >> candidate) & 1)
        for candidate in range(candidate_count)
    )


def print_label_statistics(labels, title):
    print('\n{} label statistics:'.format(title))

    sample_count = len(labels)

    for candidate in range(labels.shape[1]):
        troubled = int(
            np.sum(labels[:, candidate] == 1)
        )

        print(
            '  Candidate {} troubled: {:7d} / {:7d} ({:7.3f}%)'
            .format(
                candidate,
                troubled,
                sample_count,
                100.0 * troubled / sample_count,
            )
        )

    codes = label_mask_codes(labels)

    print('  Full mask distribution:')

    for code in range(2 ** labels.shape[1]):
        count = int(
            np.sum(codes == code)
        )

        print(
            '    [{}]: {:7d} ({:7.3f}%)'.format(
                mask_string(code),
                count,
                100.0 * count / sample_count,
            )
        )


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def random_split_indices(
    sample_count,
    train_fraction,
    validation_fraction,
    seed,
):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(sample_count)

    train_end = int(
        train_fraction * sample_count
    )

    validation_end = train_end + int(
        validation_fraction * sample_count
    )

    if (
        train_end < 1
        or validation_end <= train_end
        or validation_end >= sample_count
    ):
        raise ValueError(
            'Dataset is too small for the requested split fractions.'
        )

    return (
        indices[:train_end],
        indices[train_end:validation_end],
        indices[validation_end:],
    )


def stratified_mask_split_indices(
    labels,
    train_fraction,
    validation_fraction,
    seed,
):
    """Split while approximately preserving each full label-mask class."""

    rng = np.random.RandomState(seed)
    codes = label_mask_codes(labels)

    train_parts = []
    validation_parts = []
    test_parts = []

    for code in np.unique(codes):
        indices = np.flatnonzero(
            codes == code
        )

        indices = rng.permutation(
            indices
        )

        count = len(indices)

        # Tiny mask classes cannot be divided safely. Keep them in training.
        if count < 3:
            train_parts.append(indices)
            continue

        train_count = max(
            1,
            int(np.floor(train_fraction * count)),
        )

        validation_count = max(
            1,
            int(np.floor(validation_fraction * count)),
        )

        if train_count + validation_count >= count:
            train_count = max(
                1,
                count - 2,
            )
            validation_count = 1

        train_parts.append(
            indices[:train_count]
        )

        validation_parts.append(
            indices[
                train_count:
                train_count + validation_count
            ]
        )

        test_parts.append(
            indices[
                train_count + validation_count:
            ]
        )

    train_indices = np.concatenate(
        train_parts
    ) if train_parts else np.empty(0, dtype=int)

    validation_indices = np.concatenate(
        validation_parts
    ) if validation_parts else np.empty(0, dtype=int)

    test_indices = np.concatenate(
        test_parts
    ) if test_parts else np.empty(0, dtype=int)

    # Fall back to a normal random split when rare-mask handling leaves an
    # empty validation or test subset.
    if (
        len(train_indices) == 0
        or len(validation_indices) == 0
        or len(test_indices) == 0
    ):
        return random_split_indices(
            len(labels),
            train_fraction,
            validation_fraction,
            seed,
        )

    train_indices = rng.permutation(
        train_indices
    )

    validation_indices = rng.permutation(
        validation_indices
    )

    test_indices = rng.permutation(
        test_indices
    )

    return (
        train_indices,
        validation_indices,
        test_indices,
    )


def verify_split(
    train_indices,
    validation_indices,
    test_indices,
    sample_count,
):
    train_set = set(
        train_indices.tolist()
    )
    validation_set = set(
        validation_indices.tolist()
    )
    test_set = set(
        test_indices.tolist()
    )

    if train_set & validation_set:
        raise AssertionError(
            'Training and validation indices overlap.'
        )

    if train_set & test_set:
        raise AssertionError(
            'Training and test indices overlap.'
        )

    if validation_set & test_set:
        raise AssertionError(
            'Validation and test indices overlap.'
        )

    combined = (
        train_set
        | validation_set
        | test_set
    )

    if len(combined) != sample_count:
        raise AssertionError(
            'Dataset split does not cover every sample exactly once.'
        )


# ---------------------------------------------------------------------------
# Class-imbalance weighting
# ---------------------------------------------------------------------------

def make_training_sample_weights(
    labels,
    maximum_positive_weight,
):
    """Increase the influence of samples containing rare troubled labels."""

    positive_count = np.sum(
        labels,
        axis=0,
    )

    negative_count = (
        len(labels)
        - positive_count
    )

    positive_weights = np.ones(
        labels.shape[1],
        dtype=float,
    )

    has_positives = positive_count > 0.0

    positive_weights[has_positives] = (
        negative_count[has_positives]
        / positive_count[has_positives]
    )

    positive_weights = np.clip(
        positive_weights,
        1.0,
        maximum_positive_weight,
    )

    candidate_weights = np.where(
        labels > 0.5,
        positive_weights,
        1.0,
    )

    sample_weights = np.mean(
        candidate_weights,
        axis=1,
    )

    mean_weight = float(
        np.mean(sample_weights)
    )

    if mean_weight > 0.0:
        sample_weights = (
            sample_weights
            / mean_weight
        )

    return (
        sample_weights.astype(np.float32),
        positive_weights,
    )


# ---------------------------------------------------------------------------
# Classifier metrics and threshold calibration
# ---------------------------------------------------------------------------

def classifier_metrics(
    labels,
    probabilities,
    thresholds,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    thresholds = np.asarray(
        thresholds,
        dtype=float,
    )

    predicted = (
        probabilities
        >= thresholds
    ).astype(int)

    true_positive = np.sum(
        (labels == 1) & (predicted == 1),
        axis=0,
    )

    false_positive = np.sum(
        (labels == 0) & (predicted == 1),
        axis=0,
    )

    true_negative = np.sum(
        (labels == 0) & (predicted == 0),
        axis=0,
    )

    false_negative = np.sum(
        (labels == 1) & (predicted == 0),
        axis=0,
    )

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(
            true_positive,
            dtype=float,
        ),
        where=(
            true_positive + false_positive
        ) > 0,
    )

    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(
            true_positive,
            dtype=float,
        ),
        where=(
            true_positive + false_negative
        ) > 0,
    )

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(
            precision,
            dtype=float,
        ),
        where=(
            precision + recall
        ) > 0,
    )

    false_smooth_rate = np.divide(
        false_negative,
        true_positive + false_negative,
        out=np.zeros_like(
            false_negative,
            dtype=float,
        ),
        where=(
            true_positive + false_negative
        ) > 0,
    )

    false_discard_rate = np.divide(
        false_positive,
        true_negative + false_positive,
        out=np.zeros_like(
            false_positive,
            dtype=float,
        ),
        where=(
            true_negative + false_positive
        ) > 0,
    )

    return {
        'thresholds': thresholds.tolist(),
        'binary_accuracy': float(
            np.mean(predicted == labels)
        ),
        'exact_mask_accuracy': float(
            np.mean(
                np.all(
                    predicted == labels,
                    axis=1,
                )
            )
        ),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'macro_f1': float(np.mean(f1)),
        'false_smooth_rate': false_smooth_rate.tolist(),
        'false_discard_rate': false_discard_rate.tolist(),
        'true_positive': true_positive.astype(int).tolist(),
        'false_positive': false_positive.astype(int).tolist(),
        'true_negative': true_negative.astype(int).tolist(),
        'false_negative': false_negative.astype(int).tolist(),
        'predicted_all_discard_fraction': float(
            np.mean(
                np.all(
                    predicted == 1,
                    axis=1,
                )
            )
        ),
    }


def calibrate_thresholds(
    labels,
    probabilities,
    minimum,
    maximum,
    step,
):
    """Calibrate one threshold per candidate using validation macro behavior."""

    threshold_grid = np.arange(
        minimum,
        maximum + 0.5 * step,
        step,
    )

    selected = np.full(
        labels.shape[1],
        0.5,
        dtype=float,
    )

    for candidate in range(labels.shape[1]):
        candidate_labels = labels[:, candidate:candidate + 1]
        candidate_probabilities = probabilities[
            :,
            candidate:candidate + 1
        ]

        best_key = None
        best_threshold = 0.5

        for threshold in threshold_grid:
            result = classifier_metrics(
                candidate_labels,
                candidate_probabilities,
                np.array([threshold]),
            )

            candidate_f1 = result['f1'][0]
            candidate_recall = result['recall'][0]
            false_smooth = result['false_smooth_rate'][0]
            false_discard = result['false_discard_rate'][0]

            # Prefer F1 first, then troubled recall, then lower false-smooth
            # and lower false-discard rates.
            key = (
                candidate_f1,
                candidate_recall,
                -false_smooth,
                -false_discard,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)

        selected[candidate] = best_threshold

    return selected


def print_classifier_metrics(title, metrics):
    print('\n{}:'.format(title))

    print(
        '  Binary accuracy    : {:.6f}'.format(
            metrics['binary_accuracy']
        )
    )

    print(
        '  Exact mask accuracy: {:.6f}'.format(
            metrics['exact_mask_accuracy']
        )
    )

    print(
        '  Macro F1           : {:.6f}'.format(
            metrics['macro_f1']
        )
    )

    for candidate in range(CANDIDATE_COUNT):
        print(
            '  Candidate {} | threshold {:.3f} | '
            'precision {:.6f} | recall {:.6f} | F1 {:.6f} | '
            'false-smooth {:.6f}'.format(
                candidate,
                metrics['thresholds'][candidate],
                metrics['precision'][candidate],
                metrics['recall'][candidate],
                metrics['f1'][candidate],
                metrics['false_smooth_rate'][candidate],
            )
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    configure_tensorflow_logging()
    K.clear_session()
    set_random_seeds(args.seed)
    runtime_device = configure_training_device(
        args.device,
        args.gpu_index,
    )

    (
        metadata_path,
        history_path,
        split_path,
    ) = resolve_output_paths(args)

    print(
        'Loading original five-point stencil data: {}'.format(
            args.input_data
        ),
        flush=True,
    )

    raw_stencils = load_original_stencils(
        args.input_data
    )

    original_flux_metadata = validate_original_flux_file(
        args.original_flux_data,
        len(raw_stencils),
    )

    print(
        'Loaded {} original stencils.'.format(
            len(raw_stencils)
        ),
        flush=True,
    )

    if args.labels_data:
        print(
            'Loading precomputed labels: {}'.format(args.labels_data),
            flush=True,
        )
        labels = load_labels(
            args.labels_data,
            len(raw_stencils),
        )
    else:
        print(
            'Generating deterministic TENO5 classifier labels...',
            flush=True,
        )
        labels = generate_labels(
            raw_stencils,
            args,
        )

    print_label_statistics(
        labels,
        'Complete dataset',
    )

    # NN-TENO normalization maps each nonconstant stencil to [0, 1].
    normalized_inputs = normalize_teno_stencils(
        raw_stencils,
    ).astype(np.float32)

    if args.split_mode == 'stratified':
        split = stratified_mask_split_indices(
            labels,
            args.train_fraction,
            args.validation_fraction,
            args.seed,
        )
    else:
        split = random_split_indices(
            len(labels),
            args.train_fraction,
            args.validation_fraction,
            args.seed,
        )

    (
        train_indices,
        validation_indices,
        test_indices,
    ) = split

    verify_split(
        train_indices,
        validation_indices,
        test_indices,
        len(labels),
    )

    x_train = normalized_inputs[
        train_indices
    ]
    y_train = labels[
        train_indices
    ]

    x_validation = normalized_inputs[
        validation_indices
    ]
    y_validation = labels[
        validation_indices
    ]

    x_test = normalized_inputs[
        test_indices
    ]
    y_test = labels[
        test_indices
    ]

    print(
        '\nDataset sizes: train={} validation={} test={}'.format(
            len(x_train),
            len(x_validation),
            len(x_test),
        ),
        flush=True,
    )

    print_label_statistics(
        y_train,
        'Training subset',
    )

    sample_weights = None
    positive_weights = np.ones(
        CANDIDATE_COUNT,
        dtype=float,
    )

    if not args.disable_sample_weights:
        (
            sample_weights,
            positive_weights,
        ) = make_training_sample_weights(
            y_train,
            args.maximum_positive_weight,
        )

        print(
            '\nTroubled-label positive weights: {}'.format(
                positive_weights.tolist()
            ),
            flush=True,
        )

    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    print(
        '\nBuilding TENO5 classifier...',
        flush=True,
    )

    model = TENO5Network(
        learning_rate=args.learning_rate
    )

    if args.warm_start_path:
        # Fine-tune from an existing checkpoint's weights instead of a
        # fresh random init.
        model.load_weights(args.warm_start_path)

        if args.anchor_weight > 0.0:
            # L2-SP / elastic-weight-consolidation style: penalize the
            # weights themselves for moving away from the warm-start
            # values, so training against a different label source
            # (e.g. oracle-relabeled data) can still improve on it
            # without drifting arbitrarily far from a starting point
            # already known to produce a stable, working scheme.
            origin_weights = [
                K.constant(value) for value in model.get_weights()
            ]
            anchor_terms = [
                K.sum(K.square(weight_tensor - origin_value))
                for weight_tensor, origin_value
                in zip(model.weights, origin_weights)
            ]
            model.add_loss(
                args.anchor_weight * K.sum(K.stack(anchor_terms))
            )
            # add_loss requires recompiling to rebuild the train function
            # with the new loss term included.
            model.compile(
                optimizer=optimizers.Adam(lr=args.learning_rate),
                loss='binary_crossentropy',
                metrics=['binary_accuracy'],
            )

    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=args.model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            min_delta=1.0e-6,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=args.lr_patience,
            min_lr=args.min_learning_rate,
            verbose=1,
        ),
        CSVLogger(
            history_path,
            append=False,
        ),
    ]

    started = datetime.datetime.now()

    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weights,
        validation_data=(
            x_validation,
            y_validation,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    duration = datetime.datetime.now() - started

    if not os.path.isfile(args.model_path):
        raise RuntimeError(
            'Training finished without producing a model checkpoint.'
        )

    print(
        '\nLoading best validation checkpoint: {}'.format(
            args.model_path
        ),
        flush=True,
    )

    best_model = load_model(
        args.model_path
    )

    validation_probabilities = np.asarray(
        predict_model(
            best_model,
            x_validation,
        ),
        dtype=float,
    )

    test_probabilities = np.asarray(
        predict_model(
            best_model,
            x_test,
        ),
        dtype=float,
    )

    expected_validation_shape = (
        len(x_validation),
        CANDIDATE_COUNT,
    )

    expected_test_shape = (
        len(x_test),
        CANDIDATE_COUNT,
    )

    if validation_probabilities.shape != expected_validation_shape:
        raise ValueError(
            'Validation prediction shape {} does not match expected {}.'
            .format(
                validation_probabilities.shape,
                expected_validation_shape,
            )
        )

    if test_probabilities.shape != expected_test_shape:
        raise ValueError(
            'Test prediction shape {} does not match expected {}.'
            .format(
                test_probabilities.shape,
                expected_test_shape,
            )
        )

    thresholds = calibrate_thresholds(
        y_validation,
        validation_probabilities,
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )

    validation_metrics = classifier_metrics(
        y_validation,
        validation_probabilities,
        thresholds,
    )

    test_metrics = classifier_metrics(
        y_test,
        test_probabilities,
        thresholds,
    )

    print_classifier_metrics(
        'Validation classifier metrics',
        validation_metrics,
    )

    print_classifier_metrics(
        'Test classifier metrics',
        test_metrics,
    )

    evaluation = best_model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    if isinstance(evaluation, (list, tuple)):
        test_loss = float(evaluation[0])
        keras_binary_accuracy = (
            float(evaluation[1])
            if len(evaluation) > 1
            else None
        )
    else:
        test_loss = float(evaluation)
        keras_binary_accuracy = None

    metadata = {
        'created_at_utc': (
            datetime.datetime.utcnow().isoformat()
            + 'Z'
        ),
        'duration': str(duration),

        'model': {
            'path': os.path.abspath(args.model_path),
            'type': 'NN-TENO5 candidate classifier',
            'input_size': STENCIL_SIZE,
            'candidate_count': CANDIDATE_COUNT,
            'architecture': [5, 64, 32, 16, 8, 3],
            'label_convention': {
                '0': 'smooth_keep',
                '1': 'troubled_discard',
            },
        },

        'source_data': {
            'input_path': os.path.abspath(args.input_data),
            'input_sha256': file_sha256(args.input_data),
            'sample_count': int(len(raw_stencils)),
            'original_flux_path': (
                os.path.abspath(args.original_flux_data)
                if args.original_flux_data
                else None
            ),
            'original_flux_metadata': original_flux_metadata,
            'note': (
                'Original scalar flux targets were validated but not used '
                'as classifier labels.'
            ),
            'labels_source': (
                os.path.abspath(args.labels_data)
                if args.labels_data
                else 'generated_from_deterministic_teno5'
            ),
        },

        'normalization': {
            'method': 'per_stencil_min_range',
            'formula': '(X - min(X)) / (max(X) - min(X))',
            'constant_stencil_value': 0.0,
        },

        'teno_teacher': {
            'order': 5,
            'epsilon': args.teno_epsilon,
            'scale_constant': args.teno_scale_constant,
            'power': args.teno_power,
            'cutoff': args.teno_cutoff,
        },

        'training': {
            'runtime_device': runtime_device,
            'warm_start_path': (
                os.path.abspath(args.warm_start_path)
                if args.warm_start_path
                else None
            ),
            'anchor_weight': args.anchor_weight,
            'learning_rate': args.learning_rate,
            'minimum_learning_rate': args.min_learning_rate,
            'batch_size': args.batch_size,
            'maximum_epochs': args.epochs,
            'early_stopping_patience': args.patience,
            'learning_rate_patience': args.lr_patience,
            'seed': args.seed,
            'split_mode': args.split_mode,
            'train_fraction': args.train_fraction,
            'validation_fraction': args.validation_fraction,
            'sample_weights_enabled': (
                not args.disable_sample_weights
            ),
            'positive_weights': positive_weights.tolist(),
        },

        'dataset_sizes': {
            'train': int(len(x_train)),
            'validation': int(len(x_validation)),
            'test': int(len(x_test)),
        },

        'calibrated_thresholds': thresholds.tolist(),

        'validation_metrics': validation_metrics,
        'test_metrics': test_metrics,

        'keras_test_loss': test_loss,
        'keras_test_binary_accuracy': keras_binary_accuracy,

        'artifacts': {
            'history_path': os.path.abspath(history_path),
            'split_path': os.path.abspath(split_path),
            'metadata_path': os.path.abspath(metadata_path),
        },
    }

    with open(
        metadata_path,
        'w',
        encoding='utf-8',
    ) as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
            sort_keys=True,
        )

    print('\nTraining completed successfully.')

    print(
        '  Model      : {}'.format(
            os.path.abspath(args.model_path)
        )
    )

    print(
        '  Metadata   : {}'.format(
            os.path.abspath(metadata_path)
        )
    )

    print(
        '  History    : {}'.format(
            os.path.abspath(history_path)
        )
    )

    print(
        '  Split      : {}'.format(
            os.path.abspath(split_path)
        )
    )

    print(
        '  Thresholds : {}'.format(
            thresholds.tolist()
        )
    )

    print(
        '  Test loss  : {:.8e}'.format(
            test_loss
        )
    )

    print(
        '  Test exact mask accuracy: {:.6f}'.format(
            test_metrics['exact_mask_accuracy']
        )
    )


if __name__ == '__main__':
    main()
