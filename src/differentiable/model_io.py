"""Strict, read-only loading of the three historical JAX checkpoint schemas.

Only load trusted pickle files. Legacy Keras HDF5 models have a separate
environment and are deliberately not imported here. No checkpoint is rewritten.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKBONE_SIZES = (5, 64, 32, 16, 8, 3)


class CheckpointError(ValueError):
    """An explicitly requested model cannot safely be evaluated."""


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    sha256: str
    schema: str
    backbone: tuple
    head: object = None

    def metadata(self):
        def layers(values):
            return [dict(weight_shape=list(w.shape), bias_shape=list(b.shape),
                         weight_dtype=str(w.dtype), bias_dtype=str(b.dtype))
                    for w, b in values]
        return dict(path=str(self.path), sha256=self.sha256, schema=self.schema,
                    backbone=layers(self.backbone),
                    head=None if self.head is None else layers(
                        (self.head,) if self.schema == 'linear' else self.head))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _layers(values, sizes, location):
    if not isinstance(values, (list, tuple)) or len(values) != len(sizes) - 1:
        raise CheckpointError(f'{location}: expected {len(sizes)-1} layers')
    result = []
    for i, (pair, n_in, n_out) in enumerate(zip(values, sizes[:-1], sizes[1:])):
        label = f'{location}[{i}]'
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise CheckpointError(f'{label}: expected (weight, bias)')
        arrays = []
        for name, value, shape in zip(('weight', 'bias'), pair,
                                       ((n_in, n_out), (n_out,))):
            if not isinstance(value, np.ndarray):
                raise CheckpointError(f'{label}.{name}: expected NumPy ndarray')
            if value.shape != shape:
                raise CheckpointError(f'{label}.{name}: expected shape {shape}, got {value.shape}')
            if value.dtype not in (np.dtype('float32'), np.dtype('float64')):
                raise CheckpointError(f'{label}.{name}: expected float32/float64, got {value.dtype}')
            if not np.isfinite(value).all():
                raise CheckpointError(f'{label}.{name}: non-finite parameters')
            value.setflags(write=False)
            arrays.append(value)
        result.append(tuple(arrays))
    return tuple(result)


def load_checkpoint(path, expected_sha256=None):
    """Validate identity, exact shapes, dtypes and finiteness before inference."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f'Cannot read checkpoint {path}: {exc}') from exc
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise CheckpointError(f'{path}: SHA-256 mismatch: expected {expected_sha256}, got {digest}')
    try:
        loaded = pickle.loads(data)
    except Exception as exc:
        raise CheckpointError(f'Cannot deserialize checkpoint {path}: {exc}') from exc
    head = None
    if isinstance(loaded, dict):
        keys = set(loaded)
        if keys == {'backbone', 'lambda_head'}:
            schema = 'linear'
            head = _layers([loaded['lambda_head']], (3, 1), 'lambda_head')[0]
        elif keys == {'backbone', 'lambda_head_v2'}:
            schema = 'rich'
            head = _layers(loaded['lambda_head_v2'], (8, 8, 1), 'lambda_head_v2')
        else:
            raise CheckpointError(f'{path}: unknown checkpoint keys {sorted(map(str, keys))}')
        backbone = loaded['backbone']
    elif isinstance(loaded, (list, tuple)):
        schema, backbone = 'backbone', loaded
    else:
        raise CheckpointError(f'{path}: expected a backbone list or a recognized dictionary')
    return Checkpoint(path, digest, schema, _layers(backbone, BACKBONE_SIZES, 'backbone'), head)
