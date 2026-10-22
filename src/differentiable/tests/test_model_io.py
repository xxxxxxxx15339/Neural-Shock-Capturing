import copy
import pickle

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from src.differentiable.benchmark_cases import read_manifest
from src.differentiable.model_io import BACKBONE_SIZES, CheckpointError, REPO_ROOT, load_checkpoint
from src.differentiable.reconstruction import make_reconstruction


def backbone():
    return [(np.zeros((a, b)), np.zeros(b)) for a, b in zip(BACKBONE_SIZES[:-1], BACKBONE_SIZES[1:])]


def save(tmp_path, value):
    path = tmp_path / 'model.pkl'
    path.write_bytes(pickle.dumps(value))
    return path


@pytest.mark.parametrize('schema', ['backbone', 'linear', 'rich'])
def test_three_schemas_and_constant_preservation(tmp_path, schema):
    value = backbone()
    if schema == 'linear':
        value = dict(backbone=value, lambda_head=(np.zeros((3, 1)), np.array([-3.])))
    elif schema == 'rich':
        value = dict(backbone=value, lambda_head_v2=[(np.zeros((8, 8)), np.zeros(8)), (np.zeros((8, 1)), np.array([-3.]))])
    checkpoint = load_checkpoint(save(tmp_path, value))
    assert checkpoint.schema == schema
    assert not checkpoint.backbone[0][0].flags.writeable
    method = 'fixed-blend' if schema == 'backbone' else 'learned-blend'
    reconstruction = make_reconstruction(checkpoint, method)
    np.testing.assert_allclose(np.asarray(reconstruction(jnp.full((4, 3, 5), 2.5))), 2.5, atol=1e-15)


@pytest.mark.parametrize('kind', ['shape', 'bias', 'layers', 'nan', 'inf', 'integer', 'complex', 'not-array', 'unknown-key', 'both-heads', 'bad-head', 'rich-shape'])
def test_invalid_checkpoints_fail_before_inference(tmp_path, kind):
    value = backbone()
    if kind == 'shape': value[1] = (np.zeros((32, 64)), np.zeros(32))
    if kind == 'bias': value[0] = (value[0][0], np.zeros((1, 64)))
    if kind == 'layers': value.pop()
    if kind in ('nan', 'inf'): value[0][0][0, 0] = np.nan if kind == 'nan' else np.inf
    if kind == 'integer': value[0] = (value[0][0].astype(int), value[0][1])
    if kind == 'complex': value[0] = (value[0][0].astype(complex), value[0][1])
    if kind == 'not-array': value[0] = (value[0][0].tolist(), value[0][1])
    if kind == 'unknown-key': value = dict(backbone=value, gate=[])
    if kind == 'both-heads': value = dict(backbone=value, lambda_head=[], lambda_head_v2=[])
    if kind == 'bad-head': value = dict(backbone=value, lambda_head=(np.zeros(3), np.zeros(1)))
    if kind == 'rich-shape': value = dict(backbone=value, lambda_head_v2=[(np.zeros((15, 8)), np.zeros(8)), (np.zeros((8, 1)), np.zeros(1))])
    with pytest.raises(CheckpointError):
        load_checkpoint(save(tmp_path, value))


def test_missing_truncated_and_hash_mismatch(tmp_path):
    with pytest.raises(CheckpointError, match='Cannot read'):
        load_checkpoint(tmp_path / 'absent.pkl')
    path = save(tmp_path, backbone())
    with pytest.raises(CheckpointError, match='SHA-256 mismatch'):
        load_checkpoint(path, '0' * 64)
    path.write_bytes(b'\x80\x04')
    with pytest.raises(CheckpointError, match='deserialize'):
        load_checkpoint(path)


def test_frozen_real_models_and_preserved_failed_experiment():
    for physics, info in read_manifest()['models'].items():
        checkpoint = load_checkpoint(info['path'], info['sha256'])
        assert checkpoint.schema == 'linear'
    assert load_checkpoint('models/nn_teno5euler_unrolled_jax_params_curriculum_phase21_richgate.pkl').schema == 'rich'
    with pytest.raises(CheckpointError, match='non-finite'):
        load_checkpoint('models/nn_teno5euler_unrolled_jax_params.pkl')


def test_ablation_endpoints_match_exactly():
    info = read_manifest()['models']['scalar']
    checkpoint = load_checkpoint(info['path'], info['sha256'])
    stencils = jnp.asarray(np.random.default_rng(8).normal(size=(100, 5)))
    classical = make_reconstruction(checkpoint, 'teno5')(stencils)
    neural = make_reconstruction(checkpoint, 'neural-only')(stencils)
    np.testing.assert_array_equal(np.asarray(make_reconstruction(checkpoint, 'fixed-blend', 0)(stencils)), np.asarray(classical))
    np.testing.assert_array_equal(np.asarray(make_reconstruction(checkpoint, 'fixed-blend', 1)(stencils)), np.asarray(neural))
