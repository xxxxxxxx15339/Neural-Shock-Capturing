import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.differentiable.benchmark_cases import read_manifest
from src.differentiable.model_io import REPO_ROOT


def run(code, cwd=REPO_ROOT):
    return subprocess.run([sys.executable, '-B', '-c', code], cwd=cwd, text=True, capture_output=True)


def test_historical_scalar_import_has_no_load_or_evaluation_side_effects():
    code = '''
import importlib.util
from pathlib import Path
p=Path('scripts/unrolled_physics_loss/ValidationMatrix_ProjPDF.py')
s=importlib.util.spec_from_file_location('historical_vm',p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
assert m._NETWORK_RECONSTRUCT is None
assert callable(m.step_profile) and callable(m.run_scalar)
'''
    result = run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''


def test_shared_euler_accepts_rich_schema_and_rejects_corrupted_model():
    code = '''
import importlib.util
from pathlib import Path
import jax.numpy as jnp
from src.differentiable.model_io import CheckpointError
p=Path('scripts/unrolled_physics_loss/TENO5euler_vs_NNTENO5euler_JAX.py')
s=importlib.util.spec_from_file_location('historical_euler',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
b,h,error=m.load_network_params('models/nn_teno5euler_unrolled_jax_params_curriculum_phase21_richgate.pkl')
assert error is None
assert m.make_network_reconstruct(b,h)(jnp.ones((4,3,5))).shape==(4,3)
try:
    m.load_network_params('models/nn_teno5euler_unrolled_jax_params.pkl')
except CheckpointError:
    pass
else:
    raise AssertionError('Corrupted checkpoint was accepted')
'''
    result = run(code)
    assert result.returncode == 0, result.stderr


def test_unified_cli_missing_checkpoint_fails_explicitly(tmp_path):
    manifest = read_manifest()
    manifest['models']['scalar']['path'] = str(tmp_path / 'missing.pkl')
    path = tmp_path/'bad-manifest.json'
    path.write_text(json.dumps(manifest))
    result = subprocess.run([sys.executable, '-B', '-m', 'src.differentiable.evaluate', 'benchmark',
                             '--manifest', str(path), '--output', str(tmp_path/'output')],
                            cwd=REPO_ROOT, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'Cannot read checkpoint' in result.stderr
    assert not (tmp_path/'output').exists()


def test_wrapper_help_from_another_directory(tmp_path):
    result = subprocess.run([sys.executable, '-B', str(REPO_ROOT/'scripts/unrolled_physics_loss/Evaluate_Benchmarks.py'), '--help'],
                            cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'reference' in result.stdout
