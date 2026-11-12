"""Read-only baseline inventory and reproduction comparison; never loads test data."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import jax
jax.config.update('jax_enable_x64', True)
from src.differentiable.model_io import load_checkpoint
from src.differentiable.v2_verify import verify

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/accuracy_upgrade'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):h.update(block)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git', *args],cwd=ROOT,text=True).strip()
def inventory():
    paths=[p for directory in ['models','data','configs/v2','docs/research_release/results']
           for p in (ROOT/directory).rglob('*') if p.is_file()]
    packages={p:md.version(p) for p in ['jax','jaxlib','numpy','scipy','optax','h5py','matplotlib','pytest']}
    checkpoints={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'models').rglob('*')
                 if p.suffix in ('.pkl','.h5','.npz')}
    schemas={}
    for p in (ROOT/'models').rglob('*.pkl'):
        try:
            m=load_checkpoint(p)
            schemas[str(p.relative_to(ROOT))]=dict(schema=m.schema,
                head_inputs=('candidate troubled probabilities' if m.schema=='linear' else
                 'candidate troubled probabilities plus normalized 5-point stencil' if m.schema=='rich' else None),
                architecture=[5,64,32,16,8,3],
                solver=('scalar two-sided state reconstruction then physical flux with LF stabilization' if 'euler' not in p.name else 'historical characteristic split-flux; see adjacent metadata for training budgets'))
        except ValueError as e:schemas[str(p.relative_to(ROOT))]=dict(load_limitation=str(e))
    result=dict(schema=1,git_commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        initial_user_files=['EULER_STABILITY_RESEARCH_AUDIT.md','interview_sebastian_final/'],
        python=sys.version,executable=sys.executable,packages=packages,
        platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(': \t'),
        cpu_affinity=sorted(os.sched_getaffinity(0)),devices=[str(d) for d in jax.devices()],
        memory=Path('/proc/meminfo').read_text().splitlines()[:3],
        checkpoint_hashes=checkpoints,checkpoint_contracts=schemas,
        preserved_files={str(p.relative_to(ROOT)):digest(p) for p in paths},
        dataset_identities=json.loads((ROOT/'data/nnteno_v2/data/quality_report.json').read_text()),
        dataset_access='train and validation only; test files hashed as bytes, never opened as datasets',
        numerical_settings=dict(gamma=1.4,time_integrator='SSPRK3',dtype='float64',
            historical='cell-average-style coefficients applied to F(Ubar) characteristic local LF split flux',
            prototype='periodic characteristic Ubar reconstruction then face-local Rusanov',
            v2=dict(architecture=[5,64,32,16,8,3],candidate_policy='soft deterministic TENO gate',
                    seed=2026,euler_strength=.25,scalar_training_strength=.75,scalar_selected_strength=.25,
                    training_horizon=15,warmup='forward characteristic states, not actual split flux inputs')),
        objectives=dict(v2_scalar='normalized rollout MSE + positive TV growth penalty',
            v2_euler='normalized rollout MSE + pre-repair positivity penalty; nonconservative historical training guard',
            p0='varies by phase: MSE, TV, BVD, regularization and positivity; inspect adjacent metadata and Results.md',
            legacy='supervised candidate discard binary cross-entropy; separate Keras environment'),
        data_reuse='V2 validation used for checkpoint selection and lambda calibration; historical tests previously examined; existing independent cases also previously evaluated',
        known_failures=['general nonlinear Euler FV diagnostic ~second order in historical path',
            'Euler warm-up/inference distribution mismatch',
            'historical state repair changes mass/momentum/energy',
            'selected fixed lambda .25 is mixed across independent cases',
            'closed-wall blast neural instability; classical outcomes depend on configuration; no trustworthy full-time reference',
            'critical-point smooth convergence is not uniformly fifth order',
            'extended feature_vector.py is not wired to saved five-input backbone'],
        limitations=['CPU only; no fresh environment install',
          'active environment NumPy 2.4.6 differs from requirements-numerical pin 1.26.4',
          'legacy TensorFlow stack requires separate Python 3.7 environment',
          'short-reference dataset does not cover generic post-interaction Euler trajectories'],
        seeds=dict(saved_v2=2026,baseline_smoke=20261020,paired_training=[3101,3102],
                   holdout=2026111201,bootstrap=2026111202),
        regression_log='results/baseline_tests.log',regression='117 passed, 37 subtests passed, 240.41 s',
        release_preservation=verify(ROOT/'docs/research_release/results',protected_only=True))
    with (OUT/'baseline_manifest.json').open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':inventory()
