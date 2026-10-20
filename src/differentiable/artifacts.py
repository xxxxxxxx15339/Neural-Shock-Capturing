"""Strict JSON, provenance, and non-overwriting benchmark output directories."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .model_io import REPO_ROOT, sha256_file


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(json_safe(data), stream, indent=2, allow_nan=False)
        stream.write('\n')


def create_output(path):
    path = Path(path).resolve()
    # Protect model identities even if a mistaken output path is supplied.
    if path == REPO_ROOT / 'models' or REPO_ROOT / 'models' in path.parents:
        raise ValueError('Benchmark outputs cannot be written below models/')
    frozen = REPO_ROOT / 'src' / 'differentiable' / 'validation'
    if path == frozen or frozen in path.parents:
        raise ValueError('Frozen P0 validation directories are read-only artifacts')
    path.mkdir(parents=True, exist_ok=False)
    return path


def provenance(manifest_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=REPO_ROOT, text=True).strip()
    import jax
    import inspect
    from .euler_jax import GAMMA, euler_positivity_repair
    from .projection_jax import project_keep_weight
    from .teno5_jax import DEFAULT_EPSILON, DEFAULT_SCALE_CONSTANT, DEFAULT_POWER, DEFAULT_CUTOFF
    sources = sorted((REPO_ROOT / 'src').rglob('*.py'))
    sources += sorted((REPO_ROOT / 'scripts').rglob('*.py'))
    sources += sorted((REPO_ROOT / 'tests').glob('*.py'))
    sources += sorted(REPO_ROOT.glob('requirements*.txt'))
    sources += sorted((REPO_ROOT / 'requirements' / 'locks').glob('*.txt'))
    sources += [Path(manifest_path).resolve()]
    hashes = {str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p): sha256_file(p)
              for p in sources if p.is_file()}
    cpu = ''
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), '')
    return dict(
        schema_version=1, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        command=list(sys.argv), git_revision=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
        git_status=git('status', '--porcelain=v1', '--untracked-files=all'),
        tracked_diff_sha256=hashlib.sha256(git('diff', 'HEAD', '--', 'src', 'tests', 'requirements*.txt', 'README.md').encode()).hexdigest(),
        source_sha256=hashes, python=sys.version, python_executable=sys.executable,
        packages={d.metadata['Name']: d.version for d in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower())},
        platform=platform.platform(), machine=platform.machine(), cpu=cpu,
        logical_cpus=os.cpu_count(), backend=jax.default_backend(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        devices=[str(d) for d in jax.devices()], x64_enabled=bool(jax.config.x64_enabled),
        inference_dtype='float64', timing_policy='End-to-end wall time includes JAX compilation; not a performance comparison',
        numerical_defaults=dict(gamma=GAMMA, integrator='SSPRK3',
             teno5=dict(epsilon=DEFAULT_EPSILON, scale_constant=DEFAULT_SCALE_CONSTANT,
                        power=DEFAULT_POWER, cutoff=DEFAULT_CUTOFF, linear_weights=[.1,.6,.3]),
             neural=dict(stencil_size=5, normalization='per-stencil min-max; constant stencils map to zero',
                         inference_projection='hard', threshold=inspect.signature(project_keep_weight).parameters['threshold'].default),
             optional_repair={key:inspect.signature(euler_positivity_repair).parameters[key].default for key in ('eps','max_velocity','max_scale')},
             validity_policy='Finite raw substages; Euler density/pressure positive; no modified rows. This alone is not a stability or reference-quality proof.'),
        nonfinite_json_policy='Non-finite scalars are null; validity and failure fields remain explicit')


def finish_manifest(output):
    files = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name != 'artifact_hashes.json'}
    write_json(output / 'artifact_hashes.json', files)


def write_metrics_csv(path, records):
    fields = ('case', 'method', 'primary_error', 'l1', 'l2', 'linf',
              'valid_for_accuracy_claim', 'minimum_density', 'minimum_pressure',
              'repaired_stage_rows', 'max_courant')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json_safe(row.get(key)) for key in fields})
