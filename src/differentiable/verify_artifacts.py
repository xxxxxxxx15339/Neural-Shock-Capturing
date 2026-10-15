"""Independently recompute exported metrics and verify model/artifact identities.

This checker uses NumPy only and can run in the isolated numerical environment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .model_io import REPO_ROOT, sha256_file


def verify_run(path, compare_run=None):
    path=Path(path)
    config=json.loads((path/'configuration.json').read_text())
    metrics=json.loads((path/'metrics.json').read_text())
    artifact_hashes=json.loads((path/'artifact_hashes.json').read_text())
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    if actual!=set(artifact_hashes):
        raise AssertionError('Artifact manifest does not cover the exact output file set')
    for relative, digest in artifact_hashes.items():
        if sha256_file(path/relative)!=digest:
            raise AssertionError(f'Artifact changed: {relative}')
    for info in config['manifest']['models'].values():
        if sha256_file(REPO_ROOT/info['path'])!=info['sha256']:
            raise AssertionError('Frozen checkpoint changed')
    rows=[]
    for entry in metrics:
        case=entry['case']
        directory=path/case['id']
        with np.load(directory/'reference.npz') as data:
            reference=data['state']
        reference_info=json.loads((directory/'reference.json').read_text())
        digest=hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest()
        if digest!=reference_info['sample_array_sha256']:
            raise AssertionError('Reference sample hash mismatch')
        grids=[]
        for method in entry['methods']:
            with np.load(directory/(method['method']+'.npz')) as data:
                final=data['state']
                difference=final-reference
                primary=float(np.sqrt(np.mean(difference**2))) if final.ndim==1 else float(np.mean(np.abs(difference)))
                if np.isfinite(primary):
                    np.testing.assert_allclose(primary,method['primary_error'],rtol=1e-12,atol=1e-14)
                elif method['valid_for_accuracy_claim']:
                    raise AssertionError('Non-finite result marked valid')
                grid_values=method['time_grid']
                grids.append(tuple(grid_values[k] for k in ('steps','dt','dx')))
                if np.sum(data['history_repaired_rows'])!=method['repaired_stage_rows']:
                    raise AssertionError('Repair count mismatch')
                row=dict(case=case['id'],method=method['method'],primary_error=primary,
                         maximum_conservation_residual=float(np.max(np.abs(method['conservation_residual']))),
                         repaired_stage_rows=method['repaired_stage_rows'],
                         valid_for_accuracy_claim=method['valid_for_accuracy_claim'])
                if compare_run is not None:
                    with np.load(Path(compare_run)/case['id']/(method['method']+'.npz')) as other:
                        delta=float(np.max(np.abs(final-other['state'])))
                        row['independent_state_max_difference']=delta
                        np.testing.assert_allclose(final,other['state'],rtol=1e-10,atol=1e-10)
                    with np.load(Path(compare_run)/case['id']/'reference.npz') as other:
                        np.testing.assert_allclose(reference,other['state'],rtol=1e-12,atol=1e-12)
                rows.append(row)
        if len(set(grids))!=1:
            raise AssertionError('Ablation methods did not share timestep/grid conditions')
    return dict(verified_cases=len(metrics),verified_method_runs=len(rows),
                artifact_hashes_verified=len(artifact_hashes),frozen_checkpoint_hashes_verified=True,
                metric_tolerance=dict(rtol=1e-12,atol=1e-14),
                independent_state_tolerance=dict(rtol=1e-10,atol=1e-10),records=rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--compare-run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=verify_run(args.run,args.compare_run)
    write_json(args.output,result)
    print(f"Verified {result['verified_cases']} cases / {result['verified_method_runs']} method runs")


if __name__=='__main__':
    main()
