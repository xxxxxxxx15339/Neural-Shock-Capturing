"""Independent exported-metric, checkpoint-identity and artifact-hash checks."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())

def verify():
    rows=load(OUT/'results/euler_actual_stages/summary.json');profiles=0
    for r in rows:
        data=np.load(OUT/f'results/euler_actual_stages/{r["case"]}_{r["cells"]}_{r["variant"]}.npz')
        if not r['success']:
            assert 'l1' not in r
            continue
        residual=r['dx']*np.sum(data['final']-data['initial'],axis=0)+r['boundary_exchange']
        np.testing.assert_allclose(residual,r['conservation_residual'],atol=1e-14,rtol=1e-12)
        rho=data['final'][:,0];p=.4*(data['final'][:,2]-.5*data['final'][:,1]**2/rho)
        assert np.all(rho>0) and np.all(p>0) and np.isfinite(data['final']).all()
        if data['reference'].size:
            d=data['final']-data['reference']
            for key,value in dict(l1=np.mean(abs(d)),l2=np.sqrt(np.mean(d*d)),linf=np.max(abs(d))).items():
                np.testing.assert_allclose(value,r[key],atol=1e-14,rtol=1e-12)
            profiles+=1
    frozen=load(OUT/'results/final_holdout/frozen_candidates.json')
    assert digest(OUT/'configs/final_holdout.json')==frozen['holdout_sha256']
    for name,expected in frozen['checkpoints'].items():assert digest(OUT/name)==expected
    for p in (OUT/'checkpoints/paired_pilot').glob('*_seed*.json'):
        r=load(p)
        assert not set(r['train_case_ids']).intersection(r['validation_case_ids'])
        assert set(r['warmup_case_ids']).issubset(r['train_case_ids'])
        assert digest(OUT/'configs/final_holdout.json')==r['holdout_sha256']
    preservation=load(OUT/'results/preservation_final.json')
    assert preservation['passed'] and preservation['tracked_release']['passed']
    hashes=load(OUT/'artifact_hashes.json')
    for name,expected in hashes.items():
        p=(OUT/name).resolve()
        assert p.is_relative_to(ROOT),name
        assert digest(p)==expected,name
    return dict(passed=True,exported_euler_profiles_recomputed=profiles,
        new_artifact_hashes_checked=len(hashes),frozen_pilot_checkpoints_checked=len(frozen['checkpoints']),
        original_files_preserved=preservation['files_checked'],
        tracked_release_protected=preservation['tracked_release']['protected_files_checked'],
        baseline_missing_ignored_checkpoints=len(preservation['release']['failures']))
if __name__=='__main__':print(json.dumps(verify(),indent=2))
