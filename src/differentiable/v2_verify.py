"""Independent artifact checks, runnable without JAX or legacy TensorFlow."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def verify(root,protected_only=False,tracked_only=False):
    baseline=json.loads((ROOT/'docs/research_release/baseline/manifest.json').read_text())
    failures=[];count=0
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()) if tracked_only else None
    for name,expected in baseline['protected_files'].items():
        if tracked is not None and name not in tracked:continue
        path=ROOT/name
        if not path.is_file() or digest(path)!=expected:failures.append('protected:'+name)
        count+=1
    profiles=0;hashes=0
    if not protected_only:
        for manifest in Path(root).rglob('artifact_hashes.json'):
            for name,expected in json.loads(manifest.read_text()).items():
                path=manifest.parent/name
                if not path.is_file() or digest(path)!=expected:failures.append('artifact:'+str(path))
                hashes+=1
        candidates=[]
        for study in ['independent','wall-closure','wall-smoke','benchmark-smoke','cli-zero']:
            candidates.extend((Path(root)/study).glob('*/*.json'))
        for path in candidates:
            row=json.loads(path.read_text());ref_path=path.parent/'reference.npz'
            if 'method' not in row or 'l1' not in row:continue
            if not ref_path.exists():continue
            ref=np.load(ref_path)['state'];pred=np.load(path.with_suffix('.npz'))['state']
            d=pred-ref;metrics=dict(l1=np.mean(np.abs(d)),l2=np.sqrt(np.mean(d*d)),linf=np.max(np.abs(d)))
            for key,value in metrics.items():
                if row[key] is None and not np.isfinite(value):continue
                if not np.isclose(row[key],value,rtol=1e-12,atol=1e-14):failures.append('metric:'+str(path)+':'+key)
            profiles+=1
    return dict(passed=not failures,protected_files_checked=count,artifact_hashes_checked=hashes,
                profiles_independently_recomputed=profiles,failures=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'docs/research_release/results')
    p.add_argument('--protected-only',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--tracked-only',action='store_true',help='CI clones omit seven ignored local historical checkpoints')
    args=p.parse_args();report=verify(args.root,args.protected_only,args.tracked_only)
    text=json.dumps(report,indent=2);print(text)
    if args.output:
        with args.output.open('x') as stream:stream.write(text+'\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
