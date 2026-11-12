"""Read-only repository inventory and extraction of archived numerical evidence."""
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/four_method_comparison'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def numerical_leaves(x, pointer=''):
    if isinstance(x, dict):
        for k, v in x.items(): yield from numerical_leaves(v, pointer+'/'+str(k).replace('~','~0').replace('/','~1'))
    elif isinstance(x, list):
        # Arrays of states/stencils are evidence artifacts, not scalar measurements.
        if len(x) < 1000 and any(isinstance(v, dict) for v in x):
            for i, v in enumerate(x): yield from numerical_leaves(v, pointer+'/'+str(i))
    elif isinstance(x, (float, int)) and not isinstance(x, bool):
        yield pointer, x

def main():
    import h5py
    import jax, jaxlib, numpy, scipy, matplotlib
    from src.differentiable.model_io import load_checkpoint
    from src.differentiable.teno5_jax_toy import toy_keep_weight
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    keras = {r['path']:r for r in json.loads((OUT/'results/raw/keras_load_audit.json').read_text())}
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts
             and OUT not in p.parents and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    checkpoint_rows = []
    for p in sorted(f for f in files if f.suffix in ('.pkl','.h5','.hdf5')):
        rel = str(p.relative_to(ROOT))
        r = dict(path=rel, sha256=sha(p), input_dimension='', architecture='', output_interpretation='',
                 training_objective='Not recoverable from checkpoint alone; see metadata/source inventory',
                 mixing_head='none', mixing_coefficient='none', parameters='',
                 original_evaluation_scripts='src/differentiable/evaluate.py; src/differentiable/v2_evaluate.py',
                 existing_results='See archived_files.csv and archived_metrics.csv', load_status='', selected=False)
        if p.suffix == '.pkl':
            try:
                c = load_checkpoint(p)
                r.update(input_dimension=5, architecture='5-64-32-16-8-3',
                         output_interpretation='3 sigmoid troubled-candidate probabilities',
                         load_status='SUCCESS; strict finite array/schema validation', mixing_head=c.schema,
                         parameters=sum(w.size+b.size for w,b in c.backbone))
                out = numpy.asarray(1-toy_keep_weight(c.backbone, jnp.zeros((1,5))))
                r['constant_probabilities'] = json.dumps(out.tolist())
                if c.head is not None:
                    r['mixing_coefficient'] = 'learned sigmoid'
                    r['head_architecture'] = '3-1' if c.schema == 'linear' else '8-8-1'
                    r['parameters'] += 4 if c.schema == 'linear' else 81
                    r['training_objective'] = 'unrolled SSPRK3 physics loss; jointly trained backbone and separate head; see historical script'
                elif 'v2/' in rel or 'accuracy_upgrade' in rel:
                    r['mixing_coefficient'] = 'configuration-dependent fixed blend; NO learned head'
                    r['training_objective'] = 'dataset warm-up and/or unrolled PDE loss; adjacent JSON is authoritative'
            except Exception as exc: r['load_status'] = 'UNAVAILABLE: '+repr(exc)
        else:
            try:
                with h5py.File(p, 'r') as f:
                    if 'model_config' in f.attrs:
                        cfg = json.loads(f.attrs['model_config'])
                        layers = cfg['config']['layers'] if isinstance(cfg['config'],dict) else cfg['config']
                        dense = [x['config'] for x in layers if x['class_name']=='Dense']
                        r.update(input_dimension=5, architecture='5-'+'-'.join(str(x['units']) for x in dense),
                                 load_status='HDF5 structure readable; '+('Keras load SUCCESS' if keras.get(rel,{}).get('loaded') else 'Keras execution not verified'),
                                 output_interpretation='sigmoid troubled candidates' if dense[-1]['units']==3 else 'scalar flux regression',
                                 original_evaluation_scripts='scripts/presentation/build_final_comparisons.py; src/analysis/*regression.py')
                    else: r['load_status']='DATASET, not a model checkpoint'
                if rel in keras: r['parameters']=keras[rel].get('parameters','')
                if p.name=='nn_teno5.h5': r['training_objective']='weighted binary cross entropy against TENO teacher; metadata seed42'
            except Exception as exc: r['load_status']='UNAVAILABLE: '+repr(exc)
        r['selected'] = p.name in ('nn_teno5.h5','nn_teno5euler_unrolled_jax_params_curriculum_phase21_richgate.pkl','nn_teno5_unrolled_jax_params_run6_learnedlambda.pkl')
        checkpoint_rows.append(r)
    neno=json.loads((OUT/'results/raw/neno6_load_audit.json').read_text())
    checkpoint_rows.append(dict(path='experiments/four_method_comparison/vendor/NNENO/mytrainmodel1',
        sha256=neno['checkpoint_sha256'],input_dimension=6,architecture='6-64-32-16-8-4',
        output_interpretation='4 sigmoid troubled-candidate probabilities',training_objective='published supervised stencil classification; training code not in author repository',
        mixing_head='none',mixing_coefficient='none',parameters=3228,original_evaluation_scripts='vendor/NNENO/test_teno_interg_adv_function.py',
        existing_results='author smoke reproduced with original classes; no archived local Euler results',
        load_status='SUCCESS: PyTorch weights_only=True; exact Model strict load',selected=True))
    csv_write(OUT/'CHECKPOINT_INVENTORY.csv',checkpoint_rows)
    catalog=[]; evidence=[]
    extensions={'.json','.csv','.npz','.npy','.h5','.hdf5','.png','.pdf','.svg','.pkl','.log','.tex','.md','.py'}
    for p in sorted(f for f in files if f.suffix.lower() in extensions):
        rel=str(p.relative_to(ROOT))
        cat=dict(path=rel,sha256=sha(p),bytes=p.stat().st_size,kind=p.suffix,
                 status='ARCHIVED',use='context only; match settings before comparison')
        catalog.append(cat)
        if p.suffix=='.json' and p.stat().st_size < 5_000_000:
            try:
                v=json.loads(p.read_text())
                for pointer,value in numerical_leaves(v):
                    if any(k in pointer.lower() for k in ('l1','l2','linf','error','runtime','ratio','minimum','accuracy','loss','finite','conservation','wins')):
                        evidence.append(dict(status='ARCHIVED',source_file=rel,source_sha256=cat['sha256'],
                                             json_pointer=pointer,value=value,
                                             comparability='source configuration only; no transfer to matched table'))
            except (ValueError,UnicodeError): pass
    csv_write(OUT/'results/raw/archived_files.csv',catalog)
    csv_write(OUT/'results/raw/archived_metrics.csv',evidence)
    env=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        python=sys.version,platform=platform.platform(),cpu=platform.processor(),jax=jax.__version__,jaxlib=jaxlib.__version__,
        numpy=numpy.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,devices=[str(d) for d in jax.devices()],
        checkpoint_count=len(checkpoint_rows),archived_file_count=len(catalog),archived_metric_count=len(evidence),
        leakage='Historical canonical benchmarks already used in development/validation; this is replication and comparative evaluation, not an untouched holdout.')
    (OUT/'results/raw/environment.json').write_text(json.dumps(env,indent=2))
    print(json.dumps(env,indent=2))

if __name__=='__main__': main()
