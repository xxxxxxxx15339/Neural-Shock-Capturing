"""Attach concrete training/evaluation sources and companion metadata to each artifact."""
import csv
import json
import pickle
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.four_method_comparison.scripts.collect_existing_results import csv_write
import numpy as np
OUT=ROOT/'experiments/four_method_comparison'

def main():
    rows=list(csv.DictReader((OUT/'CHECKPOINT_INVENTORY.csv').open()))
    sources={str(p.relative_to(ROOT)):p.read_text(errors='replace') for folder in ('scripts','src','experiments/accuracy_upgrade')
             for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts}
    for r in rows:
        p=ROOT/r['path'];name=p.name
        matches=[path for path,text in sources.items() if name in text]
        r['training_source_files']='; '.join(s for s in matches if 'Train_' in s or 'training.py' in s)
        evals=[s for s in matches if s not in r['training_source_files']]
        r['artifact_source_references']='; '.join(evals)
        companions=[]
        for other in [p.with_suffix('.json'),p.with_name(p.stem+'_metadata.json')]:
            if other.is_file():companions.append(str(other.relative_to(ROOT)))
        r['companion_metadata']='; '.join(companions)
        if r['mixing_coefficient']=='learned sigmoid':
            r['training_objective']='soft-selection unrolled SSPRK3 loss: MSE + .1 TV + .1 BVD + .1 parameter-anchor regularization; joint backbone/head training; hard validation inference'
            if 'euler' in name:r['training_objective']+='; + 1.0 pre-repair RK-stage positivity penalty; historical nonconservative state repair remains in rollout'
        if name.endswith('_resume_final.pkl'):
            with p.open('rb') as f:v=pickle.load(f) # trusted local training-state artifact
            pairs=v['params'];finite=all(np.isfinite(np.asarray(x)).all() for layer in pairs for x in layer)
            r.update(load_status='SUCCESS as training-resume dictionary; not directly loadable by strict inference loader',
                     architecture='5-64-32-16-8-3',input_dimension=5,parameters=sum(np.asarray(x).size for layer in pairs for x in layer),
                     training_objective='conservative safeguarded15-step rollout component-normalized MSE; exact details in embedded config',
                     output_interpretation='3 troubled probabilities inside params; optimizer/RNG also stored',
                     mixing_coefficient='fixed .25 in embedded config; no learned head',constant_probabilities='finite params='+str(finite))
    csv_write(OUT/'CHECKPOINT_INVENTORY.csv',rows)
    print('Attached source and metadata references to',len(rows),'artifacts')

if __name__=='__main__':main()
