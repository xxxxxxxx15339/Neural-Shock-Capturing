"""Frozen pre-training backbone controls for separating backbone and gate effects."""
import argparse
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

from .artifacts import create_output, finish_manifest, provenance, write_json, write_metrics_csv
from .benchmark_cases import MANIFEST_PATH, grid, read_manifest
from .evaluation import build_reference, errors, execute, plot_case
from .model_io import load_checkpoint
from .reconstruction import make_reconstruction

CONTROLS = {
    'scalar': dict(path='models/nn_teno5_unrolled_jax_params_run3_pretrained.pkl',
                   sha256='2f801dd8d959232139ead6eeedcc159ae01ff314c228aa12c4e9bd67d19587d9'),
    'euler': dict(path='models/nn_teno5euler_unrolled_jax_params_curriculum_phase15_smoothmix.pkl',
                  sha256='6847699cfa1407084b2f03532cd7367fc0c7b620b5f6df0cc641adb44742eccb'),
}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    manifest=read_manifest()
    warm={name:load_checkpoint(info['path'],info['sha256']) for name,info in CONTROLS.items()}
    final={name:load_checkpoint(info['path'],info['sha256']) for name,info in manifest['models'].items()}
    output=create_output(args.output)
    write_json(output/'configuration.json',dict(controls=CONTROLS,fixed_lambda=manifest['fixed_lambda'],
               cases=manifest['suites']['canonical'],scope='Canonical attribution controls, not held-out tests',
               models={name:model.metadata() for name,model in {**{'warm-'+k:v for k,v in warm.items()},**{'final-'+k:v for k,v in final.items()}}.items()}))
    write_json(output/'provenance.json',provenance(MANIFEST_PATH))
    records=[]
    for case in manifest['suites']['canonical']:
        physics='scalar' if case['problem'] in ('advection','burgers') else 'euler'
        reference,info,arrays=build_reference(case)
        destination=output/case['id'];destination.mkdir()
        write_json(destination/'reference.json',info)
        np.savez_compressed(destination/'reference.npz',state=reference,x=grid(case),**arrays)
        finals={}
        specifications=[('teno5',final[physics],'teno5'),('warmstart-neural-only',warm[physics],'neural-only'),
                        ('warmstart-fixed-blend',warm[physics],'fixed-blend'),('warmstart-max-blend',warm[physics],'max-blend'),
                        ('final-fixed-blend',final[physics],'fixed-blend'),('final-learned-blend',final[physics],'learned-blend')]
        for label,model,method in specifications:
            state,diagnostics,history,timing=execute(case,make_reconstruction(model,method,manifest['fixed_lambda']))
            row=dict(case=case['id'],method=label,time_grid=timing,**errors(state,reference),**diagnostics)
            records.append(row);finals[label]=state
            write_json(destination/(label+'.json'),row)
            np.savez_compressed(destination/(label+'.npz'),state=state,**{'history_'+k:v for k,v in history.items()})
            print(case['id'],label,row['primary_error'],flush=True)
        plot_case(case,grid(case),reference,finals,destination/'controls.png')
    write_json(output/'metrics.json',records)
    write_metrics_csv(output/'metrics.csv',records)
    finish_manifest(output)


if __name__=='__main__':
    main()
