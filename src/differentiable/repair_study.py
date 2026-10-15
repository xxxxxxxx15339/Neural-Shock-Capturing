"""Before/after scientific assessment of the narrowly scoped P0 repair fix.

The historical repair is loaded read-only from the audited git revision;
the selected checkpoints and unguarded kernels are unchanged.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from .artifacts import create_output, finish_manifest, provenance, write_json
from .benchmark_cases import MANIFEST_PATH, analytic_reference, initial_state, read_manifest, time_grid
from .diagnostics import diagnostic_rollout
from .evaluation import build_reference, errors
from .model_io import REPO_ROOT, load_checkpoint
from .reconstruction import make_reconstruction

BASELINE_REVISION = '4e514b19d12247f839dba5af396154cdc37d96ec'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    output=create_output(args.output)
    source=subprocess.check_output(['git','show',BASELINE_REVISION+':src/differentiable/euler_jax.py'],cwd=REPO_ROOT,text=True)
    namespace={}
    exec(compile(source,'historical_euler_jax.py','exec'),namespace)
    historical=namespace['euler_positivity_repair']

    def old_repair(v):
        result=historical(v)
        changed=jnp.any(result!=v,axis=-1)
        # The historical implementation rewrites every row. These counts
        # deliberately label rewrites, not isolated/fallback mechanisms.
        return result,dict(changed_mask=changed,isolated_mask=jnp.zeros_like(changed),fallback_mask=changed)

    manifest=read_manifest()
    info=manifest['models']['euler']
    model=load_checkpoint(info['path'],info['sha256'])
    reconstruct=make_reconstruction(model,'learned-blend')
    write_json(output/'provenance.json',provenance(MANIFEST_PATH))
    write_json(output/'configuration.json',dict(baseline_revision=BASELINE_REVISION,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
               model=model.metadata(),historical_counter_note='fallback_repair_events counts all historical row rewrites, including roundoff',
               conservation_limitation='Neighbor averaging and residual fallback are not globally conservative. No conservative redistribution is added.'))
    cases=[]
    for case in manifest['suites']['canonical']:
        if case['problem'] in ('sod','shu_osher'):
            reference,_,_=build_reference(case)
            cases.append((case['id'],initial_state(case),time_grid(case),case['boundary'],reference))
    n=64
    dx=1/n
    centers=(np.arange(n)+.5)*dx
    speed=.2
    rho=lambda x:.01*(1+.1*np.sin(2*np.pi*x))
    def state_at(t):
        # Exact cell averages of a linear entropy wave.
        density=.01*(1+.1*np.sin(2*np.pi*(centers-speed*t))*np.sinc(dx))
        return np.column_stack((density,speed*density,.01/(1.4-1)+.5*speed**2*density))
    cases.append(('low-positive-entropy-wave',state_at(0),dict(dt=.0005,dx=dx,steps=100),'periodic',state_at(.05)))
    records=[]
    for name,initial,timing,boundary,reference in cases:
        row=dict(case=name,time_grid=timing,boundary=boundary,methods={})
        for policy,repair in [('unguarded',None),('historical',old_repair),('corrected',None)]:
            final,diagnostic,history=diagnostic_rollout(initial,reconstruct,timing['dt'],timing['dx'],timing['steps'],
                 'euler',boundary,repair_policy='none' if policy=='unguarded' else 'historical-repair',repair_function=repair)
            row['methods'][policy]=dict(**errors(final,reference),**diagnostic)
            np.savez_compressed(output/f'{name}-{policy}.npz',initial=initial,state=final,reference=reference,
                                **{'history_'+k:v for k,v in history.items()})
        records.append(row)
        print(name,{k:dict(error=v['primary_error'],changed=v['repaired_stage_rows'],balance=v['physical_balance_error']) for k,v in row['methods'].items()},flush=True)
    write_json(output/'repair_assessment.json',records)
    finish_manifest(output)


if __name__=='__main__':
    main()
