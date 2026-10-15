"""Assemble a reviewable scientific summary without changing any saved run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import create_output, finish_manifest, write_json


def summarize(benchmarks, reference_study=None, numpy_reference=None):
    benchmarks=Path(benchmarks)
    entries=json.loads((benchmarks/'metrics.json').read_text())
    rows=[]
    for entry in entries:
        methods={m['method']:m for m in entry['methods']}
        teacher=methods['teno5']['primary_error']
        learned=methods['learned-blend']
        comparison=next(c for c in entry['comparisons'] if c['method']=='learned-blend')
        rows.append(dict(case=entry['case']['id'], problem=entry['case']['problem'],
                         cells=entry['case']['cells'], metric=learned['metric_definition'],
                         teno5=teacher, neural_only=methods['neural-only']['primary_error'],
                         fixed_blend=methods['fixed-blend']['primary_error'], learned_blend=learned['primary_error'],
                         ratio=learned['primary_error']/teacher, change_percent=100*(learned['primary_error']/teacher-1),
                         status=comparison['status'], maximum_courant=learned['max_courant'],
                         repaired_stage_rows=learned['repaired_stage_rows']))
    result=dict(benchmark_comparisons=rows,reference_findings=None,numpy_crosscheck=None,
                interpretation='Canonical scalar cases selected the checkpoint. New held-out configurations were frozen before evaluation. Lower error on this finite suite is not a universal guarantee.')
    if reference_study is not None:
        studies=json.loads((Path(reference_study)/'reference_summary.json').read_text())
        result['reference_findings']=[dict(case=r['case']['id'],spatial_target_uncertainty=r['spatial_target_uncertainty'],
                 temporal_target_uncertainty=r['temporal_target_uncertainty'],findings=r['findings'],
                 finest_temporal_errors=r['temporal'][-1]['errors']) for r in studies]
    if numpy_reference is not None:
        case=next(e for e in entries if e['case']['id']=='canonical-shu-osher')['case']
        directory=benchmarks/case['id']
        records=[]
        previous=None
        for metadata in json.loads((Path(numpy_reference)/'summary.json').read_text()):
            n=metadata['case']['cells']
            with np.load(Path(numpy_reference)/f'weno7-{n}.npz') as data:
                fine=data['state']; x=data['x']
            target=np.arange(case['cells'])*case['length']/case['cells']
            reference=np.column_stack([np.interp(target,x,fine[:,k]) for k in range(3)])
            errors={}
            for method in ('teno5','neural-only','fixed-blend','learned-blend'):
                with np.load(directory/(method+'.npz')) as data:
                    errors[method]=float(np.mean(np.abs(data['state']-reference)))
            records.append(dict(cells=n,errors=errors,ratio=errors['learned-blend']/errors['teno5'],
                       target_cauchy_l1=None if previous is None else float(np.mean(np.abs(reference-previous))),
                       reference_metadata=metadata,status='Independent scheme sensitivity check; no convergence certification inferred from two grids'))
            previous=reference
        result['numpy_crosscheck']=records
    return result


def plot_profiles(benchmarks, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries=json.loads((Path(benchmarks)/'metrics.json').read_text())
    canonical=[e for e in entries if e['case']['id'].startswith('canonical')]
    figure,axes=plt.subplots(4,2,figsize=(11,12))
    for i,entry in enumerate(canonical):
        directory=Path(benchmarks)/entry['case']['id']
        with np.load(directory/'reference.npz') as data:
            x=data['x']; reference=data['state']
        target=reference[:,0] if reference.ndim==2 else reference
        axes[i,0].plot(x,target,'k:',label='shared reference')
        for method in ('teno5','fixed-blend','learned-blend'):
            with np.load(directory/(method+'.npz')) as data:
                state=data['state']
            value=state[:,0] if state.ndim==2 else state
            axes[i,0].plot(x,value,label=method,linewidth=1)
            axes[i,1].plot(x,value-target,label=method,linewidth=1)
        axes[i,0].set_ylabel(entry['case']['problem']+' density' if reference.ndim==2 else entry['case']['problem']+' u')
        axes[i,1].set_ylabel('error in plotted variable')
        for ax in axes[i]:
            ax.set_xlabel('x');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    figure.suptitle('TENO5 and blends; neural-only remains in every full ablation plot and metric table')
    figure.tight_layout()
    figure.savefig(output/'canonical_blends.png',dpi=160)
    figure.savefig(output/'canonical_blends.pdf')
    plt.close(figure)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmarks',type=Path,required=True)
    p.add_argument('--reference-study',type=Path)
    p.add_argument('--numpy-reference',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    result=summarize(args.benchmarks,args.reference_study,args.numpy_reference)
    output=create_output(args.output)
    write_json(output/'summary.json',result)
    rows=result['benchmark_comparisons']
    with (output/'comparisons.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot_profiles(args.benchmarks,output)
    finish_manifest(output)


if __name__=='__main__':
    main()
