"""Independent streaming NumPy WENO7 reference; requires no JAX or Keras.

The existing NumPy Euler flux, positivity validator, and SSPRK3 are reused.
Only storage is streamed instead of retaining the entire fine-grid trajectory.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from src.core.eulerEquations import getEulerFlux, spds
from src.core.SimulationClasses import eulerSimulation
from src.core.TimeSteppingMethods import SSPRK3
from src.schemes import WENO7euler

from .artifacts import create_output, finish_manifest, write_json
from .benchmark_cases import MANIFEST_PATH, grid, initial_state, read_manifest, reference_case
from .model_io import REPO_ROOT, sha256_file


def run_numpy_reference(case, cells, cfl):
    fine_case = reference_case(case, cells)
    fine_case['cfl'] = cfl  # record the actual independent-reference setting
    x = grid(fine_case)
    state = initial_state(fine_case)
    dx, t = case['length']/cells, 0.
    rk, flux = SSPRK3(), getEulerFlux(WENO7euler())
    simulation = eulerSimulation(cells, 2, case['length'], case['final_time'], rk, flux,
                                 lambda xx: initial_state(fine_case, xx), 3)
    minima = [float('inf'), float('inf')]
    stage_count = 0

    def validate(value):
        nonlocal stage_count
        simulation._validate_state(value, stage_count)
        stage_count += 1
        rho = value[:,0]
        pressure = (1.4-1)*(value[:,2]-.5*value[:,1]**2/rho)
        minima[0] = min(minima[0],float(np.min(rho)))
        minima[1] = min(minima[1],float(np.min(pressure)))

    validate(state)
    times, timesteps = [0.], []
    started = time.perf_counter()
    while t < case['final_time']:
        speed = float(np.max(np.abs(spds(state))))
        dt = min(cfl*dx/speed,case['final_time']-t)
        state = rk.stepItEuler(state, flux, dt, dx, 3, validator=validate)
        t += dt
        times.append(t)
        timesteps.append(dt)
    return x, state, dict(scheme='NumPy WENO7', case=fine_case, cfl=cfl,
                         timestep_policy='adaptive CFL, final step shortened to exact T',
                         steps=len(timesteps), minimum_dt=min(timesteps), maximum_dt=max(timesteps),
                         minimum_density=minima[0], minimum_pressure=minima[1],
                         validated_stage_states=stage_count, repair_count=0,
                         wall_seconds=time.perf_counter()-started), np.asarray(times)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=MANIFEST_PATH)
    p.add_argument('--case',default='canonical-shu-osher')
    p.add_argument('--resolutions',type=int,nargs='+',default=[1280,2560])
    p.add_argument('--cfl',type=float,default=.35)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    manifest=read_manifest(args.manifest)
    case=next((c for cases in manifest['suites'].values() for c in cases if c['id']==args.case),None)
    if case is None or case['problem']!='shu_osher' or not 0<args.cfl<=.5:
        p.error('Choose a predeclared Shu–Osher case and a positive CFL <= 0.5')
    output=create_output(args.output)
    write_json(output/'configuration.json',dict(case=case,resolutions=args.resolutions,cfl=args.cfl,manifest_sha256=sha256_file(args.manifest)))
    write_json(output/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
               packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
               source_sha256={str(f.relative_to(REPO_ROOT)):sha256_file(f) for f in (REPO_ROOT/'src').rglob('*.py')}))
    records=[]
    for cells in args.resolutions:
        print('Independent NumPy WENO7 reference:',case['id'],cells,flush=True)
        x,state,record,times=run_numpy_reference(case,cells,args.cfl)
        np.savez_compressed(output/f'weno7-{cells}.npz',x=x,state=state,times=times)
        write_json(output/f'weno7-{cells}.json',record)
        records.append(record)
        print('  completed',record['steps'],'steps in',round(record['wall_seconds'],2),'s',flush=True)
    write_json(output/'summary.json',records)
    finish_manifest(output)
    return 0


if __name__=='__main__':
    sys.exit(main())
