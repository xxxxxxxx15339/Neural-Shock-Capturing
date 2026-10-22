"""Reproduce tests and results using three explicitly isolated interpreters.

This driver installs nothing, changes no checkpoints, and never overwrites an
output directory. The caller supplies environments built from the release locks.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numerical-python', type=Path, required=True)
    parser.add_argument('--jax-python', type=Path, required=True)
    parser.add_argument('--legacy-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tests-only', action='store_true')
    parser.add_argument('--reference-study', action='store_true', help='Include the longer spatial/temporal reference studies')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == ROOT / 'models' or ROOT / 'models' in output.parents:
        parser.error('Reproduction outputs cannot be written below models/')
    if output.exists():
        parser.error('Output directory already exists')
    if len({str(p.absolute()) for p in (args.numerical_python, args.jax_python, args.legacy_python)}) != 3:
        parser.error('Use three different isolated interpreters')
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', MPLCONFIGDIR=str(output/'matplotlib-cache'),
               JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
    probe = '''
import sys, json
try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata
print(json.dumps(dict(python=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,
                     packages={d.metadata['Name'].lower():d.version for d in metadata.distributions()})))
'''
    interpreters = dict(numerical=args.numerical_python, jax=args.jax_python, legacy=args.legacy_python)
    for name, executable in interpreters.items():
        info = json.loads(subprocess.check_output([str(executable), '-B', '-c', probe], cwd=ROOT,env=env,text=True))
        packages = info['packages']
        forbidden = ['jax', 'keras', 'tensorflow'] if name == 'numerical' else ['keras', 'tensorflow'] if name == 'jax' else ['jax', 'jaxlib']
        if any(p in packages for p in forbidden):
            parser.error(f'{name} environment is not isolated: forbidden packages installed')
        (output/f'{name}-environment.json').write_text(json.dumps(info,indent=2)+'\n')
    numerical_code = '''
import unittest,sys
loader=unittest.TestLoader()
suite=unittest.TestSuite(loader.discover('tests',pattern=p) for p in ['test_core.py','test_teno.py','test_feature_vector.py'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
'''
    commands = [
        ('numerical-tests', args.numerical_python, ['-B','-c',numerical_code]),
        ('jax-tests', args.jax_python, ['-B','-m','pytest','-p','no:cacheprovider',
           'tests/test_teno5_jax_regression.py','tests/test_teno5_jax_euler_regression.py',
           'tests/test_projection_jax.py','tests/test_losses_jax.py','src/differentiable/tests','-q']),
        ('legacy-tests', args.legacy_python, ['-B','-m','unittest','discover','-s','tests','-p','test_networks_smoke.py','-v'])]
    if not args.tests_only:
        commands += [('benchmarks', args.jax_python, ['-B','-m','src.differentiable.evaluate','benchmark','--suite','all','--output',str(output/'benchmarks')]),
                     ('smooth', args.jax_python, ['-B','-m','src.differentiable.evaluate','smooth','--output',str(output/'smooth')])]
        if args.reference_study:
            commands.append(('reference', args.jax_python, ['-B','-m','src.differentiable.evaluate','reference','--suite','all','--output',str(output/'reference')]))
    records = []
    for name, executable, command in commands:
        print('Running',name,flush=True)
        with (output/f'{name}.log').open('x') as log:
            result = subprocess.run([str(executable), *command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(name=name,command=[str(executable),*command],exit_code=result.returncode))
        # Preserve failures and continue independent checks, rather than hiding
        # later outcomes behind the first failure.
        (output/'reproduction.json').write_text(json.dumps(records,indent=2)+'\n')
        print(name,'exit',result.returncode,flush=True)
    return int(any(record['exit_code'] != 0 for record in records))


if __name__ == '__main__':
    sys.exit(main())
