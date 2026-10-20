"""Unified research entry point; studies share the existing solver and policies.

Actions: sweep, benchmark, convergence, spatial, inputs, repair, performance, verify, p0.
Use ACTION --help for its configuration/output options. p0 retains the frozen
historical point-sampled benchmark interface; V2 uses cell-average references.
"""
import argparse
import sys


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['sweep','benchmark','convergence','spatial','inputs','repair','performance','verify','p0'])
    if not argv or argv[0] in ('-h','--help'):
        parser.print_help();return 0
    args=parser.parse_args(argv[:1]);rest=argv[1:]
    if args.action=='sweep':
        from .v2_evaluate import main as run
        return run(['sweep',*rest])
    if args.action=='benchmark':
        from .v2_benchmark import main as run
        return run(rest)
    if args.action in ('convergence','spatial','inputs','repair'):
        from .v2_studies import main as run
        return run([args.action,*rest])
    if args.action=='performance':
        from .v2_performance import main as run
        return run(rest)
    if args.action=='p0':
        from .evaluate import main as run
        return run(rest)
    if args.action=='verify':
        from .v2_verify import main as run
        sys.argv=[sys.argv[0],*rest]
        return run()


if __name__=='__main__':raise SystemExit(main())
