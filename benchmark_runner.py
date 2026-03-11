"""Thin CLI entrypoint — all logic lives in simplex_ipm.helper.benchmark."""

import argparse
from simplex_ipm.helper.benchmark import (
    create_example_problem,
    run_benchmark,
    run_suite,
)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark IPM vs CVXPY baseline')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--n-blocks', type=int, default=3)
    parser.add_argument('--density', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--suite', action='store_true',
                        help='Run full benchmark suite')
    parser.add_argument('--no-scipy', action='store_true',
                        help='Skip SciPy baseline in single-instance mode')
    args = parser.parse_args()

    if args.suite:
        run_suite(seed=args.seed, verbosity=args.verbosity)
        return

    Q, q, blocks = create_example_problem(
        n=args.n, n_blocks=args.n_blocks,
        seed=args.seed, density=args.density)

    print(f"n={args.n}  |K|={args.n_blocks}  density={args.density}  "
          f"seed={args.seed}")
    cv, sp_res, ipm = run_benchmark(
        Q, q, blocks,
        ipm_cfg={'verbosity': args.verbosity},
        run_scipy=not args.no_scipy,
    )

    print(f"\nCVXPY:  {cv['time']:.4f}s  obj={cv['obj']:.10e}")
    if sp_res is not None:
        print(f"SciPy:  {sp_res['time']:.4f}s  obj={sp_res['obj']:.10e}")
    print(f"IPM:    {ipm['time']:.4f}s  obj={ipm['obj']:.10e}  "
          f"iter={ipm.get('iter','-')}")
    if cv['converged'] and ipm['converged']:
        rel = abs(cv['obj'] - ipm['obj']) / max(abs(cv['obj']), 1e-15) #type: ignore
        ratio = cv['time'] / max(ipm['time'], 1e-15) #type: ignore
        arrow = '\u2191' if ratio >= 1 else '\u2193'
        factor = ratio if ratio >= 1 else 1/ratio
        print(f"Obj rel err: {rel:.2e}  |  "
              f"{factor:.2f}x {arrow}")


if __name__ == '__main__':
    main()

