import numpy as np
import scipy.sparse as sp
import time

from simplex_ipm.solver import IPM
from simplex_ipm.helper.baseline_solvers import solve_baseline_cvxpy, solve_baseline_scipy



# Problem generation
def create_example_problem(n=10, n_blocks=3, seed=42, density=1.0):
    """Return (Q, q, blocks) for a random QP on a product of simplices."""
    rng = np.random.default_rng(seed)

    block_size = n // n_blocks
    blocks = [list(range(k * block_size,
                         n if k == n_blocks - 1 else (k + 1) * block_size))
              for k in range(n_blocks)]

    if density >= 1.0:
        A = rng.standard_normal((n, n))
        Q = A.T @ A + 0.1 * np.eye(n)
    else:
        A = sp.random(n, n, density=density, format='csc',
                      random_state=seed) #type: ignore
        Q = A.T @ A + 0.1 * sp.eye(n, format='csc')

    q = rng.standard_normal(n)
    return Q, q, blocks


def compute_objective(Q, q, x):
    Qx = Q.dot(x) if hasattr(Q, 'dot') else Q @ x
    return 0.5 * np.dot(x, Qx) + np.dot(q, x)



# Suite generator
def generate_benchmark_suite(seed=42):
    """
    Build the systematic grid of test configurations.
    """
    suite = []

    # Scaling axis
    for n in [50, 100, 500, 1000, 2000]:
        suite.append(dict(name=f'scale_n{n}', n=n,
                          n_blocks=max(2, n // 10), density=1.0, seed=seed))
    # Sparsity axis
    for density in [1.0, 0.5, 0.1, 0.01]:
        suite.append(dict(name=f'sparse_{int(density*100)}pct', n=1000,
                          n_blocks=50, density=density, seed=seed))
    # Block-count axis
    for n_blocks in [2, 10, 50, 100, 250]:
        suite.append(dict(name=f'blocks_k{n_blocks}', n=500,
                          n_blocks=n_blocks, density=1.0, seed=seed))
    # Large sparse
    for n, density in [(5000, 0.01), (5000, 0.001),
                       (10000, 0.01), (10000, 0.001)]:
        pct = f'{density*100:.1f}'.rstrip('0').rstrip('.')
        suite.append(dict(name=f'lg_n{n}_d{pct}pct', n=n,
                          n_blocks=max(2, n // 50), density=density,
                          seed=seed))
    return suite



# Single-instance benchmark
def _time_solver(solver_func, Q, q, blocks):
    """Run a solver, return dict with time/obj/converged/x."""
    try:
        t0 = time.perf_counter()
        res = solver_func(Q, q, blocks)
        elapsed = time.perf_counter() - t0
        x = res.get('x', None)
        obj = compute_objective(Q, q, x) if x is not None else float('nan')
        return dict(time=elapsed, obj=obj,
                    converged=res.get('converged', False), x=x,
                    iter=res.get('iter'), mu=res.get('mu'))
    except Exception as exc:
        return dict(time=float('nan'), obj=float('nan'),
                    converged=False, x=None, error=str(exc))


def run_benchmark(Q, q, blocks, ipm_cfg=None, run_scipy=True):
    """ 
    un CVXPY and IPM on one problem, optionally SciPy (usually too slow).
    """
    def solve_ipm(Q, q, blocks):
        return IPM(Q, q, blocks, cfg=ipm_cfg).solve()

    cvxpy_out = _time_solver(solve_baseline_cvxpy, Q, q, blocks)
    scipy_out = (_time_solver(solve_baseline_scipy, Q, q, blocks)
                 if run_scipy else None)
    ipm_out   = _time_solver(solve_ipm, Q, q, blocks)
    return cvxpy_out, scipy_out, ipm_out



# Suite runner
def run_suite(seed=42, verbosity=0, **ipm_overrides):
    """
    Run the full benchmark grid and print a compact summary.
    """
    suite = generate_benchmark_suite(seed=seed)
    ipm_cfg = {'verbosity': verbosity}
    ipm_cfg.update(ipm_overrides)

    print(f"{'='*100}")
    print(f"BENCHMARK SUITE  ({len(suite)} configs, seed={seed})")
    print(f"{'='*100}")

    rows = []
    for i, cfg in enumerate(suite, 1):
        name, n, K = cfg['name'], cfg['n'], cfg['n_blocks']
        density = cfg['density']
        print(f"\n[{i}/{len(suite)}]  {name}  n={n}  |K|={K}  dens={density}")

        Q, q, blocks = create_example_problem(
            n=n, n_blocks=K, seed=cfg['seed'], density=density)
        cv, _scipy, ipm = run_benchmark(Q, q, blocks, ipm_cfg=ipm_cfg,
                                        run_scipy=False)

        # ratio convention: >=1 with arrow
        if cv['time'] > 0 and ipm['time'] > 0: #type: ignore
            r = cv['time'] / ipm['time'] #type: ignore
            ratio_s = (f"{r:.2f}x \u2191" if r >= 1
                       else f"{1/r:.2f}x \u2193")
        else:
            ratio_s = "-"

        obj_rel = ""
        if cv['converged'] and ipm['converged']:
            rel = abs(cv['obj'] - ipm['obj']) / max(abs(cv['obj']), 1e-15) #type: ignore
            obj_rel = f"{rel:.2e}"

        rows.append((name, n, K, density,
                      cv['time'], ipm['time'], ratio_s,
                      obj_rel, ipm['converged']))

        print(f"  CVXPY {cv['time']:.4f}s  |  IPM {ipm['time']:.4f}s "
              f"{ipm.get('iter','-')} iter  |  {ratio_s}  "
              f"|  obj rel err {obj_rel}")

    # Compact summary table
    print(f"\n\n{'='*100}")
    print("SUITE SUMMARY")
    print(f"{'='*100}")
    hdr = (f"{'Problem':<18} {'n':>6} {'|K|':>5} {'dens':>5}  "
           f"{'CVXPY(s)':>10} {'IPM(s)':>10} {'Ratio':>10}  "
           f"{'Obj rel err':>12}  {'Conv':>4}")
    print(hdr)
    print("-" * 100)
    for name, n, K, d, ct, ft, rs, om, conv in rows:
        ct_s = f"{ct:.4f}" if not np.isnan(ct) else "FAIL"
        ft_s = f"{ft:.4f}" if not np.isnan(ft) else "FAIL"
        conv_s = "\u2713" if conv else "\u2717"
        print(f"{name:<18} {n:>6} {K:>5} {d:>5.2f}  "
              f"{ct_s:>10} {ft_s:>10} {rs:>10}  "
              f"{om:>12}  {conv_s:>4}")
    print("-" * 100)
    print("\u2191 = IPM faster, \u2193 = IPM slower\n")


