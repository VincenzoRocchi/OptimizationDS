# Project 34: QP Solver on Cartesian Product of Simplices

Feasible-start primal–dual interior-point method for convex QP on a Cartesian product of simplices.

## Problem

```
min  (1/2) x^T Q x + q^T x
s.t. Ex = 1,  x >= 0
```

- Q ∈ ℝⁿˣⁿ symmetric positive semidefinite
- Blocks {I_k} partition {1,…,n} (one simplex constraint per block)
- E ∈ ℝ^{|K|×n} block-summation matrix (built as sparse csr_matrix)

## Project Structure

```
simplex_ipm/
  __init__.py            # Exports IPM
  solver.py              # IPM — H formulation, sparse E, Cholesky
  helper/
    __init__.py
    baseline_solvers.py  # CVXPY (OSQP) and SciPy reference solvers
    benchmark.py         # Benchmark suite, problem generator, timing
benchmark_runner.py      # CLI entrypoint (single problem or full suite)
```

## Solver

### `IPM` (`solver.py`)

Single-class solver. Uses the symmetric **H = Q + X⁻¹Z** formulation with:
- Sparse E matrix (`scipy.sparse.csr_matrix`)
- Schur complement S = E H⁻¹ Eᵀ assembled column-by-column
- Cholesky factorization (dense Q) or SparseLU (sparse Q)
- Fixed regularisation τ = 1e-8
- Fraction-to-boundary step sizing (γ = 0.99)

**Configuration:**

| Parameter | Default | Description |
|---|---|---|
| `sigma` | 0.1 | Centering parameter σ ∈ (0, 0.5) |
| `max_iter` | 100 | Maximum IPM iterations |
| `eps_feas` | 1e-8 | Feasibility tolerance (‖r_P‖∞, ‖r_D‖∞) |
| `eps_comp` | 1e-8 | Complementarity tolerance (μ) |
| `eps_delta` | 1e-8 | Minimum δ for initialization |
| `tau_delta` | 1e-2 | δ scaling factor for range-based rule |
| `tau_reg` | 1e-8 | Fixed regularisation of H |
| `gamma` | 0.99 | Fraction-to-boundary safety factor |
| `verbosity` | 1 | 0=silent, 1=summary, 2=per-iteration |

### Reference Solver (`baseline_solvers.py`)

CVXPY with OSQP backend. Used as the accuracy and performance baseline.

## Usage

```python
from simplex_ipm import IPM

solver = IPM(Q, q, blocks, cfg={'sigma': 0.1, 'verbosity': 2})
result = solver.solve()

x = result['x']   # primal solution
y = result['y']   # dual variables (per-block)
z = result['z']   # dual slack
```

### Problem Generation

```python
from simplex_ipm.helper.benchmark import create_example_problem

Q, q, blocks = create_example_problem(n=500, n_blocks=50, seed=42, density=1.0)
Q, q, blocks = create_example_problem(n=1000, n_blocks=50, seed=42, density=0.05)
```

## Benchmarking

```bash
# Full suite (18 configs)
uv run benchmark_runner.py --suite --seed 42

# Single problem
uv run benchmark_runner.py --n 500 --n-blocks 50
uv run benchmark_runner.py --n 1000 --n-blocks 100 --density 0.1

```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 20 | Problem size |
| `--n-blocks` | 3 | Number of simplex blocks |
| `--density` | 1.0 | Q matrix density (1.0=dense, <1.0=sparse) |
| `--seed` | 42 | Random seed |
| `--verbosity` | 0 | Verbosity level |
| `--suite` | off | Run full benchmark suite |
| `--no-scipy` | off | Skip SciPy baseline in single-instance mode |

### Suite Axes

- **Scaling** (n = 50, 100, 500, 1000, 2000)
- **Sparsity** (density = 1.0, 0.5, 0.1, 0.01 at n=1000)
- **Block count** (|K| = 2, 10, 50, 100, 250 at n=500)
- **Large sparse** (n = 5000, 10000 with density = 0.01, 0.001)

## Dependencies

- numpy ≥ 1.20
- scipy ≥ 1.7
- cvxpy (baseline comparison)

```bash
pip install -r requirements.txt
```
