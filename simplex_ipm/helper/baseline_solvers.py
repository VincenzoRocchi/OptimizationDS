import numpy as np
import scipy.sparse
import cvxpy as cp

def solve_baseline_cvxpy(Q, q, blocks):
    """
    Solve QP using CVXPY (OSQP, falling back to ECOS).
    """
    if scipy.sparse.issparse(Q):
        Q = Q.tocsc()
    else:
        Q = np.asarray(Q, dtype=float)
    if not np.allclose(Q, Q.T) if isinstance(Q, np.ndarray) else (Q - Q.T).nnz > 0:
        Q = (Q + Q.T) / 2

    n = len(q)
    x = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(x[blk]) == 1.0 for blk in blocks]
    objective = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Q)) + q @ x)
    problem = cp.Problem(objective, constraints) #type: ignore

    problem.solve(solver=cp.OSQP, verbose=False)
    if problem.status not in ('optimal', 'optimal_inaccurate'):
        problem.solve(solver=cp.ECOS, verbose=False)

    return {
        'x': x.value if x.value is not None else np.full(n, np.nan),
        'status': problem.status,
        'converged': problem.status in ('optimal', 'optimal_inaccurate'),
    }


def solve_baseline_scipy(Q, q, blocks):
    """
    Solve QP using scipy (SLSQP).
    """
    from scipy.optimize import minimize

    if scipy.sparse.issparse(Q):
        Q = Q.toarray()
    Q = np.asarray(Q, dtype=float)
    if not np.allclose(Q, Q.T):
        Q = (Q + Q.T) / 2

    q = np.asarray(q, dtype=float)
    n = len(q)

    def objective(x):
        return 0.5 * x.T @ Q @ x + q @ x

    def gradient(x):
        return Q @ x + q

    def eq_constraints(x):
        return np.array([np.sum(x[blk]) - 1.0 for blk in blocks])

    constraints = [{'type': 'eq', 'fun': eq_constraints}]
    bounds = [(0, None) for _ in range(n)]

    x0 = np.zeros(n)
    for blk in blocks:
        x0[blk] = 1.0 / len(blk)

    res = minimize(objective, x0, method='SLSQP', jac=gradient,
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-9})

    return {
        'x': res.x if res.success else np.full(n, np.nan),
        'status': 'optimal' if res.success else res.message,
        'converged': res.success,
    }
