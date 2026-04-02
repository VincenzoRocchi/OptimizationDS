import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve


class IPM:
    """
    Feasible-start primal-dual IPM for QP on Cartesian product of simplices.

    Parameters
    ----------
    Q : array-like or sparse, shape (n, n)
        Symmetric positive semi-definite cost matrix.
    q : array-like, shape (n,)
        Linear cost vector.
    blocks : list of lists
        blocks[k] lists the variable indices belonging to simplex k.
    cfg : dict, optional
        Override any key from ``default_config()``.
    """

    def __init__(self, Q, q, blocks, cfg=None):
        # ---- Q (sparse or dense) ----
        if sp.issparse(Q):
            self.Q = Q.tocsc()
            self.is_sparse = True
        else:
            Q = np.asarray(Q, dtype=float)
            if not np.allclose(Q, Q.T):
                Q = (Q + Q.T) / 2
            self.Q = Q
            self.is_sparse = False

        self.q = np.asarray(q, dtype=float)
        self.n = len(self.q)
        self.blocks = blocks
        self.K = len(blocks)  # number of simplex constraints

        # ---- validate blocks ----
        seen = set()
        for k, blk in enumerate(blocks):
            if len(blk) == 0:
                raise ValueError(f"Block {k} is empty")
            for i in blk:
                if i < 0 or i >= self.n:
                    raise ValueError(f"Index {i} in block {k} out of range")
                if i in seen:
                    raise ValueError(f"Index {i} appears in multiple blocks")
                seen.add(i)
        if len(seen) != self.n:
            raise ValueError(f"Blocks do not cover all {self.n} indices")

        # ---- build sparse E  (K × n) ----
        rows, cols = [], []
        for k, blk in enumerate(blocks):
            for i in blk:
                rows.append(k)
                cols.append(i)
        data = np.ones(len(rows), dtype=float)
        self.E = sp.csr_matrix((data, (rows, cols)), shape=(self.K, self.n))

        # ---- config ----
        self.cfg = self.default_config()
        if cfg is not None:
            self.cfg.update(cfg)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def default_config():
        return {
            'sigma': 0.1,
            'max_iter': 100,
            'eps_feas': 1e-8,
            'eps_comp': 1e-8,
            'eps_delta': 1e-8,
            'tau_delta': 1e-2,
            'tau_reg': 1e-8,        # fixed regularization
            'verbosity': 1,         # 0 silent, 1 summary, 2 per-iteration
            'gamma': 0.99,
        }

    # ------------------------------------------------------------------
    # Fraction-to-boundary step size
    # ------------------------------------------------------------------
    @staticmethod
    def _step_sizes(x, dx, z, dz, gamma=0.99):
        """Return (alpha_pri, alpha_dual, alpha)."""
        alpha_pri = 1.0
        neg = dx < 0
        if np.any(neg):
            alpha_pri = min(1.0, gamma * np.min(-x[neg] / dx[neg]))

        alpha_dual = 1.0
        neg = dz < 0
        if np.any(neg):
            alpha_dual = min(1.0, gamma * np.min(-z[neg] / dz[neg]))

        return alpha_pri, alpha_dual, min(alpha_pri, alpha_dual)

    # ------------------------------------------------------------------
    # Initialization (δ-rule)
    # ------------------------------------------------------------------
    def _initialize(self):
        """Feasible-start δ-rule: uniform x, dual from gradient spread."""
        n, blocks = self.n, self.blocks

        # x^0: uniform on each simplex
        x = np.zeros(n)
        for blk in blocks:
            x[blk] = 1.0 / len(blk)

        # w = Qx + q
        w = self.Q.dot(x) + self.q if self.is_sparse else self.Q @ x + self.q

        # per-block δ_k
        eps_d, tau_d = self.cfg['eps_delta'], self.cfg['tau_delta']
        y = np.zeros(self.K)
        for k, blk in enumerate(blocks):
            wb = w[blk]
            delta_k = max(eps_d, tau_d * (wb.max() - wb.min()))
            y[k] = -wb.min() + delta_k

        z = w + self.E.T @ y
        mu = x.dot(z) / n

        assert np.all(x > 0) and np.all(z > 0), "δ-rule produced non-positive iterates"
        return x, y, z, mu

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------
    def solve(self):
        """
        Run the IPM until convergence or ``max_iter``.

        Returns
        -------
        result : dict
            Keys: x, y, z, mu, iter, converged, history.
        """
        x, y, z, mu = self._initialize()
        cfg = self.cfg
        verb = cfg['verbosity']

        if verb >= 1:
            print(f"IPM  n={self.n}  |K|={self.K}  "
                  f"σ={cfg['sigma']}  μ₀={mu:.2e}")
        if verb >= 2:
            print(f"{'it':>4} {'mu':>10} {'‖rP‖':>10} {'‖rD‖':>10} "
                  f"{'α_pri':>7} {'α_dual':>7}")

        history = []

        for it in range(1, cfg['max_iter'] + 1):
            # ---- residuals ----
            Qx = self.Q.dot(x) if self.is_sparse else self.Q @ x
            r_D = Qx + self.q + self.E.T @ y - z # dual
            r_P = np.asarray(self.E @ x).ravel() - 1.0 # primal
            mu_target = cfg['sigma'] * mu
            r_C = x * z - mu_target # complementarity

            nr_P = np.max(np.abs(r_P))
            nr_D = np.max(np.abs(r_D))

            # ---- convergence check ----
            if nr_P <= cfg['eps_feas'] and nr_D <= cfg['eps_feas'] and mu <= cfg['eps_comp']:
                history.append({'iter': it, 'mu': mu,
                                'norm_r_P': nr_P, 'norm_r_D': nr_D})
                if verb >= 1:
                    print(f"Converged in {it} iterations  "
                          f"μ={mu:.2e}  ‖rP‖={nr_P:.2e}  ‖rD‖={nr_D:.2e}")
                break

            # ---- build H = Q + diag(z/x) + τI  and factorize ----
            x_safe = np.maximum(x, 1e-14)
            x_inv_z = z / x_safe
            tau = cfg['tau_reg']

            r"""
            (\(\
            ( -.-)   Buona Pasqua!   (-.-)
            o_(")(")
            """

            if self.is_sparse:
                H = self.Q + sp.diags(x_inv_z, format='csc')
                if tau > 0:
                    H = H + tau * sp.eye(self.n, format='csc')
                H_solve = sp.linalg.splu(H.tocsc()).solve
            else:
                H = self.Q + np.diag(x_inv_z)
                if tau > 0:
                    H = H + tau * np.eye(self.n)
                cf = cho_factor(H, lower=False)
                H_solve = lambda rhs, _cf=cf: cho_solve(_cf, rhs)

            # ---- Schur complement  S = E H⁻¹ Eᵀ ----
            rhs_core = r_D + r_C / x_safe
            H_inv_rhs = H_solve(rhs_core)

            # assemble S column-by-column
            S = np.empty((self.K, self.K))
            for j in range(self.K):
                e_j = np.zeros(self.K); e_j[j] = 1.0
                Etej = np.asarray(self.E.T @ e_j).ravel()
                S[:, j] = np.asarray(self.E @ H_solve(Etej)).ravel()

            b = r_P - np.asarray(self.E @ H_inv_rhs).ravel()

            # ---- solve for Δy ----
            try:
                S_cf = cho_factor(S, lower=False)
                dy = cho_solve(S_cf, b)
            except np.linalg.LinAlgError:
                S_reg = S + 1e-12 * np.eye(self.K)
                dy = cho_solve(cho_factor(S_reg, lower=False), b)

            # ---- back-substitute Δx, Δz ----
            Etdy = np.asarray(self.E.T @ dy).ravel()
            dx = H_solve(-r_C / x_safe - r_D - Etdy)
            dz = (self.Q.dot(dx) if self.is_sparse else self.Q @ dx) + Etdy + r_D

            # ---- step sizes ----
            # Separate primal/dual step sizes are computed for logging but the update uses the conservative min(α_p, α_d)
            a_p, a_d, alpha = self._step_sizes(x, dx, z, dz, cfg['gamma'])

            # ---- update (single step length) ----
            x = x + alpha * dx
            y = y + alpha * dy
            z = z + alpha * dz
            mu = x.dot(z) / self.n

            history.append({'iter': it, 'mu': mu,
                            'norm_r_P': nr_P, 'norm_r_D': nr_D,
                            'alpha_pri': a_p, 'alpha_dual': a_d})

            if verb >= 2:
                print(f"{it:4d} {mu:10.2e} {nr_P:10.2e} {nr_D:10.2e} "
                      f"{a_p:7.4f} {a_d:7.4f}")

        else:
            # did NOT break = did not converge
            if verb >= 1:
                print(f"Did not converge in {cfg['max_iter']} iterations  μ={mu:.2e}")

        converged = (len(history) > 0 and
                     history[-1].get('norm_r_P', np.inf) <= cfg['eps_feas'] and
                     history[-1].get('norm_r_D', np.inf) <= cfg['eps_feas'] and
                     mu <= cfg['eps_comp'])

        return {
            'x': x, 'y': y, 'z': z,
            'mu': mu,
            'iter': len(history),
            'converged': converged,
            'history': history,
        }