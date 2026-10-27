"""Whole-History Rating (Coulom 2008) with observation channels, solved as one convex problem.
A player's strength is a curve r(t) with a Wiener prior r(t2) - r(t1) ~ N(0, w^2 |t2 - t1|) (w^2 in
Elo^2 per day), one node per (player, day) on which the player played or was observed. Games are
Bradley-Terry on those nodes (a draw scores 1/2); observations are Gaussian, value +- sd in Elo, each
in its own channel (a source's stated rating, an intrinsic estimate from move quality); a weak prior
N(mean, sd) sits on each player's first node. The log-posterior is strictly concave, so its maximum
is unique: it is solved by Newton's method on the whole sparse Hessian at once (Coulom iterates the
same Newton step player by player), and the answer is a function of the set of games and
observations, not of the order they arrived in. Node indices are assigned in arrival order on
purpose, so an order-independence test exercises the solver and not a canonical sort."""
import math, numpy as np, scipy.linalg as sla, scipy.sparse as sp, scipy.sparse.linalg as spl
Q = math.log(10) / 400                                     # Elo -> natural rating
class WHR:
    def __init__(self, w2=4.0, prior=(1500.0, 350.0)):
        self.w2, self.prior = w2 * Q * Q, prior
        self.node = {}; self.pdays = {}                    # (player, day) -> index ; player -> [days]
        self.games = []; self.obs = []
    def _n(self, p, d):
        k = (p, d)
        if k not in self.node:
            self.node[k] = len(self.node); self.pdays.setdefault(p, []).append(d)
        return self.node[k]
    def game(self, day, a, b, s): self.games.append((self._n(a, day), self._n(b, day), s))
    def observe(self, day, p, value, sd, channel): self.obs.append((self._n(p, day), value, sd, channel))
    def solve(self, tol=1e-10, maxit=50, x0=None):
        n = len(self.node)
        ga = np.array([g[0] for g in self.games], dtype=np.int64); gb = np.array([g[1] for g in self.games], dtype=np.int64)
        gs = np.array([g[2] for g in self.games], dtype=float)
        # quadratic part: Wiener chain per player, first-node prior, observations (constant Hessian)
        rows, cols, vals = [], [], []; lin = np.zeros(n); self.chain = {}
        mu0, sd0 = (self.prior[0] - 1500) * Q, self.prior[1] * Q
        for p, days in self.pdays.items():
            ds = sorted(days); idx = [self.node[(p, d)] for d in ds]; self.chain[p] = (ds, idx)
            rows.append(idx[0]); cols.append(idx[0]); vals.append(1 / sd0 ** 2); lin[idx[0]] += mu0 / sd0 ** 2
            for (d1, i1), (d2, i2) in zip(zip(ds, idx), zip(ds[1:], idx[1:])):
                k = 1 / (self.w2 * (d2 - d1))
                rows += [i1, i2, i1, i2]; cols += [i1, i2, i2, i1]; vals += [k, k, -k, -k]
        for i, v, sd, _ in self.obs:
            k = 1 / (sd * Q) ** 2; rows.append(i); cols.append(i); vals.append(k); lin[i] += (v - 1500) * Q * k
        P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))   # duplicates are summed
        # chain order: every player's nodes consecutive, by day; the preconditioner is the
        # tridiagonal part of the Hessian in that order (each player's own block, as in Coulom)
        order = np.array([i for p in self.chain for i in self.chain[p][1]], dtype=np.int64)
        brk = np.zeros(n, dtype=bool); pos = 0
        for p in self.chain: pos += len(self.chain[p][1]); brk[pos - 1] = True
        Po = P[order][:, order]; sub = np.where(brk[:-1], 0.0, Po.diagonal(1))
        x = np.full(n, mu0) if x0 is None else x0.copy()
        for it in range(maxit):
            d = x[ga] - x[gb]; pr = 1 / (1 + np.exp(-d)); gr = gs - pr; cu = pr * (1 - pr)
            grad = lin - P @ x
            np.add.at(grad, ga, gr); np.add.at(grad, gb, -gr)
            Hg = sp.csr_matrix((np.concatenate([cu, cu, -cu, -cu]), (np.concatenate([ga, gb, ga, gb]), np.concatenate([ga, gb, gb, ga]))), shape=(n, n))
            H = P + Hg
            diag = H.diagonal()[order]; ab = np.zeros((3, n)); ab[0, 1:] = sub; ab[1] = diag; ab[2, :-1] = sub
            def prec(v):
                out = np.empty(n); out[order] = sla.solve_banded((1, 1), ab, v[order]); return out
            M = spl.LinearOperator((n, n), matvec=prec)
            step, info = spl.cg(H, grad, rtol=1e-12, atol=0.0, maxiter=5000, M=M)
            x += step
            if np.max(np.abs(step)) < tol: break
        self.x, self.iters = x, it + 1
        # per-player uncertainty of each node (Coulom App. B.2: the player's own block, opponents fixed)
        Hd = H.diagonal(); self.var = np.zeros(n)
        for p, (ds, idx) in self.chain.items():
            m = len(idx); a = Hd[idx]; b = [-1 / (self.w2 * (ds[j + 1] - ds[j])) for j in range(m - 1)]
            # forward elimination gives the last node's marginal variance of the tridiagonal block
            f = a[0]
            for j in range(1, m): f = a[j] - b[j - 1] ** 2 / f
            self.var[idx[-1]] = 1 / f
        return self
    def rating(self, p, day):
        """(Elo, sd) of player p on a day after the fitted history: the last node, the Wiener variance added."""
        if p not in self.chain: return self.prior[0], self.prior[1]
        ds, idx = self.chain[p]; i = idx[-1]
        return 1500 + self.x[i] / Q, math.sqrt(self.var[i] + self.w2 * max(day - ds[-1], 0)) / Q
def expect(ra, sa, rb, sb):
    """P(a beats b), the uncertainty folded in as Glicko does."""
    v = (sa * sa + sb * sb) * Q * Q
    return 1 / (1 + math.exp(-(ra - rb) * Q / math.sqrt(1 + 3 * v / math.pi ** 2)))
