"""Intrinsic per-game skill from move quality: a simplified Regan & Haworth (2011) model.
Each turn (moves 1-8, repetitions and positions beyond three pawns excluded) gives the engine's
evaluations e(m_i) from the mover's side. A move's scaled error is the integral of dx/(1+|x|) from
e(m_i) to e(m_0), in pawns; the proxy is y_i = exp(-(delta_i/s)^c); probabilities solve
p_i = p_0^(1/y_i), sum p_i = 1. Simplifications, all uniform across players: the K listed moves
plus the played move are evaluated, every other legal move is given the K-th move's error (a lower
bound on its error); c is fixed at 0.5 (the paper's fitted c lies in 0.43-0.55); s is fitted per
player per game on a log grid, maximum a posteriori under a weak prior (the paper fits pooled sets
by percentiles).
The fitted s depends only on that game's moves: on neither opponent's rating nor any other game."""
import json, math, numpy as np
C = 0.5
GRID = np.exp(np.linspace(math.log(0.01), math.log(1.5), 121))
def F(cp):                                                 # antiderivative of 1/(1+|x|), x in pawns
    x = max(min(cp / 100, 10.0), -10.0); return math.copysign(math.log1p(abs(x)), x)
def turn_deltas(t):
    """(deltas of all modelled moves, index of the played move) or None if the turn is excluded."""
    top = t["top"]; e0 = top[0][1]
    if abs(e0) > 300 or t["cp"] is None: return None
    f0 = F(e0); d = {u: max(f0 - F(cp), 0.0) for u, cp in top}
    if t["played"] not in d: d[t["played"]] = max(f0 - F(t["cp"]), 0.0)
    rest = t["legal"] - len(d); dk = max(f0 - F(top[-1][1]), 0.0)
    ds = list(d.values()) + [dk] * max(rest, 0)
    return np.array(ds), list(d).index(t["played"])
def loglik(turns):
    """Log-likelihood of the played moves for each s in GRID (vector over the grid)."""
    ll = np.zeros(len(GRID))
    for ds, j in turns:
        y = np.exp(-(ds[None, :] / GRID[:, None]) ** C)    # grid x moves
        lo = np.full(len(GRID), -700.0); hi = np.zeros(len(GRID))   # u = ln p0: sum exp(u / y) = 1
        for _ in range(60):
            mid = (lo + hi) / 2; f = np.exp(mid[:, None] / y).sum(1) - 1
            lo = np.where(f < 0, mid, lo); hi = np.where(f < 0, hi, mid)
        u = (lo + hi) / 2
        ll += u / y[:, j]                                  # ln p_j = ln p0 / y_j
    return ll
TAU = 1.0                                                  # sd of the weak prior on ln s
def per_game(path):
    """{(gid, side): (s, turns, scaled average error)}; side 0 = White (even plies), 1 = Black.
    s is the maximum a posteriori under a weak prior on ln s, N(median of the per-side maximum-
    likelihood values, TAU^2): without it a game whose every move was the engine's first choice has
    s -> 0 (an infinite rating), and one blunder in a short game sends s to the top of the grid.
    The prior uses no rating, so it is the same with or without stated Elo."""
    curves = {}
    for line in open(path):
        r = json.loads(line); sides = ([], [])
        for t in r["turns"]:
            td = turn_deltas(t)
            if td: sides[t["ply"] % 2].append(td)
        for side in (0, 1):
            ts = sides[side]
            if len(ts) >= 5: curves[(r["gid"], side)] = (loglik(ts), len(ts), sum(ds[j] for ds, j in ts) / len(ts))
    lg = np.log(GRID); mu = float(np.median([lg[int(np.argmax(c[0]))] for c in curves.values()]))
    prior = -0.5 * ((lg - mu) / TAU) ** 2
    return {k: (float(GRID[int(np.argmax(ll + prior))]), n, ae) for k, (ll, n, ae) in sorted(curves.items())}
