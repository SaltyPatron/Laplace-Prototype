"""Glicko-2 one game per period with Lichess's settings, as ratings.py replays it, as a function of
an arrival order: games are rated in the order given, so a game that arrives late is rated late."""
import math
S = 173.7178; TAU = 0.75
def g(p): return 1 / math.sqrt(1 + 3 * p * p / math.pi ** 2)
def E(mu, mu_j, p_j): return 1 / (1 + math.exp(-g(p_j) * (mu - mu_j)))
def update(r, rd, vol, ro, rdo, s):
    mu, p, muj, pj = (r - 1500) / S, rd / S, (ro - 1500) / S, rdo / S
    e = E(mu, muj, pj); v = 1 / (g(pj) ** 2 * e * (1 - e)); delta = v * g(pj) * (s - e)
    a = math.log(vol ** 2); A = a
    f = lambda x: math.exp(x) * (delta ** 2 - p * p - v - math.exp(x)) / (2 * (p * p + v + math.exp(x)) ** 2) - (x - a) / TAU ** 2
    if delta ** 2 > p * p + v: B = math.log(delta ** 2 - p * p - v)
    else:
        k = 1
        while f(a - k * TAU) < 0: k += 1
        B = a - k * TAU
    fA, fB = f(A), f(B)
    for _ in range(100):
        if abs(B - A) < 1e-6: break
        C = A + (A - B) * fA / (fB - fA); fC = f(C)
        if fC * fB <= 0: A, fA = B, fB
        else: fA /= 2
        B, fB = C, fC
    vol2 = min(math.exp(A / 2), 0.1)
    pstar = math.sqrt(p * p + vol2 * vol2); p2 = 1 / math.sqrt(1 / pstar ** 2 + 1 / v)
    mu2 = mu + p2 * p2 * g(pj) * (s - e)
    return 1500 + S * mu2, min(max(S * p2, 45), 500), vol2
class Glicko:
    def __init__(self, seed=False, seed_rd=100.0): self.st, self.last, self.seed, self.seed_rd = {}, {}, seed, seed_rd
    def _enter(self, pl, elo, w):
        if pl not in self.st: self.st[pl] = [float(elo), self.seed_rd, 0.06] if (self.seed and elo) else [1500.0, 350.0, 0.06]
        elif pl in self.last:
            days = max((w - self.last[pl]).total_seconds() / 86400, 0)
            r, rd, vol = self.st[pl]; self.st[pl][1] = min(math.sqrt(rd * rd + (vol * S) ** 2 * days * 0.21436), 500)
    def step(self, w, a, b, s, ea, eb):
        """Expected score of a before the game, then the game is rated."""
        self._enter(a, ea, w); self._enter(b, eb, w)
        (ra, rda, va), (rb, rdb, vb) = self.st[a], self.st[b]
        pe = E((ra - 1500) / S, (rb - 1500) / S, math.sqrt(rda ** 2 + rdb ** 2) / S)
        self.st[a] = list(update(ra, rda, va, rb, rdb, s)); self.st[b] = list(update(rb, rdb, vb, ra, rda, 1 - s))
        self.last[a] = self.last[b] = w
        return pe
