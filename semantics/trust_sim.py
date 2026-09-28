"""Signed source trust played through Glicko-2 (simulation). Each claim is true or false; witnesses of known accuracy
p attest it (a win if they say true, a loss if false). A witness plays as an opponent whose weight g(phi) equals |t|,
with t = 2p - 1, so phi = (pi/sqrt 3) sqrt(1/t^2 - 1); a negative t flips the outcome. Compared: every witness at the
same weight; signed trust; with copiers (witnesses that repeat another's answer) counted once by lineage or not; and a
MANDATE anchor (a verifier at t = 1 that attests a share of claims). Consensus says true when the rating is above 1500.
Everything here is simulated; the accuracies are chosen, not measured."""
import math, random
S = 173.7178
def glicko(r, rd, vol, ro, rdo, score, tau=0.5):
    mu, p, muj, pj = (r - 1500) / S, rd / S, (ro - 1500) / S, rdo / S
    g = 1 / math.sqrt(1 + 3 * pj * pj / math.pi ** 2); E = 1 / (1 + math.exp(-g * (mu - muj)))
    v = 1 / (g * g * E * (1 - E)); d = v * g * (score - E); a = math.log(vol * vol)
    f = lambda x: math.exp(x) * (d * d - p * p - v - math.exp(x)) / (2 * (p * p + v + math.exp(x)) ** 2) - (x - a) / tau ** 2
    A = a; B = math.log(d * d - p * p - v) if d * d > p * p + v else next(a - k * tau for k in range(1, 100) if f(a - k * tau) >= 0)
    fA, fB = f(A), f(B)
    while abs(B - A) > 1e-6:
        C = A + (A - B) * fA / (fB - fA); fC = f(C)
        if fC * fB <= 0: A, fA = B, fB
        else: fA /= 2
        B, fB = C, fC
    nv = math.exp(A / 2); ps = math.sqrt(p * p + nv * nv); np_ = 1 / math.sqrt(1 / ps ** 2 + 1 / v)
    return 1500 + S * (mu + np_ * np_ * g * (score - E)), max(30.0, S * np_), nv
def phi_of(t):                                   # opponent deviation (rating scale) whose weight g equals |t|
    t = min(abs(t), 0.999999); return S * (math.pi / math.sqrt(3)) * math.sqrt(1 / (t * t) - 1)
def run(witnesses, n_claims=4000, mode="signed", lineage=True, mandate=0.0, seed=1):
    rng = random.Random(seed); ok = 0; brier = 0.0
    for _ in range(n_claims):
        truth = rng.random() < 0.5; said = {}; r, rd, vol = 1500.0, 350.0, 0.06; seen = set()
        if rng.random() < mandate: r, rd = (1900.0 if truth else 1100.0), 30.0      # a verifier settles it
        for w in witnesses:
            ans = said[w["copies"]] if w.get("copies") else (truth if rng.random() < w["p"] else not truth)
            said[w["name"]] = ans
            root = w.get("copies") or w["name"]
            if lineage and root in seen: continue
            seen.add(root)
            t = 2 * w["p"] - 1 if mode == "signed" else 0.8
            if abs(t) < 1e-6: continue
            score = 1.0 if ans else 0.0
            if t < 0: score = 1.0 - score                                          # reliably wrong: its win counts as a loss
            r, rd, vol = glicko(r, rd, vol, 1500.0, phi_of(t), score)
        pred = r > 1500; ok += pred == truth
        pt = 1 / (1 + 10 ** (-(r - 1500) / 400)); brier += (pt - truth) ** 2
    return ok / n_claims, brier / n_claims
print("== trust t -> the deviation a witness plays with (weight g(phi) = |t|)")
print("   " + "   ".join(f"t={t:<4} RD {phi_of(t):7.0f}" for t in (1.0, 0.9, 0.67, 0.5, 0.3, 0.1, 0.05)))
honest = [{"name": f"h{i}", "p": p} for i, p in enumerate((0.9, 0.75, 0.6))]
liar = [{"name": "liar", "p": 0.2}]
noise = [{"name": f"n{i}", "p": 0.5} for i in range(4)]
copies = [{"name": f"c{i}", "p": 0.55, "copies": "src"} for i in range(6)]
src = [{"name": "src", "p": 0.55}]
cases = [("three honest witnesses (p = .9, .75, .6)", honest),
         ("... plus four coin-flippers (p = .5)", honest + noise),
         ("... plus a reliably wrong witness (p = .2)", honest + liar),
         ("... plus a weak source (p = .55) copied six times", honest + src + copies)]
print("\n== consensus accuracy / Brier score (lower is better), 4,000 simulated claims each")
print(f"   {'witnesses':<50} {'equal weight':>16} {'signed trust':>16} {'signed, no lineage':>20}")
for title, ws in cases:
    a = run(ws, mode="equal"); b = run(ws); c = run(ws, lineage=False)
    print(f"   {title:<50} {a[0]*100:6.1f}% / {a[1]:.3f} {b[0]*100:6.1f}% / {b[1]:.3f} {c[0]*100:8.1f}% / {c[1]:.3f}")
print("\n== with a MANDATE anchor settling a share of claims (signed trust, lineage)")
for m in (0.0, 0.1, 0.5):
    a = run(honest + liar + noise, mandate=m); print(f"   mandate on {m*100:3.0f}% of claims: {a[0]*100:5.1f}% / {a[1]:.3f}")
