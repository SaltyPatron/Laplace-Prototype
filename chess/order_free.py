"""Order-free chess ratings: intrinsic move quality + Whole-History Rating, against Glicko-2.
Pool: TWIC over-the-board classical (pool.py). The most recent 10% of games are held out; every
held-out day is predicted by a WHR fitted to everything before that day (ratings stated at the
board on that day included), and scored by Brier on games where both players state an Elo, as in
Research/Chess.md. Channels, never merged into one source's claim:
  R  game results (Bradley-Terry on the whole history)
  E  the source's stated Elo: one observation per (player, stated value, month), on the first
     day it was stated; repeating the same number in more games is not more testimony
  I  intrinsic estimates from move quality (intrinsic.py), calibrated to Elo by cross-fitting
usage: order_free.py analysis.jsonl out.json"""
import sys, json, math, random, time, datetime as dt, numpy as np, blake3
import pool, whr, intrinsic, glicko2
T0 = time.time()
def say(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
D0 = dt.datetime(2000, 1, 1)
def day(w): return (w - D0).days
G = pool.otb_classical(); cut = int(len(G) * 0.9); train, test = G[:cut], G[cut:]
say(f"{len(G):,} games; {cut:,} before the cut ({G[0][0]:%Y-%m-%d} .. {G[cut-1][0]:%Y-%m-%d}); held out {len(test):,} to {G[-1][0]:%Y-%m-%d}")
OUT = {"pool": len(G), "cut": cut}
# ---------------------------------------------------------------- intrinsic channel, cross-fitted
byid = {g[6]: g for g in G}
fits = intrinsic.per_game(sys.argv[1])
rows = []                                                  # gid, side, player, day, ln s, turns, elo
for (gid, side), (s, n, ae) in sorted(fits.items()):
    g = byid[gid]; rows.append((gid, side, g[1 + side], day(g[0]), math.log(s), n, g[4 + side]))
def fold(gid): return blake3.blake3(bytes.fromhex(gid)).digest()[0] & 1
INTR = []; cal = {}
for f in (0, 1):
    fit_rows = [r for r in rows if fold(r[0]) != f and r[6]]
    X = np.array([r[6] for r in fit_rows], float); Y = np.array([r[4] for r in fit_rows]); N = np.array([r[5] for r in fit_rows], float)
    b, a = np.polyfit(X, Y, 1); res2 = (Y - a - b * X) ** 2
    beta, alpha = np.polyfit(1 / N, res2, 1)              # residual variance = alpha + beta / turns
    cal[f] = dict(a=float(a), b=float(b), alpha=float(alpha), beta=float(beta), n=len(fit_rows), corr=float(np.corrcoef(X, Y)[0, 1]))
    for r in rows:
        if fold(r[0]) == f:
            v = max(alpha + beta / r[5], 1e-3)
            INTR.append((r[2], r[3], (r[4] - a) / b, math.sqrt(v) / abs(b), r[0]))
OUT["intrinsic"] = {"game_sides": len(rows), "calibration": cal,
                    "sd_elo_at_turns": {n: float(np.mean([math.sqrt(max(c['alpha'] + c['beta'] / n, 1e-3)) / abs(c['b']) for c in cal.values()])) for n in (10, 23, 40, 80)}}
bands = {}
for r in rows:
    if r[6]: bands.setdefault(r[6] // 200 * 200, []).append(r[4])
OUT["intrinsic"]["median_s_by_elo"] = {b: (len(v), float(np.exp(np.median(v)))) for b, v in sorted(bands.items())}
say(f"intrinsic: {len(rows):,} game sides; ln s = a + b Elo, b = {cal[0]['b']:.6f}/{cal[1]['b']:.6f}; "
    f"Elo sd of one game side at 23 turns ~ {OUT['intrinsic']['sd_elo_at_turns'][23]:.0f}")
TRAIN_IDS = {g[6] for g in train}
IMEAN = float(np.average([o[2] for o in INTR], weights=[1 / o[3] ** 2 for o in INTR]))
OUT["intrinsic"]["weighted_mean"] = IMEAN
# ---------------------------------------------------------------- stated Elo channel, folded
def stated(games):
    first = {}
    for g in games:
        for p, e in ((g[1], g[4]), (g[2], g[5])):
            if e:
                k = (p, e, g[0].strftime("%Y-%m")); d = day(g[0])
                if k not in first or d < first[k]: first[k] = d
    return [(p, d, e) for (p, e, _), d in first.items()]
# ---------------------------------------------------------------- one WHR fit
def fit(games, channels, w2, sd_elo, prior, obs_upto):
    m = whr.WHR(w2=w2, prior=prior)
    if "R" in channels:
        for g in games: m.game(day(g[0]), g[1], g[2], g[3])
    if "E" in channels:
        for p, d, e in stated(obs_upto): m.observe(d, p, e, sd_elo, "E")
    if "I" in channels:
        for p, d, v, sd, gid in INTR:
            if gid in TRAIN_IDS: m.observe(d, p, v, sd, "I")
    return m.solve()
def rolling(games_all, start, channels, w2, sd_elo):
    """Predict every game from index start on, day by day, each day from a fit to all earlier days."""
    prior = (IMEAN if "E" not in channels and "I" in channels else 1500.0, 350.0)
    preds = []; i = start
    while i < len(games_all):
        d = day(games_all[i][0]); j = i
        while j < len(games_all) and day(games_all[j][0]) == d: j += 1
        m = fit(games_all[:i], channels, w2, sd_elo, prior, games_all[:j])
        for g in games_all[i:j]:
            ra, sa = m.rating(g[1], d); rb, sb = m.rating(g[2], d)
            preds.append((g, whr.expect(ra, sa, rb, sb), ra, rb))
        i = j
    return preds
def brier(preds, need=lambda g: g[4] and g[5]):
    xs = [(p[1] - p[0][3]) ** 2 for p in preds if need(p[0])]; return sum(xs) / len(xs), len(xs)
# ---------------------------------------------------------------- tune on the 81-90% slice
VALCUT = int(cut * 0.9)
OUT["tuning"] = {}
for ch in ("R", "RE"):
    best = None; grid = []
    for w2 in (2.0, 8.0, 30.0):
        for sd in ((50.0, 100.0, 200.0, 400.0) if "E" in ch else (None,)):
            b, n = brier(rolling(train, VALCUT, ch, w2, sd or 100.0)); say(f"tune {ch} w2={w2} sd={sd}: {b:.4f} ({n})")
            grid.append((w2, sd, b))
            if best is None or b < best[2]: best = (w2, sd, b)
    OUT["tuning"][ch] = {"grid": grid, "best": best}
W2R = OUT["tuning"]["R"]["best"][0]; W2E, SDE = OUT["tuning"]["RE"]["best"][:2]
say(f"chosen: results-only w2={W2R}; with stated Elo w2={W2E} sd={SDE}")
# ---------------------------------------------------------------- held-out 10%
def glicko(order, seed, start):
    gl = glicko2.Glicko(seed=seed); pr = []
    for i, g in enumerate(order):
        p = gl.step(g[0], g[1], g[2], g[3], g[4], g[5])
        if i >= start: pr.append((g, p))
    return pr
covered = {o[0] for o in INTR if o[4] in TRAIN_IDS}
def both_cov(g): return g[4] and g[5] and g[1] in covered and g[2] in covered
PRED = {"FIDE (header Elo)": [(g, 1 / (1 + 10 ** ((g[5] - g[4]) / 400))) for g in test if g[4] and g[5]],
        "Glicko-2 cold start": glicko(G, False, cut), "Glicko-2 first attested": glicko(G, True, cut)}
for ch, w2, sd in (("R", W2R, None), ("I", W2R, None), ("RI", W2R, None), ("E", W2E, SDE), ("RE", W2E, SDE), ("REI", W2E, SDE)):
    PRED["WHR " + ch] = rolling(G, cut, ch, w2, sd or 100.0); say(f"WHR {ch} done")
PRED["constant 1/2"] = [(g, 0.5) for g in test]
def decisive(g): return g[4] and g[5] and g[3] != 0.5
OUT["brier"] = {k: brier(v) for k, v in PRED.items()}; OUT["brier_covered"] = {k: brier(v, both_cov) for k, v in PRED.items()}
OUT["brier_decisive"] = {k: brier(v, decisive) for k, v in PRED.items()}
for k in PRED: say(f"  {k:<26} {OUT['brier'][k][0]:.4f} ({OUT['brier'][k][1]:,})   both covered {OUT['brier_covered'][k][0]:.4f} ({OUT['brier_covered'][k][1]:,})   decisive {OUT['brier_decisive'][k][0]:.4f} ({OUT['brier_decisive'][k][1]:,})")
with open(sys.argv[2][:-5] + ".predictions.tsv", "w") as f:
    for k, v in PRED.items():
        for x in v: f.write("\t".join((k, x[0][6], f"{x[1]:.6f}")) + "\n")
# ---------------------------------------------------------------- no metadata: how close to the stated Elo
def closeness(preds):
    firsts = {}
    for g, p, ra, rb in preds:
        for pl, r, e in ((g[1], ra, g[4]), (g[2], rb, g[5])):
            if e and pl not in firsts: firsts[pl] = (r, e)
    out = {}
    for name, sel in (("all", lambda pl: True), ("covered", lambda pl: pl in covered)):
        v = [(r, e) for pl, (r, e) in firsts.items() if sel(pl)]
        R = np.array([x[0] for x in v]); E = np.array([x[1] for x in v])
        out[name] = dict(players=len(v), bias=float((R - E).mean()), rmse=float(np.sqrt(((R - E) ** 2).mean())),
                         mae=float(np.abs(R - E).mean()), corr=float(np.corrcoef(R, E)[0, 1]), sd_stated=float(E.std()))
    return out
OUT["closeness"] = {ch: closeness(PRED["WHR " + ch]) for ch in ("R", "I", "RI", "RE", "REI")}
for ch, c in OUT["closeness"].items(): say(f"closeness {ch}: {c}")
# ---------------------------------------------------------------- order independence
LAST = day(G[cut - 1][0])
def final_whr(order):
    m = fit(order, "RE", W2E, SDE, (1500.0, 350.0), order)
    return {p: m.rating(p, LAST)[0] for p in m.chain}
def final_glicko(order):
    gl = glicko2.Glicko(seed=True)
    for g in order: gl.step(g[0], g[1], g[2], g[3], g[4], g[5])
    return {p: v[0] for p, v in gl.st.items()}
def diff(a, b): d = np.array([abs(a[p] - b[p]) for p in a]); return float(d.max()), float(d.mean()), int((d > 1).sum())
orders = {"played": list(train)}
for s in range(1, 4):
    o = list(train); random.Random(s).shuffle(o); orders[f"shuffled {s}"] = o
lateset = {g[6] for g in train if blake3.blake3(bytes.fromhex(g[6])).digest()[0] < 26}     # ~10% arrive last
orders["10% late"] = [g for g in train if g[6] not in lateset] + [g for g in train if g[6] in lateset]
orders["reversed"] = list(reversed(train))
OI = {}; ref_w = final_whr(orders["played"]); ref_g = final_glicko(orders["played"])
for k, o in orders.items():
    if k == "played": continue
    t = time.time(); w = final_whr(o); tw = time.time() - t
    OI[k] = {"whr": diff(ref_w, w), "glicko": diff(ref_g, final_glicko(o)), "whr_seconds": tw}
    if k in ("shuffled 1", "10% late"): OI[k]["glicko_brier"] = brier(glicko(o + test, True, cut))
    say(f"order {k}: {OI[k]}")
OUT["order"] = OI; OUT["players_train"] = len(ref_w)
json.dump(OUT, open(sys.argv[2], "w"), indent=1, default=float)
say("done")
