"""Own Glicko-2 chess ratings from observed games (prototype).
Games are deduplicated by content, pooled by (platform, game mode), replayed in the order they were
played, one game per rating period (Lichess settings), and checked on the most recent 10% of games."""
import glob, re, math, time, collections, blake3, datetime as dt
T0 = time.time()
def say(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
TAG = re.compile(r'^\[(\w+) "(.*)"\]')
def games(path):
    head, moves = {}, []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = TAG.match(line)
        if m:
            if moves: yield head, "".join(moves); head, moves = {}, []
            head[m.group(1)] = m.group(2)
        elif line.strip(): moves.append(line)
    if head: yield head, "".join(moves)
def mode(h, otb):
    tc = h.get("TimeControl", "")
    if otb:
        e = h.get("Event", "").lower()
        return "otb-blitz" if "blitz" in e else "otb-rapid" if "rapid" in e else "otb-classical"
    if "/" in tc: return "com-daily"
    b, _, inc = tc.partition("+")
    try: est = int(b) + 40 * int(inc or 0)
    except ValueError: return None
    return "com-bullet" if est < 180 else "com-blitz" if est < 480 else "com-rapid" if est < 1500 else "com-classical"
def when(h):
    d = (h.get("UTCDate") or h.get("Date", "")).replace("?", "1"); t = h.get("UTCTime", "12:00:00")
    try: return dt.datetime.strptime(d + " " + t, "%Y.%m.%d %H:%M:%S")
    except ValueError: return None
files = [(f, False) for f in glob.glob("/vault/Data/Games/Chess/*.pgn")] + [(f, True) for f in sorted(glob.glob("/vault/Data/Games/Chess/twic/*.pgn"))]
pools = collections.defaultdict(list); seen = set(); raw = dup = 0
for f, otb in files:
    for h, mv in games(f):
        raw += 1
        if h.get("Result") not in ("1-0", "0-1", "1/2-1/2") or not h.get("White") or not h.get("Black"): continue
        gid = blake3.blake3((h.get("White", "") + h.get("Black", "") + h.get("Date", "") + h.get("Round", "") + mv).encode()).digest()[:16]
        if gid in seen: dup += 1; continue
        seen.add(gid)
        m, w = mode(h, otb), when(h)
        if m and w: pools[m].append((w, h["White"], h["Black"], {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[h["Result"]],
                                     int(h["WhiteElo"]) if h.get("WhiteElo", "").isdigit() else None,
                                     int(h["BlackElo"]) if h.get("BlackElo", "").isdigit() else None))
say(f"parsed {raw:,} games from {len(files)} files; {dup:,} duplicates removed by content ID; {len(seen):,} distinct")
# ---- Glicko-2, one game per period (Lichess: tau .75, RD in [45,500], vol <= .1, RD grows .21436 periods/day)
import sys
SEED = len(sys.argv) > 1
SEED_RD = 100.0     # the attested platform/FIDE rating is a strong witness: start fairly certain
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
for name in sorted(pools):
    G = sorted(pools[name]); cut = int(len(G) * 0.9)
    st = {}; last = {}
    brier_own = brier_elo = n_eval = 0
    for i, (w, a, b, s, ea, eb) in enumerate(G):
        for pl in (a, b):
            if pl not in st:                                             # enter at a stock default for the attestation level
                elo = ea if pl == a else eb
                st[pl] = [float(elo), SEED_RD, 0.06] if (SEED and elo) else [1500.0, 350.0, 0.06]
            elif pl in last:                                            # deviation grows with idle time
                days = max((w - last[pl]).total_seconds() / 86400, 0)
                r, rd, vol = st[pl]; st[pl][1] = min(math.sqrt(rd * rd + (vol * S) ** 2 * days * 0.21436), 500)
        (ra, rda, va), (rb, rdb, vb) = st[a], st[b]
        if i >= cut and ea and eb:
            pe = E((ra - 1500) / S, (rb - 1500) / S, math.sqrt(rda ** 2 + rdb ** 2) / S)
            pelo = 1 / (1 + 10 ** ((eb - ea) / 400))
            brier_own += (pe - s) ** 2; brier_elo += (pelo - s) ** 2; n_eval += 1
        st[a] = list(update(ra, rda, va, rb, rdb, s)); st[b] = list(update(rb, rdb, vb, ra, rda, 1 - s))
        last[a] = last[b] = w
    players = len(st)
    top = sorted(((r, rd, p) for p, (r, rd, v) in st.items() if rd < 90), reverse=True)[:6]
    print(f"\n== {name}: {len(G):,} games, {players:,} players, {G[0][0]:%Y-%m-%d} … {G[-1][0]:%Y-%m-%d}")
    if n_eval: print(f"   held-out last 10% ({n_eval:,} games with platform Elo): Brier own {brier_own/n_eval:.4f}  vs platform Elo {brier_elo/n_eval:.4f}  (lower is better)")
    for r, rd, p in top: print(f"   {p:<24} {r:7.0f} ± {rd:3.0f}")
    if "Anthony-Hart" in st: r, rd, v = st["Anthony-Hart"]; print(f"   {'Anthony-Hart (you)':<24} {r:7.0f} ± {rd:3.0f}")
say("done")
