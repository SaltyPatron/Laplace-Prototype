"""How much a claim's standing depends on the order its attestations arrive (prototype measurement). Claims attested by
two or more lineages are replayed from stock in the recorded order (most trusted first) and in random orders; reported:
the spread of final ratings and deviations across orders, and how often the ranking of a word's candidate senses changes."""
import math, random, statistics, sys, psycopg2
sys.argv = [sys.argv[0]]
exec(open("/repos/src/Laplace-Prototype/semantics/witnesses.py").read().split("# ---------------------------------------------------------------- witnesses, in order")[0])
S_ = 173.7178
dev = lambda t: S_ * (math.pi / math.sqrt(3)) * math.sqrt(1 / t ** 2 - 1)
import psycopg2.extras; psycopg2.extras.register_uuid()
con = psycopg2.connect(host="localhost", port=5432, user="laplace", dbname="laplace"); cur = con.cursor()
cur.execute("SELECT id, lineage, trust FROM witness"); W = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
cur.execute("""SELECT claim, witnesses FROM standing TABLESAMPLE SYSTEM (2) WHERE cardinality(witnesses) >= 2 LIMIT 60000""")
rows = [(c, [w for w in ws if w in W]) for c, ws in cur.fetchall()]
rows = [(c, ws) for c, ws in rows if len({W[w][0] for w in ws}) >= 2]
def play(ws):
    r, rd, vol = 1500.0, 350.0, 0.06; seen = set()
    for w in ws:
        lin, t = W[w]
        if lin in seen: continue
        seen.add(lin); r, rd, vol = glicko(r, rd, vol, 1500.0, dev(min(t, 0.999)), 1.0)
    return r, rd
rng = random.Random(7); spread_r, spread_d = [], []
for c, ws in rows:
    base = sorted(ws, key=lambda w: -W[w][1]); outs = [play(base)]
    for _ in range(8): p = ws[:]; rng.shuffle(p); outs.append(play(p))
    spread_r.append(max(o[0] for o in outs) - min(o[0] for o in outs)); spread_d.append(max(o[1] for o in outs) - min(o[1] for o in outs))
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
print(f"{len(rows):,} claims attested by 2+ lineages, each replayed in 9 orders")
print(f"  rating spread across orders: median {statistics.median(spread_r):.2f}, 95th pct {q(spread_r, .95):.2f}, max {max(spread_r):.2f} rating points")
print(f"  deviation spread:            median {statistics.median(spread_d):.2f}, 95th pct {q(spread_d, .95):.2f}, max {max(spread_d):.2f}")
