"""Word sense disambiguation pulled through the witnessed web (prototype). Every quantity is read as recorded and
combined at query time, never folded in advance: candidates are [lemma, operating language, ILI] claims, each with its
consensus standing (is it true) and its occurrence count (how often it is meant); context pulls through WordNet
relation claims with hub discounting, each context word spread over its senses by prevalence.

Usage: python3 wsd_web.py [operating_language]      (default: eng; the evaluation sets are English)"""
import sys, math, collections, time, xml.etree.ElementTree as ET, psycopg2
OPLANG = sys.argv[1] if len(sys.argv) > 1 else "eng"
sys.argv = [sys.argv[0]]
exec(open("witnesses.py").read().split("# ---------------------------------------------------------------- Glicko-2")[0])
W = "/vault/Data/WSD/WSD_Evaluation_Framework"; POS = {"NOUN", "VERB", "ADJ", "ADV"}
con = psycopg2.connect(host="/tmp", port=5439, user="laplace", dbname="laplace"); cur = con.cursor(); B = psycopg2.Binary
OP = label(OPLANG); ili_ent = {}
for line in open("/vault/Data/CILI/ili-map-pwn30.tab"):
    i, s_ = line.split(); ili_ent[label(i)] = i
cur.execute("""select c.subject, c.object, s.rating, s.deviation,
                      (select coalesce(sum(o.count), 0) from occurrence o where o.claim = c.id),
                      (select d.position from ordinal d where d.claim = c.id and d.witness = %s),
                      (select d.position from ordinal d where d.claim = c.id and d.witness = %s)
               from claim c join consensus s on s.claim = c.id
               where c.predicate = %s""", (B(label("Open English WordNet 2025+")), B(label("Princeton WordNet 3.0")), B(OP)))
                                                                                     # the operating language filters every lookup
rows = cur.fetchall()
cur.execute("select c.subject, c.object from claim c where c.predicate = %s", (B(label("POS")),))  # [ILI, POS, pos]
ipos = {ili_ent[bytes(a)]: bytes(b) for a, b in cur.fetchall() if bytes(a) in ili_ent}
UPOS = {"NOUN": label("n"), "VERB": label("v"), "ADJ": label("a"), "ADV": label("r")}
cands = collections.defaultdict(list)                                                # keyed by (lemma, part of speech): a query filter
for subj, obj, r, rd, n, o_oewn, o_wn in rows:
    I = ili_ent.get(bytes(obj))
    if I and I in ipos: cands[(bytes(subj), ipos[I])].append((I, r, rd, int(n), o_oewn or 99, o_wn or 99))
def prev(cs, a=0.5):                                                                 # prevalence: share of observed usages
    t = sum(c[3] for c in cs) + a * len(cs); return {c[0]: (c[3] + a) / t for c in cs}
PREV = {}
def pv(L):
    if L not in PREV: PREV[L] = prev(cands[L])
    return PREV[L]
REL = [label(x) for x in ("hypernym", "hyponym", "instance hypernym", "instance hyponym", "part meronym", "part holonym",
                          "member meronym", "member holonym", "derivationally related form", "domain of synset topic",
                          "member of domain topic", "similar to", "attribute", "entailment", "cause")]
cur.execute("select subject, object from claim where predicate = any(%s)", ([B(x) for x in REL],))
nbr = collections.defaultdict(set)
for a, b in cur.fetchall():
    A, Bb = ili_ent.get(bytes(a)), ili_ent.get(bytes(b))
    if A and Bb: nbr[A].add(Bb)
deg = {k: len(v) for k, v in nbr.items()}
print(f"loaded {sum(len(v) for v in cands.values()):,} {OPLANG} lexicalization claims ({sum(1 for v in cands.values() for c in v if c[3]):,} with observed usages), {sum(deg.values()):,} relation edges", flush=True)
def hood(I):
    out = {I: 1.0}
    for n in nbr.get(I, ()):
        out[n] = max(out.get(n, 0), 0.6 / math.log(2 + deg.get(n, 0)))
        for m in nbr.get(n, ()):
            if m != I: out[m] = max(out.get(m, 0), 0.25 / math.log(2 + deg.get(m, 0)))
    return out
H = {}
def hd(I):
    if I not in H: H[I] = hood(I)
    return H[I]
def gold_of(name):
    g = {}
    for line in open(f"/vault/Data/WSD/ili_mapped/{name}.gold.ili.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4: g.setdefault(p[0], set()).add(p[3])
    return g
def load(name):
    root = ET.parse(f"{W}/Evaluation_Datasets/{name}/{name}.data.xml").getroot()
    return [[(t.get("id"), t.get("lemma"), t.get("pos")) for t in s] for s in root.iter("sentence")], gold_of(name)
def run(dg, lam, mode):
    data, gold = dg; ok = tot = 0
    for sent in data:
        ctx = [(label(l.replace('_', ' ')), UPOS[p]) for _, l, p in sent if p in POS and cands.get((label(l.replace('_', ' ')), UPOS[p]))]
        for tid, l, p in sent:
            if tid is None or p not in POS: continue
            L = (label(l.replace('_', ' ')), UPOS[p]); cs = cands.get(L)
            if not cs: continue
            tot += 1
            pull = collections.Counter()
            if lam:
                for c in ctx:
                    if c == L: continue
                    wc = pv(c) if mode != "standing" else {x[0]: 1 / len(cands[c]) for x in cands[c]}
                    for I, w0 in wc.items():
                        for n, w in hd(I).items(): pull[n] += w * w0
            pr = pv(L)
            def score(x):
                I, r, rd, n, oo, ow = x
                ctxs = lam * sum(w * pull.get(m, 0) for m, w in hd(I).items() if m != I)
                if mode == "standing": return r / 173.7178 + ctxs
                if mode == "OEWN order": return -math.log(oo) + ctxs
                if mode == "WN3.0 order": return -math.log(ow) + ctxs
                if mode == "prevalence": return math.log(pr[I]) + ctxs - 1e-6 * oo      # recorded order breaks count ties
                return math.log(pr[I]) + ctxs + 0.25 * (r - 2 * rd) / 173.7178 - 1e-6 * oo  # all: standing's lower bound as confidence
            best = max(cs, key=score)
            ok += best[0] in gold.get(tid, ())
    return ok / max(tot, 1), tot
TSTART = time.time(); dev = load("semeval2007"); tests = [load(n) for n in ("senseval2", "senseval3", "semeval2013", "semeval2015")]
for mode in ("standing", "OEWN order", "WN3.0 order", "prevalence", "all"):
    res = {lam: run(dev, lam, mode)[0] for lam in (0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)}
    lam = max(res, key=res.get)
    okp = okm = n = 0; t1 = time.time()
    for d in tests:
        a_, t = run(d, lam, mode); m, _ = run(d, 0, mode); okp += a_ * t; okm += m * t; n += t
    print(f"{mode:<12} tuned λ={lam:<4} four sets ({n:,}): prior only {okm/n*100:5.1f}   prior + pull {okp/n*100:5.1f}"
          f"   ({(time.time()-t1)/(2*n)*1e6:.0f} µs per word)", flush=True)
