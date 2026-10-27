"""Word-sense disambiguation by pulling (prototype): prior from attested sense frequency (WordNet tag
counts), plus pulls from the sentence's other words through highway hubs (hypernyms, supersense,
gloss words, synset members), discounted by hub frequency. Tuned on SemEval-2007, scored on the rest."""
import math, re, collections, time, xml.etree.ElementTree as ET
T0 = time.time(); WN = "/vault/Data/Wordnet/WordNet-3.0/dict"; W = "/vault/Data/LaplaceResearch/semantics/wsd/WSD_Evaluation_Framework"
POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
key2syn, senses = {}, collections.defaultdict(list)
for line in open(f"{WN}/index.sense"):
    key, off, sn, cnt = line.split()
    lemma, rest = key.split("%"); ss = int(rest.split(":")[0]); p = "nvars"[ss - 1]; p = "a" if p == "s" else p
    syn = f"{off}{'s' if ss == 5 else p}"; key2syn[key] = syn; senses[(lemma, p)].append((syn, int(cnt), int(sn)))
hyper, nbr, lex = collections.defaultdict(list), {}, {}
for p in "nvar":
    for line in open(f"{WN}/data.{ {'n':'noun','v':'verb','a':'adj','r':'adv'}[p] }", encoding="latin-1"):
        if line.startswith(" "): continue
        head, _, gloss = line.partition("|"); f = head.split()
        off, lexn, sst, nw = f[0], f[1], f[2], int(f[3], 16); syn = off + ("s" if sst == "s" else sst)
        words = [f[4 + 2 * i].lower() for i in range(nw)]; i = 4 + 2 * nw; npt = int(f[i]); i += 1
        for k in range(npt):
            sym, toff, tpos = f[i], f[i + 1], f[i + 2]; i += 4
            if sym in ("@", "@i"): hyper[syn].append(toff + tpos)
        lex[syn] = "lex" + lexn
        g = re.findall(r"[a-z]+", gloss.split('"')[0].lower())
        nbr[syn] = set(words) | set(g)
STOP = set("a an the of in to and or for is are be by with as at on that this from it its which who not such any some one".split())
df = collections.Counter(w for s in nbr.values() for w in s)
N = len(nbr); idf = lambda n: math.log(N / (1 + df.get(n, 0)))
def hood(syn, depth=3):
    out = {syn: 1.0, lex.get(syn, ""): 0.3}
    frontier = [syn]
    for d in range(depth):
        frontier = [h for s in frontier for h in hyper.get(s, [])]
        for h in frontier: out[h] = max(out.get(h, 0), 0.7 ** (d + 1))
    for w in nbr.get(syn, ()):
        if w not in STOP: out["w:" + w] = max(out.get("w:" + w, 0), 0.5 * min(idf(w), 8) / 8)
    return out
HOOD = {}
def H(s):
    if s not in HOOD: HOOD[s] = hood(s)
    return HOOD[s]
def load(name):
    root = ET.parse(f"{W}/Evaluation_Datasets/{name}/{name}.data.xml").getroot()
    gold = {l.split()[0]: {key2syn[k] for k in l.split()[1:] if k in key2syn} for l in open(f"{W}/Evaluation_Datasets/{name}/{name}.gold.key.txt")}
    return [[(t.get("id"), t.get("lemma").lower(), t.get("pos")) for t in s] for s in root.iter("sentence")], gold
def run(data, gold, lam):
    ok = tot = 0
    for sent in data:
        ctx = [(l, POS[p]) for _, l, p in sent if p in POS and senses.get((l, POS[p]))]
        for tid, l, p in sent:
            if tid is None or p not in POS: continue
            cands = senses.get((l, POS[p]), [])
            if not cands: continue
            tot += 1
            pull = collections.Counter()
            for cl, cp in ctx:
                if cl == l: continue
                cs = senses[(cl, cp)]; wsum = sum(c + 1 for _, c, _ in cs)
                for s, c, _ in cs:
                    for n, w in H(s).items(): pull[n] += w * (c + 1) / wsum
            total = sum(c + 1 for _, c, _ in cands)
            ctxl = {cl for cl, _ in ctx if cl != l and cl not in STOP}
            def usage(x):                  # SemCor witness: smoothed log P(context lemma | sense), naive-Bayes style
                c = co.get(x[0]); n = sense_n.get(x[0], 0)
                return sum(math.log((c[w] + 0.1) / (n + 1)) - math.log(0.1) for w in ctxl) / max(len(ctxl), 1) if c else 0.0
            best = max(cands, key=lambda x: math.log((x[1] + 1) / total) + lam * sum(w * pull.get(n, 0) for n, w in H(x[0]).items() if n != x[0]) + MU * usage(x))
            ok += best[0] in gold.get(tid, ())
    return ok / tot, tot
# SemCor as a witness: 226k witnessed usages -> which context lemmas co-occur with each sense
sc_root = ET.parse(f"{W}/Training_Corpora/SemCor/semcor.data.xml").getroot()
sc_gold = {l.split()[0]: [key2syn[k] for k in l.split()[1:] if k in key2syn] for l in open(f"{W}/Training_Corpora/SemCor/semcor.gold.key.txt")}
co = collections.defaultdict(collections.Counter); sense_n = collections.Counter()
for sent in sc_root.iter("sentence"):
    toks = [(t.get("id"), t.get("lemma").lower(), t.get("pos")) for t in sent]
    lem = {l for _, l, p in toks if p in POS and l not in STOP}
    for tid, l, p in toks:
        for syn in sc_gold.get(tid, [])[:1]:
            sense_n[syn] += 1
            for c in lem:
                if c != l: co[syn][c] += 1
print(f"[{time.time()-T0:.0f}s] SemCor witness: {sum(sense_n.values()):,} witnessed usages, {len(sense_n):,} senses")
dev = load("semeval2007")
MU = 0.0
grid = [0, 0.2, 0.4, 0.8, 1.6, 3.2]
res = {lam: run(*dev, lam)[0] for lam in grid}
lam = max(res, key=res.get)
best_mu, best = 0.0, res[lam]
for mu in (0.05, 0.1, 0.2, 0.4, 0.8):
    MU = mu; r = run(*dev, lam)[0]
    if r > best: best, best_mu = r, mu
MU = best_mu
print(f"   usage witness weight tuned on SemEval-2007: μ={MU} ({best*100:.1f})")
print(f"[{time.time()-T0:.0f}s] tuned on SemEval-2007: " + ", ".join(f"λ={k}: {v*100:.1f}" for k, v in res.items()) + f"  → λ={lam}")
allsets = ["senseval2", "senseval3", "semeval2013", "semeval2015"]
okp = okm = totn = 0
for name in allsets:
    d = load(name); a, n = run(*d, lam); m, _ = run(*d, 0)
    okp += a * n; okm += m * n; totn += n
    print(f"   {name:<12} {n:>5} instances   prior only (first/most-frequent sense) {m*100:5.1f}   prior + pull {a*100:5.1f}")
print(f"== ALL without SemEval-2007 ({totn:,}): prior only {okm/totn*100:.1f}   prior + pull {okp/totn*100:.1f}   ({time.time()-T0:.0f}s)")
