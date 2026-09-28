"""Word senses by intersection, with role trust fitted from data (prototype). A candidate sense's record set is
everything recorded about it: the words of its definitions and examples (Open English WordNet), its synset members,
the members of its one-hop relation neighbors, and SemCor's co-occurrence observations (sense, context lemma, count).
The pull of a sentence on a candidate is its context intersected with that record set, each shared record weighted by
    strand weight x role(UPOS) x role(deprel) x role(syntactic link to the target) x role(lexname) x informativeness
and added to the candidate's log prevalence. The UPOS, deprel, and link of every context word come from a UD parse
(an outside witness, parse_ud.py); informativeness is from Gutenberg container counts (entity_stats).

Weights are fitted by coordinate ascent on SemCor itself, split by document: prevalence and co-occurrence counts come
from the even documents, and accuracy is measured on the odd ones. The four standard test sets are scored once, after
fitting, with prevalence from the occurrence table and co-occurrence from all of SemCor."""
import sys, math, collections, time, gzip, re, xml.etree.ElementTree as ET
import numpy as np, psycopg2
sys.argv = [sys.argv[0]]
exec(open("witnesses.py").read().split("# ---------------------------------------------------------------- Glicko-2")[0])
T00 = time.time()
def say(m): print(f"[{time.time()-T00:6.0f}s] {m}", flush=True)
W = "/vault/Data/WSD/WSD_Evaluation_Framework"; PARSE = "/vault/Data/LaplaceResearch/semantics/parses"
OEWN = "/vault/Data/.refresh-20260903/OpenEnglishWordNet-2025-plus/english-wordnet-2025-plus.xml.gz"
con = psycopg2.connect(host="/tmp", port=5439, user="laplace", dbname="laplace"); cur = con.cursor(); B = psycopg2.Binary

# ---------------------------------------------------------------- candidates: [lemma, eng, ILI] claims, read as recorded
ili_ent = {}
for line in open("/vault/Data/CILI/ili-map-pwn30.tab"):
    i, s_ = line.split(); ili_ent[label(i)] = i
cur.execute("""select c.subject, c.object, (select coalesce(sum(o.count), 0) from occurrence o where o.claim = c.id),
                      (select d.position from ordinal d where d.claim = c.id and d.witness = %s)
               from claim c where c.predicate = %s""", (B(label("Open English WordNet 2025+")), B(label("eng"))))
rows = cur.fetchall()
cur.execute("select subject, object from claim where predicate = %s", (B(label("POS")),))
ipos = {ili_ent[bytes(a)]: bytes(b) for a, b in cur.fetchall() if bytes(a) in ili_ent}
WNPOS = {"NOUN": label("n"), "VERB": label("v"), "ADJ": label("a"), "ADV": label("r")}
cands = collections.defaultdict(list)                        # (lemma entity, pos entity) -> [(ILI, wn count, OEWN order)]
for subj, obj, n, od in rows:
    I = ili_ent.get(bytes(obj))
    if I and I in ipos: cands[(bytes(subj), ipos[I])].append((I, int(n), od or 99))
say(f"candidates: {sum(len(v) for v in cands.values()):,} claims")

# ---------------------------------------------------------------- record sets from Open English WordNet
TOK = re.compile(r"[\w'-]+")
gloss = collections.defaultdict(set); members = collections.defaultdict(set); lexfile = {}; syn2ili = {}; rel = collections.defaultdict(set)
entry_senses = collections.defaultdict(list)                  # (lowercased lemma, wn pos) -> ILIs in OEWN order
pend = []
for _, el in ET.iterparse(gzip.open(OEWN)):
    if el.tag == "Synset":
        I = el.get("ili", "")
        if I.startswith("i") and I != "in":
            syn2ili[el.get("id")] = I; lexfile[I] = el.get("lexfile")
            for d in el.findall("Definition") + el.findall("Example"):
                gloss[I] |= {w.lower() for w in TOK.findall(d.text or "")}
            pend += [(I, r.get("target")) for r in el.findall("SynsetRelation")]
        el.clear()
    elif el.tag == "LexicalEntry":
        lm = el.find("Lemma"); w = lm.get("writtenForm").lower(); p = lm.get("partOfSpeech"); p = "a" if p == "s" else p
        for sn in el.findall("Sense"): entry_senses[(w, p)].append(sn.get("synset"))
        el.clear()
for k, v in entry_senses.items():
    entry_senses[k] = [syn2ili[s] for s in v if s in syn2ili]
    for I in entry_senses[k]: members[I].add(k[0])
for I, tgt in pend:
    J = syn2ili.get(tgt)
    if J: rel[I] |= members[J]
say(f"record sets: {len(gloss):,} concepts with definitions/examples, {len(rel):,} with relation neighbors")

# ---------------------------------------------------------------- corpora with their UD parse
def load(name, path):
    parse = collections.defaultdict(list)
    for line in open(f"{PARSE}/{name}.ud.tsv"):
        sid, i, tid, form, upos, head, dep = line.rstrip("\n").split("\t"); parse[sid].append((upos, int(head), dep.split(":")[0]))
    out = []
    for s in ET.parse(path).getroot().iter("sentence"):
        toks = [(t.get("id"), t.get("lemma").lower(), t.get("pos"), (t.text or "").lower()) for t in s]
        pr = parse.get(s.get("id"))
        if not toks or not pr or len(pr) != len(toks): continue
        out.append((s.get("id"), [tk + pr[j] for j, tk in enumerate(toks)]))   # (tid, lemma, pos, form, upos, head, deprel)
    return out
def gold_of(name):
    g = {}
    for line in open(f"/vault/Data/WSD/ili_mapped/{name}.gold.ili.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4: g.setdefault(p[0], set()).add(p[3])
    return g
semcor = load("semcor", f"{W}/Training_Corpora/SemCor/semcor.data.xml"); gsem = gold_of("semcor")
tests = {n: (load(n, f"{W}/Evaluation_Datasets/{n}/{n}.data.xml"), gold_of(n)) for n in ("semeval2007", "senseval2", "senseval3", "semeval2013", "semeval2015")}
say(f"SemCor {len(semcor):,} parsed sentences; tests " + ", ".join(f"{k} {len(v[0])}" for k, v in tests.items()))

# ---------------------------------------------------------------- informativeness from Gutenberg containers
lem_ids = {}
for sents in [semcor] + [t[0] for t in tests.values()]:
    for _, toks in sents:
        for tk in toks: lem_ids.setdefault(tk[1], None)
for l in lem_ids: lem_ids[l] = label(l.replace("_", " "))
cur.execute("select count(*) from entity_stats s join entity e on e.id = s.id where e.tier = 3"); NS = cur.fetchone()[0]  # Gutenberg sentences
cur.execute("select id, parents from entity_stats where id = any(%s)", ([B(v) for v in set(lem_ids.values())],))
par = {bytes(i): p for i, p in cur.fetchall()}
INFO = {l: min(1.0, max(0.0, math.log(NS / (1 + par.get(i, 0))) / math.log(NS))) for l, i in lem_ids.items()}
say(f"informativeness for {len(INFO):,} lemmas from {NS:,} Gutenberg sentences")

# ---------------------------------------------------------------- categorical role vocabularies
UPOS = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "NUM", "INTJ", "PUNCT", "SYM", "X"]
DEPS = sorted({tk[6] for _, toks in semcor for tk in toks}); LINKS = ["head", "child", "sibling", "other"]
LEX = sorted(set(lexfile.values())) + ["none"]; STRANDS = ["gloss", "members", "relations", "cooccurrence"]
iU = {u: k for k, u in enumerate(UPOS)}; iD = {d: k for k, d in enumerate(DEPS)}; iL = {l: k for k, l in enumerate(LEX)}
U2WN = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
def lex_of(lemma, upos):
    s = entry_senses.get((lemma.replace("_", " "), U2WN.get(upos, "")))
    return lexfile.get(s[0], "none") if s else "none"

# ---------------------------------------------------------------- observations: prevalence and co-occurrence
def observe(sents, gold):
    ns = collections.Counter(); c = collections.defaultdict(collections.Counter); lc = collections.Counter(); N = 0
    for _, toks in sents:
        lem = [tk[1] for tk in toks]; N += len(lem); lc.update(lem)
        for tk in toks:
            for I in gold.get(tk[0], ()):
                ns[I] += 1; c[I].update(l for l in lem if l != tk[1])
    return ns, c, lc, N

# ---------------------------------------------------------------- features: every shared record as one sparse entry
def features(sents, gold, prior, obs):
    ns, co, lc, N = obs
    inst_c0, cand_prior, cand_gold, E = [], [], [], collections.defaultdict(list)
    nc = 0; ni = 0
    for _, toks in sents:
        for i, (tid, lem, pos, form, upos, head, dep) in enumerate(toks):
            if tid is None or pos not in WNPOS: continue
            cs = cands.get((lem_ids[lem], WNPOS[pos]))
            if not cs: continue
            inst_c0.append(nc); ni += 1
            pr = prior(cs)
            ctx = []
            for j, t2 in enumerate(toks):
                if j == i: continue
                link = "head" if head == j + 1 else "child" if t2[5] == i + 1 else "sibling" if t2[5] == head else "other"
                ctx.append((t2[1], t2[3], iU.get(t2[4], iU["X"]), iD.get(t2[6], 0), LINKS.index(link), iL[lex_of(t2[1], t2[4])]))
            for (I, n, od) in cs:
                cand_prior.append(pr[I]); cand_gold.append(I in gold.get(tid, ()))
                g, m, r = gloss.get(I, ()), members.get(I, ()), rel.get(I, ())
                ng, nm_, nr = (1 / math.sqrt(len(x)) if x else 0 for x in (g, m, r))   # normalized by set size, as cosine is
                cI = co.get(I) or {}; nI = ns.get(I, 0)
                for (l, f, u, d, lk, lx) in ctx:
                    w = INFO.get(l, 0.5); key = (u, d, lk, lx)
                    if l in g or f in g: E[0].append((nc, *key, w * ng))
                    if l in m: E[1].append((nc, *key, w * nm_))
                    if l in r: E[2].append((nc, *key, w * nr))
                    if lc[l]:                                                      # evidence from every context word, for or against
                        pl = lc[l] / N; llr = math.log2((cI.get(l, 0) + BETA * pl) / (nI + BETA) / pl)
                        E[3].append((nc, *key, llr))
                nc += 1
    inst_c0.append(nc)
    arr = {k: np.array(v, dtype=np.float64).reshape(-1, 6) for k, v in E.items()}
    return np.array(inst_c0), np.array(cand_prior), np.array(cand_gold, dtype=bool), arr
BETA = 20.0                                                                    # co-occurrence smoothing toward the word's base rate
def prior_from(counts):
    def pr(cs):                                                                # observed usages, with the recorded sense order as pseudo-counts
        pc = {I: counts(I, n) + 1.0 / od for I, n, od in cs}; t = sum(pc.values())
        return {I: math.log(pc[I] / t) for I in pc}
    return pr

# ---------------------------------------------------------------- scoring
def evaluate(F, P):
    c0, prior, gold, arr = F
    s = prior.copy()
    for k, a in arr.items():
        if not len(a): continue
        w = P["strand"][k] * P["upos"][a[:, 1].astype(int)] * P["dep"][a[:, 2].astype(int)] * P["link"][a[:, 3].astype(int)] * P["lex"][a[:, 4].astype(int)] * a[:, 5]
        s += np.bincount(a[:, 0].astype(int), weights=w, minlength=len(s))
    best = np.maximum.reduceat(s, c0[:-1])
    ok = 0
    for k in range(len(c0) - 1):                                               # first candidate reaching the best score
        seg = s[c0[k]:c0[k + 1]]; ok += gold[c0[k] + int(np.argmax(seg >= best[k] - 1e-12))]
    return ok / (len(c0) - 1)
def params(strand=(0, 0, 0, 0), upos=None, dep=None, link=None, lex=None):
    return {"strand": np.array(strand, float), "upos": np.ones(len(UPOS)) if upos is None else upos,
            "dep": np.ones(len(DEPS)) if dep is None else dep, "link": np.ones(len(LINKS)) if link is None else link,
            "lex": np.ones(len(LEX)) if lex is None else lex}
def fit(F, P, groups, grid, sweeps=2):
    best = evaluate(F, P)
    for sw in range(sweeps):
        for g in groups:
            for j in range(len(P[g])):
                keep = P[g][j]
                for v in grid[g]:
                    P[g][j] = v; a = evaluate(F, P)
                    if a > best + 1e-9: best, keep = a, v
                P[g][j] = keep
        say(f"   sweep {sw + 1}: {best * 100:.2f}")
    return P, best

# ---------------------------------------------------------------- the drafted and the data-driven role tables
HAND_U = {"PROPN": 1, "NOUN": .9, "VERB": .8, "ADJ": .7, "NUM": .7, "ADV": .5, "PRON": .3, "INTJ": .3, "SYM": .3, "AUX": .2, "SCONJ": .2,
          "X": .2, "ADP": .15, "DET": .1, "CCONJ": .1, "PART": .1, "PUNCT": .05}
HAND_D = {"root": 1, "nsubj": .9, "obj": .9, "iobj": .8, "csubj": .8, "ccomp": .8, "compound": .8, "flat": .8, "obl": .7, "nmod": .7,
          "appos": .7, "amod": .6, "conj": .6, "advmod": .5, "aux": .2, "cop": .2, "discourse": .2, "case": .15, "mark": .15,
          "det": .1, "cc": .1, "punct": .05}
def table(names, d, default): return np.array([d.get(n, default) for n in names], float)
def from_role_info(section):
    out = {}; on = False
    for line in open("/vault/Data/LaplaceResearch/semantics/role_info.out"):
        if line.startswith("== "): on = section in line; continue
        if on and line.strip():
            p = line.split(); out[p[0]] = float(p[1])
    return out

if __name__ == "__main__":
    docs = lambda s: int(s[0].split(".")[0][1:])
    A = [s for s in semcor if docs(s) % 2 == 0]; Bs = [s for s in semcor if docs(s) % 2 == 1]
    assert not ({docs(s) for s in A} & {docs(s) for s in Bs})
    obsA = observe(A, gsem)
    rng = np.random.default_rng(7); Bsub = [Bs[k] for k in sorted(rng.choice(len(Bs), size=min(len(Bs), 6000), replace=False))]
    Ftune = features(Bsub, gsem, prior_from(lambda I, n: obsA[0].get(I, 0)), obsA)
    say(f"tuning: {len(Ftune[0]) - 1:,} SemCor instances from odd documents; observations from {len(A):,} even-document sentences")
    GRID = {"strand": [0, .003, .01, .03, .1, .3, 1, 3, 10, 30], "upos": [0, .1, .25, .5, 1, 1.5, 2], "dep": [0, .25, .5, 1, 1.5, 2],
            "link": [0, .25, .5, 1, 2, 4], "lex": [0, .25, .5, 1, 1.5, 2]}
    results = {}
    say(f"prior only: {evaluate(Ftune, params()) * 100:.2f}")
    settings = {"uniform role trust": {}, "hand-drafted role trust": {"upos": table(UPOS, HAND_U, .2), "dep": table(DEPS, HAND_D, .3)}}
    try:
        ui, di = from_role_info("UPOS"), from_role_info("head from the dependent")
        settings["role trust from UD information (E2)"] = {"upos": table(UPOS, ui, .2), "dep": table(DEPS, di, .3)}
    except FileNotFoundError: pass
    for name, fixed in settings.items():
        P = params(strand=(0, 0, 0, 0), **fixed); P, a = fit(Ftune, P, ["strand"], GRID, sweeps=2); results[name] = (P, a)
        say(f"{name}: tuned strands {P['strand'].tolist()} -> {a*100:.2f}")
    P = params(strand=results["uniform role trust"][0]["strand"].copy()); P, a = fit(Ftune, P, ["strand", "upos", "link", "dep", "lex"], GRID, sweeps=2)
    results["fitted role trust"] = (P, a); say(f"fitted: {a*100:.2f}")
    PK = params(strand=(.1, .1, .1, 0)); PK, ak = fit(Ftune, PK, ["upos", "link", "dep", "lex"], GRID, sweeps=2)
    PK["strand"][3] = 0; results["fitted, knowledge only (no co-occurrence)"] = (PK, ak); say(f"knowledge only: {ak*100:.2f}")

    # ---------------------------------------------------------------- test once
    obsAll = observe(semcor, gsem); pr = prior_from(lambda I, n: n)
    Ft = {n: features(s, g, pr, obsAll) for n, (s, g) in tests.items()}
    four = ("senseval2", "senseval3", "semeval2013", "semeval2015")
    print("\n== four test sets (Senseval-2, Senseval-3, SemEval-2013, SemEval-2015), scored once")
    tot = sum(len(Ft[n][0]) - 1 for n in four)
    print(f"   {'setting':<45} {'SemCor tuning':>13} {'test ALL':>9}   " + "  ".join(f"{n:>11}" for n in four) + f"   ({tot:,} instances)")
    prior_only = sum(evaluate(Ft[n], params()) * (len(Ft[n][0]) - 1) for n in four) / tot
    print(f"   {'prior only':<45} {'':>13} {prior_only*100:9.1f}")
    for name, (P, a) in results.items():
        per = {n: evaluate(Ft[n], P) for n in four}; allv = sum(per[n] * (len(Ft[n][0]) - 1) for n in four) / tot
        print(f"   {name:<45} {a*100:13.1f} {allv*100:9.1f}   " + "  ".join(f"{per[n]*100:11.1f}" for n in four))
    P = results["fitted role trust"][0]
    def show(title, names, v):
        m = max(v.max(), 1e-9); print(f"\n== fitted role trust: {title} (normalized to the largest)")
        print("   " + ", ".join(f"{n} {x/m:.2f}" for n, x in sorted(zip(names, v), key=lambda z: -z[1])))
    print("\n== fitted strand weights: " + ", ".join(f"{n} {x}" for n, x in zip(STRANDS, P["strand"])))
    show("UPOS of the context word", UPOS, P["upos"]); show("syntactic link to the target", LINKS, P["link"])
    show("deprel of the context word", DEPS, P["dep"]); show("lexname of the context word's first sense", LEX, P["lex"])
    np.savez("/vault/Data/LaplaceResearch/semantics/wsd_intersect_params.npz", **{f"{k}": v for k, v in P.items()},
             UPOS=UPOS, DEPS=DEPS, LINKS=LINKS, LEX=LEX)
