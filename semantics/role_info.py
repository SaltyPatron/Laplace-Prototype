"""Role trust from data (prototype measurement): how much information each part of speech and each dependency
relation carries, across every language in Universal Dependencies v2.17. Sentences are split in halves by parity; counts
come from one half and are scored on the other, so nothing is measured on the data that produced it.
  UPOS:   mean held-out surprisal (bits) of a word with that tag, under a unigram model of its language.
  deprel: held-out information gain (bits) about the dependent's lemma from knowing its head's lemma and the relation,
          and about the head from the dependent: log2 p(x | other, rel) - log2 p(x), Dirichlet-smoothed toward p(x).
Each language is measured on its own, normalized by its largest value, and the languages are averaged, so large
treebanks do not dominate. English (all English treebanks) is reported beside the average."""
import glob, math, os, collections, sys, time
UD = "/vault/Data/UD-Treebanks/ud-treebanks-v2.17"; ALPHA = 5.0; MIN_TOKENS = 20000; T0 = time.time()
bylang = collections.defaultdict(list)
for d in sorted(glob.glob(f"{UD}/UD_*")):
    bylang[os.path.basename(d)[3:].split("-")[0]] += glob.glob(f"{d}/*.conllu")
def sents(files):
    for f in files:
        s = []
        for line in open(f, encoding="utf-8"):
            if line.startswith("#"): continue
            if not line.strip():
                if s: yield s
                s = []; continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 8 or not c[0].isdigit(): continue                    # skip ranges and empty nodes
            lem = c[2] if c[2] != "_" else c[1]
            s.append((int(c[0]), lem.lower(), c[3], int(c[6]) if c[6].isdigit() else 0, c[7].split(":")[0]))
        if s: yield s
def measure(files):
    half = [[], []]
    for i, s in enumerate(sents(files)): half[i % 2].append(s)
    ntok = sum(len(s) for h in half for s in h)
    if ntok < MIN_TOKENS: return None
    up = collections.defaultdict(list); dep_gain = collections.defaultdict(list); head_gain = collections.defaultdict(list)
    for a, b in ((0, 1), (1, 0)):                                          # count on one half, score on the other
        uni = collections.Counter(); N = 0; hd = collections.Counter(); hdn = collections.Counter()
        dh = collections.Counter(); dhn = collections.Counter()
        for s in half[a]:
            for i, l, u, h, r in s:
                uni[l] += 1; N += 1
                if h:
                    hl = s[h - 1][1]; hd[(hl, r, l)] += 1; hdn[(hl, r)] += 1; dh[(l, r, hl)] += 1; dhn[(l, r)] += 1
        V = len(uni) + 1
        P = lambda x: (uni[x] + 1) / (N + V)
        for s in half[b]:
            for i, l, u, h, r in s:
                up[u].append(-math.log2(P(l)))
                if h:
                    hl = s[h - 1][1]
                    pd = P(l); dep_gain[r].append(math.log2((hd[(hl, r, l)] + ALPHA * pd) / (hdn[(hl, r)] + ALPHA) / pd))
                    ph = P(hl); head_gain[r].append(math.log2((dh[(l, r, hl)] + ALPHA * ph) / (dhn[(l, r)] + ALPHA) / ph))
    mean = lambda d: {k: sum(v) / len(v) for k, v in d.items() if len(v) >= 30}
    return ntok, mean(up), mean(dep_gain), mean(head_gain)
res = {}
for k, (lang, files) in enumerate(sorted(bylang.items())):
    r = measure(files)
    if r: res[lang] = r
    if (k + 1) % 25 == 0: print(f"[{time.time()-T0:5.0f}s] {k+1}/{len(bylang)} languages, {len(res)} measured", file=sys.stderr, flush=True)
def norm(d):
    m = max(d.values()); return {k: max(0.0, v) / m for k, v in d.items()} if m > 0 else {}
def avg(idx):
    acc = collections.defaultdict(list)
    for lang, r in res.items():
        for k, v in norm(r[idx]).items(): acc[k].append(v)
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items() if len(v) >= 10}
print(f"{len(res)} languages with at least {MIN_TOKENS:,} tokens, {sum(r[0] for r in res.values()):,} tokens\n")
en = res.get("English")
for idx, title, unit in ((1, "UPOS: held-out surprisal of the word", "bits"),
                         (2, "deprel: information about the dependent from the head", "bits"),
                         (3, "deprel: information about the head from the dependent", "bits")):
    a = avg(idx); e = en[idx] if en else {}; en_n = norm(e) if e else {}
    print(f"== {title}  (normalized to the language's largest; average over languages; English raw {unit} and normalized)")
    for k, (v, n) in sorted(a.items(), key=lambda x: -x[1][0]):
        print(f"   {k:<11} {v:5.2f}  ({n:>3} languages)   English {e.get(k, float('nan')):6.2f} {en_n.get(k, float('nan')):5.2f}")
    print()
