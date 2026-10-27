"""Truths cluster, lies scatter (prototype measurement). For languages with an expert-built wordnet in Open
Multilingual Wordnet, that wordnet is the gold for its language. Every lexicalization [lemma, language, concept] from
the derived witnesses (Wiktionary-derived, CLDR-derived) is scored against it, on concepts the gold covers. Reports
each witness's precision and earned trust t = 2p - 1, precision by the number of independent witnesses that agree, and
whether wrong lexicalizations agree with each other as often as right ones do."""
import glob, os, collections
OMW = "/vault/Data/OMW/wns"
GOLD = {"ita": "ita", "pol": "pol", "fin": "fin", "spa": "mcr", "cat": "mcr", "nld": "nld", "dan": "dan", "arb": "arb",
        "bul": "bul", "jpn": "jpn", "heb": "heb", "por": "por", "ell": "ell", "swe": "swe", "hrv": "hrv", "isl": "isl"}
def read(path, lang_default):
    out = collections.defaultdict(set)                          # (lang, synset) -> lemmas
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("#"): continue
        c = line.rstrip("\r\n").split("\t")
        if len(c) < 3 or not (c[1] == "lemma" or c[1].endswith(":lemma")) or not c[2].strip(): continue
        lang = c[1].split(":")[0] if ":" in c[1] else lang_default
        syn = c[0][:-2] + "-a" if c[0].endswith("-s") else c[0]
        out[(lang, syn)].add(c[2].strip())
    return out
gold = collections.defaultdict(set)
for lang, pj in GOLD.items():
    for f in glob.glob(f"{OMW}/{pj}/wn-data-*.tab"):
        for (lg, syn), ls in read(f, os.path.basename(f).rsplit("-", 1)[1].split(".")[0]).items():
            if lg == lang: gold[(lg, syn)] |= ls
nodia = collections.defaultdict(set)                            # Arabic WordNet without diacritics: the same witness, other form
for k, ls in read(f"{OMW}/arb/wn-nodia-arb.tab", "arb").items(): nodia[k] |= ls
wit = {}
for name, pat in (("Wiktionary-derived", "wikt/wn-wikt-{}.tab"), ("CLDR-derived", "cldr/wn-cldr-{}.tab")):
    d = collections.defaultdict(set)
    for lang in GOLD:
        f = f"{OMW}/{pat.format(lang)}"
        if os.path.exists(f):
            for k, ls in read(f, lang).items(): d[k] |= ls
    wit[name] = d
iwn = collections.defaultdict(set)                              # ItalWordNet: a second, independent expert Italian wordnet
for f in glob.glob(f"{OMW}/iwn/wn-data-*.tab"):
    for k, ls in read(f, "ita").items(): iwn[k] |= ls
wit["ItalWordNet (expert)"] = iwn

print("== each witness against the expert wordnet of its language, on concepts the gold covers")
print(f"   {'witness':<22} {'claims scored':>13} {'precision':>9} {'earned trust 2p-1':>18}   (case-insensitive precision)")
for name, d in wit.items():
    n = ok = okc = 0
    for k, ls in d.items():
        if k not in gold: continue
        g = gold[k]; gl = {x.lower() for x in g}
        for l in ls: n += 1; ok += l in g; okc += l.lower() in gl
    print(f"   {name:<22} {n:>13,} {ok/n*100:>8.1f}% {2*ok/n-1:>18.2f}   ({okc/n*100:.1f}%)")

print("\n== precision by the number of independent derived witnesses asserting the lexicalization (Wiktionary, CLDR)")
by = collections.Counter(); byok = collections.Counter(); per_lang = collections.defaultdict(collections.Counter)
keys = set(wit["Wiktionary-derived"]) | set(wit["CLDR-derived"])
for k in keys:
    if k not in gold: continue
    ls = wit["Wiktionary-derived"].get(k, set()) | wit["CLDR-derived"].get(k, set())
    for l in ls:
        m = (l in wit["Wiktionary-derived"].get(k, ())) + (l in wit["CLDR-derived"].get(k, ()))
        by[m] += 1; c = l in gold[k]; byok[m] += c; per_lang[k[0]][(m, c)] += 1
for m in sorted(by): print(f"   {m} witness{'es' if m > 1 else ''}: {by[m]:>7,} lexicalizations, {byok[m]/by[m]*100:5.1f}% correct")
right = sum(v for (m, c), v in sum(per_lang.values(), collections.Counter()).items() if c)
wrong = sum(v for (m, c), v in sum(per_lang.values(), collections.Counter()).items() if not c)
rc = sum(v for (m, c), v in sum(per_lang.values(), collections.Counter()).items() if c and m == 2)
wc = sum(v for (m, c), v in sum(per_lang.values(), collections.Counter()).items() if not c and m == 2)
print(f"   right lexicalizations agreed on by both: {rc/right*100:.1f}%;  wrong ones agreed on by both: {wc/wrong*100:.1f}%")
print("   per language (1 witness correct % / 2 witnesses correct %):")
for lg in sorted(per_lang):
    c = per_lang[lg]; a1 = c[(1, True)] + c[(1, False)]; a2 = c[(2, True)] + c[(2, False)]
    if a2 >= 30: print(f"     {lg}: {c[(1, True)]/a1*100:5.1f}% of {a1:>6,}   {c[(2, True)]/a2*100:5.1f}% of {a2:>5,}")

print("\n== Italian: two independent expert wordnets, MultiWordNet and ItalWordNet, on concepts both cover")
both = [k for k in gold if k[0] == "ita" and k in iwn]; agree = sum(len(gold[k] & iwn[k]) for k in both)
print(f"   concepts {len(both):,}; MultiWordNet lemmas {sum(len(gold[k]) for k in both):,}; ItalWordNet lemmas {sum(len(iwn[k]) for k in both):,}; shared {agree:,}")
for name in ("Wiktionary-derived", "CLDR-derived"):
    d = wit[name]; n = c0 = c1 = c2 = 0
    for k in both:
        for l in d.get(k, ()):
            n += 1; s = (l in gold[k]) + (l in iwn[k]); c0 += s == 0; c1 += s == 1; c2 += s == 2
    if n: print(f"   {name}: {n:,} lexicalizations; confirmed by both experts {c2/n*100:.1f}%, by one {c1/n*100:.1f}%, by neither {c0/n*100:.1f}%")

print("\n== Arabic: the expert wordnet writes lemmas with diacritics; scored against its undiacritized form as well")
d = wit["Wiktionary-derived"]; n = a = b = 0
for k, ls in d.items():
    if k[0] != "arb" or k not in gold: continue
    for l in ls: n += 1; a += l in gold[k]; b += l in gold[k] or l in nodia.get(k, ())
print(f"   Wiktionary-derived: {n:,} lexicalizations; {a/n*100:.1f}% match the diacritized form, {b/n*100:.1f}% match either form")
