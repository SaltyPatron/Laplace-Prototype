"""Semantics layer, first witnesses (prototype): WordNet 3.0, CILI, and Open Multilingual Wordnet.

Claims are tuples of real entities, [subject, predicate, object], hashed like a composition (BLAKE3 over the
components' IDs). Every label is content: lemmas, ILI identifiers ("i46360"), predicates ("sense", "lang",
"POS"), language codes ("deu"), POS labels. Missing label entities are created with real IDs, fixed-point
coordinates (centroid of their constituents), and physicality paths.

Consensus is a Glicko-2 standing per claim, updated inline as attestations arrive, first in, first out. Each
attestation is a matchup in which the witness is the opponent: a trusted witness plays with a low deviation
(its result moves the standing a lot), a low-trust witness with a high one. Witnesses in the same lineage
count once per claim; later ones only join the claim's witness set.
"""
import glob, math, os, struct, sys, time, collections, pathlib
import numpy as np, blake3, psycopg2, psycopg2.extras

ROOT = pathlib.Path(__file__).resolve().parent.parent
T0 = np.fromfile(ROOT / "tier0/tier0.bin", dtype=[("id", "V16"), ("m", "<i8", (4,)), ("hilbert", "<u8"), ("rank", "<u4"), ("pad", "<u4")])
TIME0 = time.time()
def say(m): print(f"[{time.time()-TIME0:6.1f}s] {m}", flush=True)

# ---------------------------------------------------------------- content: entities for labels
def comp(ids): return ids[0] if len(ids) == 1 else blake3.blake3(b"".join(ids)).digest()[:16]
def trunc_mean(ms):
    s = [sum(m[d] for m in ms) for d in range(4)]; n = len(ms)
    return [v // n if v >= 0 else -((-v) // n) for v in s]              # exact integer division, truncating toward zero
new_entities = {}                                                          # id -> (coord m, tier, child ids)
def cp_node(ch):
    cp = ord(ch); return T0["id"][cp].tobytes(), [int(v) for v in T0["m"][cp]]
def word(s):
    """A word segment: composition of its codepoints (tier 2)."""
    nodes = [cp_node(c) for c in s]
    if len(nodes) == 1: return nodes[0]
    ids = [n[0] for n in nodes]; nid = comp(ids); m = trunc_mean([n[1] for n in nodes])
    new_entities.setdefault(nid, (m, 2, ids)); return nid, m
def text(s):
    """A label: words and the separators between them (tier 3 when it has more than one segment)."""
    parts, cur = [], ""
    for ch in s:
        if ch.isalnum() or ch in "-'’": cur += ch
        else:
            if cur: parts.append(word(cur)); cur = ""
            parts.append(cp_node(ch))
    if cur: parts.append(word(cur))
    if len(parts) == 1: return parts[0]
    ids = [p[0] for p in parts]; nid = comp(ids); m = trunc_mean([p[1] for p in parts])
    new_entities.setdefault(nid, (m, 3, ids)); return nid, m
LABEL = {}
def label(s):
    if s not in LABEL: LABEL[s] = text(s)[0]
    return LABEL[s]

# ---------------------------------------------------------------- Glicko-2 (one matchup per attestation)
S, TAU = 173.7178, 0.5
def g(p): return 1 / math.sqrt(1 + 3 * p * p / math.pi ** 2)
def glicko(r, rd, vol, ro, rdo, score):
    mu, p, muj, pj = (r - 1500) / S, rd / S, (ro - 1500) / S, rdo / S
    e = 1 / (1 + math.exp(-g(pj) * (mu - muj))); v = 1 / (g(pj) ** 2 * e * (1 - e)); delta = v * g(pj) * (score - e)
    a = math.log(vol * vol); A = a
    f = lambda x: math.exp(x) * (delta ** 2 - p * p - v - math.exp(x)) / (2 * (p * p + v + math.exp(x)) ** 2) - (x - a) / TAU ** 2
    if delta ** 2 > p * p + v: B = math.log(delta ** 2 - p * p - v)
    else:
        k = 1
        while f(a - k * TAU) < 0: k += 1
        B = a - k * TAU
    fA, fB = f(A), f(B)
    for _ in range(60):
        if abs(B - A) < 1e-6: break
        C = A + (A - B) * fA / (fB - fA); fC = f(C)
        if fC * fB <= 0: A, fA = B, fB
        else: fA /= 2
        B, fB = C, fC
    vol2 = math.exp(A / 2); ps = math.sqrt(p * p + vol2 * vol2); p2 = 1 / math.sqrt(1 / ps ** 2 + 1 / v)
    return 1500 + S * (mu + p2 * p2 * g(pj) * (score - e)), max(S * p2, 30.0), vol2   # deviation floor 30

# stock defaults for a new claim, by predicate (the level of attestation)
STOCK = {"lex": (1500.0, 350.0, 0.06), "POS": (1500.0, 250.0, 0.06)}

# ---------------------------------------------------------------- witnesses, in order: most trusted first
WN = "/vault/Data/Wordnet/WordNet-3.0/dict"; OMW = "/vault/Data/OMW/wns"
OEWN = "/vault/Data/.refresh-20260903/OpenEnglishWordNet-2025-plus/english-wordnet-2025-plus.xml.gz"
witnesses = [("Open English WordNet 2025+", "wordnet", 1500.0, 35.0, "oewn"),              # maintained successor: same lineage
             ("Princeton WordNet 3.0", "wordnet", 1500.0, 40.0, None)]            # name, lineage root, opponent rating, deviation (trust), files
proj = sorted({os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{OMW}/*/wn-data-*.tab")} - {"eng"})
for pj in proj: witnesses.append((f"Open Multilingual Wordnet: {pj}", f"omw-{pj}", 1500.0, 90.0, sorted(glob.glob(f"{OMW}/{pj}/wn-data-*.tab"))))
witnesses.append(("Open Multilingual Wordnet: Wiktionary-derived", "wiktionary", 1500.0, 160.0, sorted(glob.glob(f"{OMW}/wikt/wn-wikt-*.tab"))))
witnesses.append(("Open Multilingual Wordnet: CLDR-derived", "cldr", 1500.0, 160.0, sorted(glob.glob(f"{OMW}/cldr/wn-cldr-*.tab"))))

ili = {}
for line in open("/vault/Data/CILI/ili-map-pwn30.tab"):
    i, s = line.split(); ili[s] = i
claims = {}                 # claim id -> [subject, predicate, object]
standing = {}               # claim id -> [rating, deviation, volatility, wins, lineages set, witnesses set]
ordinal = {}                # (claim id, witness) -> the claim's position in the witness's ordered list (sense order)
wn_form = {}                # (lowercased lemma with underscores, synset key) -> lemma entity as written in WordNet's data files
P_SENSE, P_LANG, P_POS = label("sense"), label("lang"), label("POS")
def attest(subject, pred_name, pred, obj, wname, root, opp_r, opp_rd):
    cid = comp([subject, pred, obj])
    if cid not in claims: claims[cid] = (subject, pred, obj); r, rd, vol = STOCK[pred_name]; standing[cid] = [r, rd, vol, 0, set(), set()]
    st = standing[cid]; st[5].add(wname)
    if root in st[4]: return                                                   # same lineage: joins the witness set only
    st[4].add(root); st[0], st[1], st[2] = glicko(st[0], st[1], st[2], opp_r, opp_rd, 1.0); st[3] += 1

n_att = collections.Counter()
for wname, root, opp_r, opp_rd, files in witnesses:
    if files == "oewn":                                                        # Open English WordNet: WN-LMF XML
        import gzip, xml.etree.ElementTree as ET
        syn_ili, syn_pos = {}, {}
        for _, el in ET.iterparse(gzip.open(OEWN)):
            if el.tag == "Synset":
                if el.get("ili", "").startswith("i") and el.get("ili") != "in":
                    syn_ili[el.get("id")] = el.get("ili"); syn_pos[el.get("id")] = "a" if el.get("partOfSpeech") == "s" else el.get("partOfSpeech")
                el.clear()
        for _, el in ET.iterparse(gzip.open(OEWN)):
            if el.tag != "LexicalEntry": continue
            lm = el.find("Lemma"); lemma = label(lm.get("writtenForm")); k = 0
            for sn in el.findall("Sense"):
                sid = sn.get("synset")
                if sid not in syn_ili: continue
                k += 1; I, pos = label(syn_ili[sid]), label(syn_pos[sid])
                attest(lemma, "lex", label("eng"), I, wname, root, opp_r, opp_rd)
                ordinal[(comp([lemma, label("eng"), I]), wname)] = k                  # OEWN lists an entry's senses most prominent first
                attest(lemma, "POS", P_POS, pos, wname, root, opp_r, opp_rd)
                attest(I, "POS", P_POS, pos, wname, root, opp_r, opp_rd); n_att[wname] += 3
            el.clear()
    elif files is None:                                                        # Princeton WordNet: data.* files
        for pos, fn in (("n", "noun"), ("v", "verb"), ("a", "adj"), ("r", "adv")):
            for line in open(f"{WN}/data.{fn}", encoding="latin-1"):
                if line.startswith(" "): continue
                f = line.split(); off, sst, nw = f[0], f[2], int(f[3], 16); key = f"{off}-{'a' if sst == 's' else sst}"
                if key not in ili: continue
                I = label(ili[key])
                attest(I, "POS", P_POS, label(pos), wname, root, opp_r, opp_rd); n_att[wname] += 1   # [ILI, POS, pos]: the concept's part of speech
                for k in range(nw):
                    w = f[4 + 2 * k].split("(")[0]; lemma = label(w.replace("_", " ")); wn_form[(w.lower(), key)] = lemma
                    attest(lemma, "lex", label("eng"), I, wname, root, opp_r, opp_rd)         # [lemma, eng, ILI]
                    attest(lemma, "POS", P_POS, label(pos), wname, root, opp_r, opp_rd); n_att[wname] += 2
    else:
        for fpath in files:
            for line in open(fpath, encoding="utf-8", errors="replace"):
                if line.startswith("#"): continue
                c = line.rstrip("\r\n").split("\t")
                if len(c) < 3 or not (c[1] == "lemma" or c[1].endswith(":lemma")) or not c[2].strip(): continue
                syn = c[0]
                lang = c[1].split(":")[0] if ":" in c[1] else os.path.basename(fpath).rsplit("-", 1)[1].split(".")[0]
                key = syn if not syn.endswith("-s") else syn[:-2] + "-a"
                if key not in ili: continue
                lemma = label(c[2].strip())
                attest(lemma, "lex", label(lang), label(ili[key]), wname, root, opp_r, opp_rd); n_att[wname] += 1
    say(f"{wname:<52} attestations {n_att[wname]:>9,}   claims so far {len(claims):,}")

# sense frequency witness: tagged usages are observations, not attestations. They say how often a sense is meant,
# not whether it is true, so they are counted beside the claim and never folded into its standing.
FREQ = "Princeton WordNet 3.0 sense frequencies"; observed = collections.Counter(); nm = 0
for line in open(f"{WN}/index.sense"):
    key, off, sn, cnt = line.split(); cnt = int(cnt)
    lem, rest = key.split("%"); ss = int(rest.split(":")[0]); k = f"{off}-{'nvara'[ss - 1]}"
    if k not in ili: continue
    L = wn_form.get((lem, k))
    if L is None: continue
    cid = comp([L, label("eng"), label(ili[k])])
    if cid not in standing: continue
    ordinal[(cid, "Princeton WordNet 3.0")] = int(sn)
    if not cnt: continue
    observed[cid] += cnt; nm += cnt; standing[cid][5].add(FREQ)
FW = label(FREQ)                                                               # witnesses are entities too
WID = {w: label(w) for w in {w for _, w in ordinal}}
say(f"{FREQ:<52} observations {nm:>9,} on {len(observed):,} claims")

# ---------------------------------------------------------------- write: new entities, claims, consensus
con = psycopg2.connect(host="/tmp", port=5439, user="laplace", dbname="laplace"); cur = con.cursor()
cur.execute("drop table if exists ordinal, occurrence, consensus, claim, witness_set")
cur.execute("""create table if not exists claim (id bytea primary key, subject bytea not null references entity(id),
                 predicate bytea not null references entity(id), object bytea not null references entity(id));
               create table if not exists consensus (claim bytea primary key references claim(id), rating real, deviation real,
                 volatility real, matches int, witnesses bytea not null);
               create table if not exists occurrence (claim bytea not null references claim(id), witness bytea not null references entity(id),
                 count bigint not null, primary key (claim, witness));
               create table if not exists ordinal (claim bytea not null references claim(id), witness bytea not null references entity(id),
                 position int not null, primary key (claim, witness));
               create table if not exists witness_set (id bytea primary key, members text[] not null);""")
cur.execute("select id from entity where id = any(%s)", ([psycopg2.Binary(k) for k in new_entities],))
existing = {bytes(r[0]) for r in cur.fetchall()}
def ewkb_point(m): return b"\x01" + struct.pack("<I", 0xC0000001) + struct.pack("<4d", *(v / 9007199254740992.0 for v in m))
def id_xyz(i):
    v = int.from_bytes(i, "little"); parts = [v & ((1 << 43) - 1), (v >> 43) & ((1 << 43) - 1), v >> 86]
    return [struct.unpack("<d", struct.pack("<Q", (1021 << 52) | p))[0] for p in parts]
def ewkb_path(ids):
    runs = []
    for i in ids:
        if runs and runs[-1][0] == i: runs[-1][1] += 1
        else: runs.append([i, 1])
    if len(runs) == 1: x, y, z = id_xyz(runs[0][0]); return b"\x01" + struct.pack("<I", 0xC0000001) + struct.pack("<4d", x, y, z, runs[0][1])
    return b"\x01" + struct.pack("<II", 0xC0000002, len(runs)) + b"".join(struct.pack("<4d", *id_xyz(i), r) for i, r in runs)
todo = [(k, v) for k, v in new_entities.items() if k not in existing]
psycopg2.extras.execute_values(cur, "insert into entity (id, coord, tier, hilbert) values %s",
    [(psycopg2.Binary(k), psycopg2.Binary(ewkb_point(m)), t, 0) for k, (m, t, _) in todo], template="(%s, st_geomfromewkb(%s), %s, %s)", page_size=5000)
psycopg2.extras.execute_values(cur, "insert into physicality (entity, path) values %s",
    [(psycopg2.Binary(k), psycopg2.Binary(ewkb_path(ch))) for k, (_, _, ch) in todo], template="(%s, st_geomfromewkb(%s))", page_size=5000)
say(f"label entities: {len(new_entities):,} needed, {len(existing):,} already stored, {len(todo):,} created")
sets = {}
for st in standing.values():
    key = tuple(sorted(st[5]))
    if key not in sets: sets[key] = comp([label(w) for w in key])
psycopg2.extras.execute_values(cur, "insert into witness_set (id, members) values %s",
    [(psycopg2.Binary(i), list(k)) for k, i in sets.items()], page_size=2000)
psycopg2.extras.execute_values(cur, "insert into claim (id, subject, predicate, object) values %s",
    [tuple(psycopg2.Binary(x) for x in (cid, *c)) for cid, c in claims.items()], page_size=10000)
psycopg2.extras.execute_values(cur, "insert into consensus (claim, rating, deviation, volatility, matches, witnesses) values %s",
    [(psycopg2.Binary(cid), st[0], st[1], st[2], st[3], psycopg2.Binary(sets[tuple(sorted(st[5]))])) for cid, st in standing.items()], page_size=10000)
psycopg2.extras.execute_values(cur, "insert into occurrence (claim, witness, count) values %s",
    [(psycopg2.Binary(cid), psycopg2.Binary(FW), n) for cid, n in observed.items()], page_size=10000)
psycopg2.extras.execute_values(cur, "insert into ordinal (claim, witness, position) values %s",
    [(psycopg2.Binary(cid), psycopg2.Binary(WID[w]), n) for (cid, w), n in ordinal.items()], page_size=10000)
cur.execute("create index if not exists claim_subject on claim (subject, predicate); create index if not exists claim_object on claim (object, predicate); analyze claim; analyze consensus;")
con.commit()
say(f"claims {len(claims):,}; witness sets {len(sets):,}; done")
