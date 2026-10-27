"""WordNet's semantic relations as claims between concepts: [ILI_a, relation, ILI_b], witnessed by Open English
WordNet 2025+ and Princeton WordNet 3.0 (one lineage: a claim both attest gets one matchup and both witnesses),
added to the existing claim and consensus tables. Relation names are content (the words WordNet's
documentation uses), so no strings are invented."""
import sys, collections, psycopg2, psycopg2.extras
sys.argv = [sys.argv[0]]
exec(open("witnesses.py").read().split("# ---------------------------------------------------------------- witnesses, in order")[0])
POINTER = {"@": "hypernym", "@i": "instance hypernym", "~": "hyponym", "~i": "instance hyponym", "#m": "member holonym",
           "#s": "substance holonym", "#p": "part holonym", "%m": "member meronym", "%s": "substance meronym",
           "%p": "part meronym", "!": "antonym", "*": "entailment", ">": "cause", "&": "similar to", "^": "also see",
           "+": "derivationally related form", "=": "attribute", ";c": "domain of synset topic", "-c": "member of domain topic",
           "$": "verb group", "<": "participle of verb", "\\\\": "pertainym"}
WN = "/vault/Data/Wordnet/WordNet-3.0/dict"
ili = {}
for line in open("/vault/Data/CILI/ili-map-pwn30.tab"):
    i, s_ = line.split(); ili[s_] = i
OEWN_REL = {"hypernym": "hypernym", "hyponym": "hyponym", "instance_hypernym": "instance hypernym", "instance_hyponym": "instance hyponym",
            "mero_member": "member meronym", "holo_member": "member holonym", "mero_part": "part meronym", "holo_part": "part holonym",
            "mero_substance": "substance meronym", "holo_substance": "substance holonym", "similar": "similar to", "also": "also see",
            "attribute": "attribute", "entails": "entailment", "causes": "cause", "domain_topic": "domain of synset topic",
            "has_domain_topic": "member of domain topic"}                     # OEWN names that WordNet 3.0 also uses; others as written
rels = {}; counts = collections.Counter()                                      # claim id -> [a, p, b, witnesses]
def add(src, rel, dst, w):
    a, p_, b = label(src), label(rel), label(dst); cid = comp([a, p_, b])
    if cid not in rels: rels[cid] = [a, p_, b, set()]; counts[rel] += 1
    rels[cid][3].add(w)
import gzip, xml.etree.ElementTree as ET
OEWN = "/vault/Data/.refresh-20260903/OpenEnglishWordNet-2025-plus/english-wordnet-2025-plus.xml.gz"; syn_ili = {}; pend = []
for _, el in ET.iterparse(gzip.open(OEWN)):
    if el.tag == "Synset":
        if el.get("ili", "").startswith("i") and el.get("ili") != "in": syn_ili[el.get("id")] = el.get("ili")
        pend += [(el.get("id"), r.get("relType"), r.get("target")) for r in el.findall("SynsetRelation")]; el.clear()
for a_, rt, b_ in pend:
    if a_ in syn_ili and b_ in syn_ili and syn_ili[a_] != syn_ili[b_]:
        add(syn_ili[a_], OEWN_REL.get(rt, rt.replace("_", " ")), syn_ili[b_], "Open English WordNet 2025+")
say(f"Open English WordNet 2025+ relation claims: {len(rels):,}")
W = "Princeton WordNet 3.0"
for fn in ("noun", "verb", "adj", "adv"):
    for line in open(f"{WN}/data.{fn}", encoding="latin-1"):
        if line.startswith(" "): continue
        f = line.split("|")[0].split(); off, sst, nw = f[0], f[2], int(f[3], 16)
        src = ili.get(f"{off}-{'a' if sst == 's' else sst}"); i = 4 + 2 * nw; npt = int(f[i]); i += 1
        for k in range(npt):
            sym, toff, tpos = f[i], f[i + 1], f[i + 2]; i += 4
            dst = ili.get(f"{toff}-{'a' if tpos == 's' else tpos}"); rel = POINTER.get(sym)
            if not (src and dst and rel) or src == dst: continue
            add(src, rel, dst, W)
rows = []; sets = {}
for cid, (a, p, b, ws) in rels.items():                                        # one lineage: one matchup, by its most trusted member
    r, rd, vol = glicko(1500.0, 250.0, 0.06, 1500.0, 35.0 if "Open English WordNet 2025+" in ws else 40.0, 1.0)
    k = tuple(sorted(ws))
    if k not in sets: sets[k] = comp([label(w) for w in k])
    rows.append((cid, a, p, b, r, rd, vol, sets[k]))
say(f"relation claims: {len(rows):,}  " + ", ".join(f"{k} {v:,}" for k, v in counts.most_common(8)))
con = psycopg2.connect(host="/tmp", port=5439, user="laplace", dbname="laplace"); cur = con.cursor(); B = psycopg2.Binary
cur.execute("select id from entity where id = any(%s)", ([B(k) for k in new_entities],)); have = {bytes(r[0]) for r in cur.fetchall()}
todo = [(k, v) for k, v in new_entities.items() if k not in have]
import struct
def ewkb_point(m): return b"\x01" + struct.pack("<I", 0xC0000001) + struct.pack("<4d", *(v / 9007199254740992.0 for v in m))
def id_xyz(i):
    v = int.from_bytes(i, "little"); pr = [v & ((1 << 43) - 1), (v >> 43) & ((1 << 43) - 1), v >> 86]
    return [struct.unpack("<d", struct.pack("<Q", (1021 << 52) | x))[0] for x in pr]
def ewkb_path(ids):
    runs = []
    for i in ids:
        if runs and runs[-1][0] == i: runs[-1][1] += 1
        else: runs.append([i, 1])
    if len(runs) == 1: return b"\x01" + struct.pack("<I", 0xC0000001) + struct.pack("<4d", *id_xyz(runs[0][0]), runs[0][1])
    return b"\x01" + struct.pack("<II", 0xC0000002, len(runs)) + b"".join(struct.pack("<4d", *id_xyz(i), r) for i, r in runs)
psycopg2.extras.execute_values(cur, "insert into entity (id, coord, tier, hilbert) values %s",
    [(B(k), B(ewkb_point(m)), t, 0) for k, (m, t, _) in todo], template="(%s, st_geomfromewkb(%s), %s, %s)")
psycopg2.extras.execute_values(cur, "insert into physicality (entity, path) values %s",
    [(B(k), B(ewkb_path(ch))) for k, (_, _, ch) in todo], template="(%s, st_geomfromewkb(%s))")
# ask first: which of these IDs are already recorded; only the rest are written, so nothing can conflict
cur.execute("select id from witness_set where id = any(%s)", ([B(i) for i in sets.values()],)); have_ws = {bytes(r[0]) for r in cur.fetchall()}
psycopg2.extras.execute_values(cur, "insert into witness_set (id, members) values %s", [(B(i), list(k)) for k, i in sets.items() if i not in have_ws])
cur.execute("select id from claim where id = any(%s)", ([B(r[0]) for r in rows],)); have_c = {bytes(r[0]) for r in cur.fetchall()}
rows = [r for r in rows if r[0] not in have_c]
psycopg2.extras.execute_values(cur, "insert into claim (id, subject, predicate, object) values %s",
    [(B(c), B(a), B(p), B(b)) for c, a, p, b, *_ in rows], page_size=10000)
psycopg2.extras.execute_values(cur, "insert into consensus (claim, rating, deviation, volatility, matches, witnesses) values %s",
    [(B(c), r, rd, vol, 1, B(ws)) for c, a, p, b, r, rd, vol, ws in rows], page_size=10000)
cur.execute("analyze claim; analyze consensus"); con.commit(); say(f"written; {len(todo)} label entities created")
