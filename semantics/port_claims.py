"""Bring the prototype's witnessed claims (port 5439, bytea IDs) into the engine database (port 5432, uuid IDs).
IDs are content hashes, so they carry over unchanged. Label entities (lemmas, ILIs, predicates, witness names) the
engine lacks are copied with their coordinates and paths; Hilbert keys are recomputed in the engine's order-preserving
form. The prototype kept standings only, so the ledger is rebuilt from each claim's witness set: one attestation per
witness, in the order the witnesses were played (most trusted first)."""
import io, sys, time, uuid, psycopg2
sys.argv = [sys.argv[0]]
exec(open("/repos/src/Laplace-Prototype/semantics/witnesses.py").read().split("# ---------------------------------------------------------------- Glicko-2")[0])
T = time.time(); say = lambda m: print(f"[{time.time()-T:6.1f}s] {m}", flush=True)
src = psycopg2.connect(host="/tmp", port=5439, user="laplace", dbname="laplace")
dst = psycopg2.connect(host="localhost", port=5432, user="laplace", dbname="laplace"); dc = dst.cursor()
U = lambda b: str(uuid.UUID(bytes=bytes(b)))
def stream(sql, n=200000):
    c = src.cursor(name=f"s{time.time_ns()}"); c.itersize = n; c.execute(sql); return c
def copy(table, cols, rows):
    buf = io.StringIO(); k = 0
    for r in rows: buf.write("\t".join(r) + "\n"); k += 1
    buf.seek(0); dc.copy_expert(f"COPY {table} ({cols}) FROM STDIN", buf); return k

# witnesses: names from the prototype's witness sets; trust as its opponent deviations implied (see witnesses.py)
TRUST = {"Open English WordNet 2025+": 0.95, "Princeton WordNet 3.0": 0.94, "Open Multilingual Wordnet: Wiktionary-derived": 0.52,
         "Open Multilingual Wordnet: CLDR-derived": 0.38, "Princeton WordNet 3.0 sense frequencies": 0.94}
LIN = {"Open English WordNet 2025+": "wordnet", "Princeton WordNet 3.0": "wordnet", "Princeton WordNet 3.0 sense frequencies": "wordnet"}
sc = src.cursor(); sc.execute("select id, members from witness_set"); sets = {bytes(i): m for i, m in sc.fetchall()}
names = sorted({m for ms in sets.values() for m in ms})
wid = {n: label(n) for n in names}; lin = {n: label(LIN.get(n, n)) for n in names}
dc.execute("TRUNCATE claim, witness, attestation, standing, occurrence, ordinal")
copy("witness", "id, name, lineage, trust, kind", ([U(wid[n]), n.replace("\t", " "), U(lin[n]), str(TRUST.get(n, 0.9 if n.startswith("Open Multilingual Wordnet") else 0.6)), "curated"] for n in names))
say(f"{len(names)} witnesses")

# label entities the engine lacks
sc.execute("select distinct x from (select subject x from claim union select predicate from claim union select object from claim) t")
need = {bytes(r[0]) for r in sc.fetchall()} | set(wid.values()) | set(lin.values())
need = list(need); have = set()
for k in range(0, len(need), 200000):                                      # a set query per chunk of IDs
    dc.execute("select id from entity where id = any(%s::uuid[])", ([U(x) for x in need[k:k + 200000]],))
    have |= {uuid.UUID(r[0]).bytes for r in dc.fetchall()}
miss = [x for x in need if x not in have]
say(f"label entities: {len(need):,} referenced, {len(miss):,} to copy")
dc.execute("CREATE TEMP TABLE e_stage (id uuid, tier smallint, coord geometry, path geometry)")
for k in range(0, len(miss), 100000):
    chunk = miss[k:k + 100000]
    sc.execute("select e.id, e.tier, st_asewkb(e.coord), st_asewkb(p.path) from entity e join physicality p on p.entity = e.id where e.id = any(%s)", ([psycopg2.Binary(x) for x in chunk],))
    copy("e_stage", "id, tier, coord, path", ([U(i), str(t), bytes(c).hex(), bytes(pth).hex()] for i, t, c, pth in sc.fetchall()))
dc.execute("""INSERT INTO entity (id, tier, coord, hilbert) SELECT id, tier, coord, laplace_hilbert4(coord) # (-9223372036854775808)::bigint FROM e_stage;
              INSERT INTO physicality (entity, tier, hilbert, path) SELECT id, tier, laplace_hilbert4(coord) # (-9223372036854775808)::bigint, path FROM e_stage""")
say("label entities copied")

n = copy("claim", "id, subject, predicate, object", ([U(i), U(s), U(p), U(o)] for i, s, p, o in stream("select id, subject, predicate, object from claim")))
say(f"{n:,} claims")
ORDER = {n_: k for k, n_ in enumerate(["Open English WordNet 2025+", "Princeton WordNet 3.0"])}
def standings():
    for c, r, rd, vol, m, ws in stream("select claim, rating, deviation, volatility, matches, witnesses from consensus"):
        mem = sets[bytes(ws)]; yield [U(c), str(r), str(rd), str(vol), str(m), "{" + ",".join(U(wid[x]) for x in mem) + "}"]
n = copy("standing", "claim, rating, deviation, volatility, matches, witnesses", standings()); say(f"{n:,} standings")
def ledger():
    for c, ws in stream("select claim, witnesses from consensus"):
        mem = sorted(sets[bytes(ws)], key=lambda x: (ORDER.get(x, 5), x))
        for x in mem:
            if x == "Princeton WordNet 3.0 sense frequencies": continue          # observations, not attestations
            yield [U(c), U(wid[x]), "\\N", "1", "\\N"]
n = copy("attestation", "claim, witness, condition, score, z", ledger()); say(f"{n:,} ledger rows rebuilt")
n = copy("occurrence", "claim, witness, count", ([U(c), U(w), str(k)] for c, w, k in stream("select claim, witness, count from occurrence"))); say(f"{n:,} occurrences")
n = copy("ordinal", "claim, witness, position", ([U(c), U(w), str(p)] for c, w, p in stream("select claim, witness, position from ordinal"))); say(f"{n:,} ordinals")
dst.commit(); dc.execute("ANALYZE claim; ANALYZE standing; ANALYZE attestation"); dst.commit(); say("done")
