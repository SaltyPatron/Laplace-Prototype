"""Storage queries against the prototype database. IDs and coordinates are computed client-side from
tier0.bin alone; the database is only asked to find and count."""
import struct, time, collections, pathlib, sys
import numpy as np, blake3, psycopg2

ROOT = pathlib.Path(__file__).resolve().parent.parent
T0 = np.fromfile(ROOT / "tier0/tier0.bin", dtype=[("id", "V16"), ("m", "<i8", (4,)), ("hilbert", "<u8"), ("rank", "<u4"), ("pad", "<u4")])
ATOM = {T0["id"][cp].tobytes(): cp for cp in range(len(T0))}
con = psycopg2.connect(host="/tmp", port=5439, user="laplace", dbname="laplace"); cur = con.cursor()
B = psycopg2.Binary

def q(sql, *a):
    t = time.time(); cur.execute(sql, a); rows = cur.fetchall(); return rows, (time.time() - t) * 1000
def leaf(c): return T0["id"][ord(c)].tobytes()
def word_id(s): ids = [leaf(c) for c in s]; return ids[0] if len(ids) == 1 else blake3.blake3(b"".join(ids)).digest()[:16]
def word_coord(s):
    m = np.array([[int(v) for v in T0["m"][ord(c)]] for c in s], dtype=object).sum(0)
    return [int(v) // len(s) if v >= 0 else -((-int(v)) // len(s)) for v in m]   # truncate toward zero, as the engine does
def parse_path(ewkb):
    b = bytes(ewkb); typ = struct.unpack_from("<I", b, 1)[0]; off = 5; n = 1
    if typ & 0xFF == 2: n = struct.unpack_from("<I", b, off)[0]; off += 4
    out = []
    for i in range(n):
        x, y, z, m = struct.unpack_from("<4d", b, off + 32 * i)
        p = [struct.unpack("<Q", struct.pack("<d", v))[0] & ((1 << 52) - 1) for v in (x, y, z)]
        out.append(((p[0] | p[1] << 43 | p[2] << 86).to_bytes(16, "little"), int(m)))
    return out
_paths = {}
def paths(ids):
    need = [i for i in ids if i not in _paths and i not in ATOM]
    if need:
        for eid, p in q("select entity, st_asewkb(path) from physicality where entity = any(%s)", [B(i) for i in need])[0]:
            _paths[bytes(eid)] = parse_path(p)
    return {i: _paths.get(i) for i in ids}
def text(i, depth=0):
    if i in ATOM: return chr(ATOM[i])
    p = paths([i])[i]
    return "".join(text(c, depth + 1) * r for c, r in p)
def expanded(i): return [c for c, r in paths([i])[i] for _ in range(r)]
def containers(i):
    return q("select entity from physicality where laplace_vertex_ids(st_asewkb(path)) @> array[%s::bytea]", B(i))
def section(t): print(f"\n== {t}")

# 1. lookup and counts
section("lookup by client-computed ID")
for w in ["Holmes", "Watson", "Laplace", "probability", "Xyzzyq"]:
    wid = word_id(w)
    rows, ms = q("select s.parents, s.occurrences from entity e left join entity_stats s on s.id = e.id where e.id = %s", B(wid))
    print(f"  {w:<12} {'found' if rows else 'not stored':<11} " + (f"containers {rows[0][0]:>6,}  occurrences {int(rows[0][1]):>7,}" if rows else "") + f"   ({ms:.1f} ms)")

# 2. containers via GIN on the trajectory geometry
section("containers via GIN on the physicality geometry")
rows, ms = containers(word_id("Holmes"))
print(f"  compositions containing 'Holmes': {len(rows):,}  ({ms:.1f} ms)")

# 3. a phrase as a run inside containers
section("phrase 'Sherlock Holmes' as a run [Sherlock, ' ', Holmes]")
t = time.time()
a, _ = containers(word_id("Sherlock")); b, _ = containers(word_id("Holmes"))
cand = list({bytes(r[0]) for r in a} & {bytes(r[0]) for r in b})
run = [word_id("Sherlock"), leaf(" "), word_id("Holmes")]
hits = []
for c in cand:
    seq = expanded(c)
    k = sum(1 for i in range(len(seq) - 2) if seq[i:i + 3] == run)
    if k: hits.append((c, k))
occ = dict(q("select id, occurrences from entity_stats where id = any(%s)", [B(c) for c, _ in hits])[0]) if hits else {}
total = sum(k * int(occ.get(B(c), 0) or [v for kk, v in occ.items() if bytes(kk) == c][0]) for c, k in hits)
print(f"  candidate containers {len(cand):,}; containers with the run {len(hits):,}; occurrences across the corpus {total:,}  ({(time.time()-t)*1000:.0f} ms)")
for c, _ in hits[:3]: print("   ·", text(c).strip().replace("\n", " ")[:110])

# 4. continuation: what follows 'the capital of'
section("continuations of 'the capital of'")
t = time.time()
sets = [ {bytes(r[0]) for r in containers(word_id(w))[0]} for w in ("capital", "the") ]
cand = list(sets[0] & sets[1])
run = [word_id("the"), leaf(" "), word_id("capital"), leaf(" "), word_id("of"), leaf(" ")]
nxt = collections.Counter()
for c in cand:
    seq = expanded(c)
    for i in range(len(seq) - len(run)):
        if seq[i:i + len(run)] == run: nxt[seq[i + len(run)]] += 1
print(f"  {sum(nxt.values()):,} continuations in {len(cand):,} candidate sentences  ({(time.time()-t)*1000:.0f} ms)")
print("  " + ", ".join(f"{text(k)} ({v})" for k, v in nxt.most_common(12)))

# 5. 4D nearest neighbours through the GiST index
section("nearest word segments to 'king' in 4D (GiST, <<->>)")
x, y, z, w = [v / 9007199254740992.0 for v in word_coord("king")]
rows, ms = q("""select id, coord <<->> st_makepoint(%s, %s, %s, %s) as d from entity
                where tier = 2 order by coord <<->> st_makepoint(%s, %s, %s, %s) limit 16""", x, y, z, w, x, y, z, w)
print(f"  ({ms:.1f} ms)  " + ", ".join(f"{text(bytes(i))} {d:.3f}" for i, d in rows))

# 6. most frequent word segments
section("most frequent word segments (occurrences across the corpus)")
rows, ms = q("""select e.id, s.occurrences from entity_stats s join entity e on e.id = s.id
                where e.tier = 2 order by s.occurrences desc limit 20""")
print(f"  ({ms:.0f} ms)  " + ", ".join(f"{text(bytes(i))!r} {int(o):,}" for i, o in rows))
