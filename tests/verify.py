"""Verify the Laplace storage prototype against the database. Prints progress and a PASS/FAIL line per check.

Client-side IDs are computed from tier0.bin alone (no database): leaf = tier-0 record,
composition = BLAKE3(concatenated child IDs, repeats included)[:16], one-child collapse.
"""
import struct, sys, time, random, subprocess, tempfile, pathlib
import numpy as np, blake3, psycopg2

ROOT = pathlib.Path(__file__).resolve().parent.parent
T0 = np.fromfile(ROOT / "tier0/tier0.bin", dtype=[("id", "V16"), ("m", "<i8", (4,)), ("hilbert", "<u8"), ("rank", "<u4"), ("pad", "<u4")])
ATOM_BY_ID = {T0["id"][cp].tobytes(): cp for cp in range(len(T0))}
LIMIT = 1 << 106
t_start = time.time()
results = []

def say(msg): print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)
def check(name, ok, detail=""):
    results.append((name, ok)); say(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

def leaf(ch): return T0["id"][ord(ch)].tobytes()
def comp(ids):
    return ids[0] if len(ids) == 1 else blake3.blake3(b"".join(ids)).digest()[:16]
def word(s): return comp([leaf(c) for c in s])

def parse_path(ewkb):
    """-> list of (child_id, run) from a POINT ZM / LINESTRING ZM EWKB."""
    b = bytes(ewkb); typ = struct.unpack_from("<I", b, 1)[0]; off = 5 + (4 if typ & 0x20000000 else 0)
    n = 1
    if typ & 0xFF == 2: n = struct.unpack_from("<I", b, off)[0]; off += 4
    out = []
    for i in range(n):
        x, y, z, m = struct.unpack_from("<4d", b, off + 32 * i)
        parts = [struct.unpack("<Q", struct.pack("<d", v))[0] & ((1 << 52) - 1) for v in (x, y, z)]
        v = parts[0] | (parts[1] << 43) | (parts[2] << 86)
        out.append((v.to_bytes(16, "little"), int(m)))
    return out

con = psycopg2.connect(host="/tmp", port=5439, user="laplace", dbname="laplace"); cur = con.cursor()
def q(sql, *a): cur.execute(sql, a); return cur.fetchall()

# 1. counts and uniqueness
ne = q("select count(*) from entity")[0][0]; nph = q("select count(*) from physicality")[0][0]
check("every entity has exactly one physicality (counts match)", ne == nph, f"entity {ne:,} physicality {nph:,}")
nd = q("select count(distinct id) from entity")[0][0]
check("entity IDs are unique", nd == ne, f"distinct {nd:,}")
na = q("select count(*) from entity where tier = 0")[0][0]
check("all 1,114,112 codepoints recorded at tier 0", na == 1114112, f"{na:,}")

# 2. the wall, exactly: every coordinate is m / 2^53; check sum(m^2) <= 2^106 in integer arithmetic
say("checking the wall on every entity (exact integer arithmetic)...")
cur2 = con.cursor("wall"); cur2.itersize = 200000
cur2.execute("select st_x(coord), st_y(coord), st_z(coord), st_m(coord), tier from entity")
over, on_wall_comp, n = 0, 0, 0
for x, y, z, w, tier in cur2:
    s = sum(int(v * 9007199254740992.0) ** 2 for v in (x, y, z, w)); n += 1
    over += s > LIMIT
    if tier > 0 and s == LIMIT: on_wall_comp += 1
cur2.close()
check("nothing falls outside the wall (exact)", over == 0, f"{n:,} entities, {over} outside")

# 3. pure repeats [n,n,...] sit exactly on n's point (n a codepoint or a composition)
reps = q("""select e.id, st_asewkb(p.path), st_x(e.coord), st_y(e.coord), st_z(e.coord), st_m(e.coord)
            from entity e join physicality p on p.entity = e.id
            where e.tier > 0 and geometrytype(p.path) = 'POINT'""")
fixed = lambda vals: tuple(int(v * 9007199254740992.0) for vals_ in [vals] for v in vals_)
child_coord = {}
kids = [parse_path(path)[0][0] for _, path, *_ in reps]
comp_kids = [k for k in kids if k not in ATOM_BY_ID]
for k in range(0, len(comp_kids), 20000):
    for cid, x, y, z, w in q("select id, st_x(coord), st_y(coord), st_z(coord), st_m(coord) from entity where id = any(%s)",
                             [psycopg2.Binary(c) for c in comp_kids[k:k + 20000]]):
        child_coord[bytes(cid)] = fixed((x, y, z, w))
exact = n_atom = n_comp = 0
for eid, path, x, y, z, w in reps:
    (child, run), = parse_path(path)
    if child in ATOM_BY_ID: target = tuple(int(v) for v in T0["m"][ATOM_BY_ID[child]]); n_atom += 1
    else: target = child_coord[child]; n_comp += 1
    exact += target == fixed((x, y, z, w))
check("pure repeats [n,n,…] sit exactly on n's point", exact == len(reps),
      f"{exact}/{len(reps)} (repeats of codepoints {n_atom}, of compositions {n_comp})")

# 4. identity and tiers from the database alone (random sample)
say("recomputing IDs and tiers for a random sample of compositions...")
sample = q("""select e.id, e.tier, st_asewkb(p.path) from entity e join physicality p on p.entity = e.id
              where e.tier > 0 order by random() limit 20000""")
tier_of = {}
child_ids = {c for _, _, path in sample for c, _ in parse_path(path)}
for chunk in [list(child_ids)[i:i + 5000] for i in range(0, len(child_ids), 5000)]:
    for cid, t in q("select id, tier from entity where id = any(%s)", [psycopg2.Binary(c) for c in chunk]): tier_of[bytes(cid)] = t
id_ok = tier_ok = 0
for eid, tier, path in sample:
    verts = parse_path(path)
    expanded = [c for c, r in verts for _ in range(r)]
    id_ok += comp(expanded) == bytes(eid)
    tier_ok += all(tier > tier_of[c] for c, _ in verts)
check("IDs recompute from children (pure content, repeats included)", id_ok == len(sample), f"{id_ok}/{len(sample)}")
check("tiers only go up (every child below its parent)", tier_ok == len(sample), f"{tier_ok}/{len(sample)}")

# 4b. depth law: every composition is at least as deep as the average depth of its own constituents
#     (triangle inequality; truncation toward zero only deepens). Checked exactly with fixed-point integers.
say("checking the depth law on the sample ...")
coords = {}
ids_needed = list({c for _, _, path in sample for c, _ in parse_path(path) if c not in ATOM_BY_ID} | {bytes(e) for e, _, _ in sample})
for k in range(0, len(ids_needed), 20000):
    for cid, x, y, z, w in q("select id, st_x(coord), st_y(coord), st_z(coord), st_m(coord) from entity where id = any(%s)",
                             [psycopg2.Binary(c) for c in ids_needed[k:k + 20000]]):
        coords[bytes(cid)] = [int(v * 9007199254740992.0) for v in (x, y, z, w)]
def m_of(i): return [int(v) for v in T0["m"][ATOM_BY_ID[i]]] if i in ATOM_BY_ID else coords[i]
from decimal import Decimal, getcontext
getcontext().prec = 80
def norm(m): return sum(Decimal(v) * v for v in m).sqrt()
law_ok = 0
for eid, _, path in sample:
    kids = [(c, r) for c, r in parse_path(path)]
    k = sum(r for _, r in kids)
    # compare k*|parent| with sum(r*|child|) (no division); square roots are irrational, so allow a
    # tolerance of 1e-40 grid units, far below the 1-unit resolution of the fixed-point grid
    law_ok += k * norm(coords[bytes(eid)]) <= sum(norm(m_of(c)) * r for c, r in kids) + Decimal("1e-40")
check("depth law: each composition is at least as deep as its constituents' average", law_ok == len(sample), f"{law_ok}/{len(sample)}")

# 5. recompose a whole book from the database and compare with the source
def recompose(trunk):
    memo = {}
    def fetch(ids):
        need = [i for i in ids if i not in memo and i not in ATOM_BY_ID]
        for k in range(0, len(need), 20000):
            for eid, path in q("select entity, st_asewkb(path) from physicality where entity = any(%s)",
                               [psycopg2.Binary(i) for i in need[k:k + 20000]]):
                memo[bytes(eid)] = parse_path(path)
    level = [trunk]
    while level:
        fetch(level)
        level = list({c for i in level if i in memo for c, _ in memo[i]} - set(memo) - set(ATOM_BY_ID))
    out = []
    def walk(i):
        if i in ATOM_BY_ID: out.append(chr(ATOM_BY_ID[i])); return
        for c, r in memo[i]:
            for _ in range(r): walk(c)
    sys.setrecursionlimit(100000); walk(trunk)
    return "".join(out).encode("utf-8", "surrogatepass")
for name in ("alice.txt", "sherlock.txt"):
    trunk, digest = q("select trunk, content_blake3 from source where origin like %s", f"%/{name}")[0]
    data = recompose(bytes(trunk))
    check(f"{name} recomposed from the database, byte for byte", blake3.blake3(data).digest() == bytes(digest), f"{len(data):,} bytes")

# 6. idempotent ingest: ingesting Alice again finds the same trunk and adds nothing new
with tempfile.TemporaryDirectory() as d:
    subprocess.run([str(ROOT / "dag/ingest"), str(ROOT / "tier0/tier0.bin"), d, "/vault/Data/ProjectGutenberg/text/alice.txt"], check=True, capture_output=True)
    trunk2 = bytes.fromhex(open(f"{d}/source.copy").read().split("\t")[0][3:])
    ids = [bytes.fromhex(l.split("\t")[0][3:]) for l in open(f"{d}/entity.copy") if not l.split("\t")[2] == "0"]
    found = q("select count(*) from entity where id = any(%s)", [psycopg2.Binary(i) for i in ids])[0][0]
    alice_trunk = bytes(q("select trunk from source where origin like %s", "%/alice.txt")[0][0])
check("re-ingesting Alice: same trunk, nothing new", trunk2 == alice_trunk and found == len(ids), f"{found:,}/{len(ids):,} entities already present")

print()
fails = [n for n, ok in results if not ok]
print(f"== {len(results) - len(fails)}/{len(results)} checks passed" + (f"; FAILED: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
