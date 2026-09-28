"""A model's embedding as testimony (prototype). For every token of the model's vocabulary: its neighbours that stand
out from the row's own noise (z >= 3 in cosine, at most 64), each recorded as the claim [token, near, token] between the
entities the tokens decompose to. Every one is appended to the ledger (witness: the model; condition: the component;
z kept), and standings are maintained inline with set-based statements: a new claim enters at the model's trust; a claim
already attested by another lineage plays one matchup; testimony from the same lineage only joins the witness set.

Usage: python3 testimony.py model_dir tensor_name witness lineage trust"""
import io, math, os, struct, sys, time, uuid
import numpy as np, torch, blake3, psycopg2
from safetensors import safe_open
MD, TENSOR, WNAME, LINEAGE, TRUST = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
sys.argv = [sys.argv[0]]
exec(open("/repos/src/Laplace-Prototype/semantics/witnesses.py").read().split("# ---------------------------------------------------------------- witnesses, in order")[0])
torch.set_num_threads(12); T = time.time(); say = lambda m: print(f"[{time.time()-T:6.1f}s] {m}", flush=True)
U = lambda b: str(uuid.UUID(bytes=bytes(b)))
con = psycopg2.connect(host="localhost", port=5432, user="laplace", dbname="laplace"); cur = con.cursor()

# token index -> entity, from the ingester's own decomposition of this vocabulary
tag = MD.split("/models/")[1].replace("/", "_").rstrip("_") + "_tokenizer.json.vocabmap"
raw = open(f"/vault/Data/LaplaceResearch/models/vocabmaps/{tag}", "rb").read()
ent = {struct.unpack_from("<I", raw, k)[0]: raw[k + 4:k + 20] for k in range(0, len(raw), 20)}
f = [p for p in os.listdir(MD) if p.endswith(".safetensors")][0]
with safe_open(os.path.join(MD, f), "pt") as h: E = h.get_tensor(TENSOR).float()
E = torch.nn.functional.normalize(E[: max(ent) + 1], dim=1); V = E.shape[0]
say(f"{WNAME}: {V:,} tokens x {E.shape[1]}, {len(ent):,} mapped to entities")

NEAR, COMP = label("near"), label("embedding"); W, L = label(WNAME), label(LINEAGE)
idx = torch.tensor(sorted(ent)); rows = []
for k in range(0, len(idx), 2048):
    blk = idx[k:k + 2048]; Sim = E[blk] @ E.T
    Z = (Sim - Sim.mean(1, keepdim=True)) / Sim.std(1, keepdim=True); Z[torch.arange(len(blk)), blk] = -1e9
    top = torch.topk(Z, 64, dim=1)
    for r in range(len(blk)):
        a = ent[int(blk[r])]
        for z, j in zip(top.values[r].tolist(), top.indices[r].tolist()):
            if z < 3: break
            b = ent.get(j)
            if b is None or b == a: continue
            rows.append((a, b, z))
say(f"{len(rows):,} relations above the noise (z >= 3), {len(rows)/len(ent):.1f} per token")

# entities for the labels (predicate, component, witness, lineage) the engine may lack
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
cur.execute("select id from entity where id = any(%s::uuid[])", ([U(k) for k in new_entities],)); have = {uuid.UUID(r[0]).bytes for r in cur.fetchall()}
todo = [(k, v) for k, v in new_entities.items() if k not in have]
if todo:
    cur.execute("CREATE TEMP TABLE e_stage (id uuid, tier smallint, coord geometry, path geometry)")
    buf = io.StringIO()
    for k, (m, t, ch) in todo: buf.write(f"{U(k)}\t{t}\t{ewkb_point(m).hex()}\t{ewkb_path(ch).hex()}\n")
    buf.seek(0); cur.copy_expert("COPY e_stage FROM STDIN", buf)
    cur.execute("""INSERT INTO entity SELECT id, tier, coord, laplace_hilbert4(coord) # (-9223372036854775808)::bigint FROM e_stage;
                   INSERT INTO physicality SELECT id, tier, laplace_hilbert4(coord) # (-9223372036854775808)::bigint, path FROM e_stage""")
cur.execute("DELETE FROM witness WHERE id = %s", (U(W),))
cur.execute("INSERT INTO witness VALUES (%s, %s, %s, %s, 'model')", (U(W), WNAME, U(L), TRUST))

# stage the testimony, then: claims, ledger, standings — each one set-based statement
t = time.time()
cur.execute("CREATE TEMP TABLE t_stage (claim uuid, subject uuid, object uuid, z real)")
buf = io.StringIO()
for a, b, z in rows: buf.write(f"{U(blake3.blake3(a + NEAR + b).digest()[:16])}\t{U(a)}\t{U(b)}\t{z:.3f}\n")
buf.seek(0); cur.copy_expert("COPY t_stage FROM STDIN", buf); cur.execute("ANALYZE t_stage")
cur.execute("""INSERT INTO claim SELECT DISTINCT s.claim, s.subject, %s::uuid, s.object FROM t_stage s
               WHERE NOT EXISTS (SELECT 1 FROM claim c WHERE c.id = s.claim)""", (U(NEAR),)); new_claims = cur.rowcount
cur.execute("INSERT INTO attestation (claim, witness, condition, score, z) SELECT claim, %s, %s, 1, z FROM t_stage ORDER BY claim",
            (U(W), U(COMP))); ledger = cur.rowcount
# standings: Glicko-2 for one win against an opponent whose weight g(phi) equals the trust
S_ = 173.7178; phi_o = (math.pi / math.sqrt(3)) * math.sqrt(1 / TRUST ** 2 - 1)
def one_win(r, rd, vol):
    return glicko(r, rd, vol, 1500.0, S_ * phi_o, 1.0)
r1, rd1, v1 = one_win(1500.0, 350.0, 0.06)
cur.execute("""INSERT INTO standing SELECT DISTINCT s.claim, %s, %s, %s, 1, ARRAY[%s::uuid] FROM t_stage s
               WHERE NOT EXISTS (SELECT 1 FROM standing x WHERE x.claim = s.claim)""", (r1, rd1, v1, U(W))); fresh = cur.rowcount
# claims other witnesses already hold: same lineage joins the set; another lineage plays a matchup
cur.execute("""SELECT x.claim, x.rating, x.deviation, x.volatility, x.matches,
                      EXISTS (SELECT 1 FROM witness w WHERE w.id = ANY (x.witnesses) AND w.lineage = %s) AS same_lineage
               FROM standing x JOIN (SELECT DISTINCT claim FROM t_stage) s ON s.claim = x.claim
               WHERE NOT (%s::uuid = ANY (x.witnesses))""", (U(L), U(W)))
upd = []; joined = 0
for c, r, rd, vol, m, same in cur.fetchall():
    if same: upd.append((c, r, rd, vol, m)); joined += 1
    else: nr, nrd, nv = one_win(r, rd, vol); upd.append((c, nr, nrd, nv, m + 1))
if upd:
    cur.execute("CREATE TEMP TABLE u_stage (claim uuid, rating real, deviation real, volatility real, matches int)")
    buf = io.StringIO(); buf.write("".join(f"{c}\t{r}\t{rd}\t{v}\t{m}\n" for c, r, rd, v, m in upd)); buf.seek(0)
    cur.copy_expert("COPY u_stage FROM STDIN", buf)
    cur.execute("""UPDATE standing x SET rating = u.rating, deviation = u.deviation, volatility = u.volatility, matches = u.matches,
                   witnesses = x.witnesses || %s::uuid FROM u_stage u WHERE u.claim = x.claim""", (U(W),))
con.commit()
say(f"ledger +{ledger:,}; claims +{new_claims:,}; standings: {fresh:,} new, {len(upd) - joined:,} matched against another lineage, {joined:,} joined by lineage ({time.time()-t:.1f} s in SQL)")
