"""dog -> concepts -> other languages, by consensus. IDs computed client-side; the database only finds and ranks."""
import sys, time, psycopg2
sys.argv = [sys.argv[0]]
exec(open("witnesses.py").read().split("# ---------------------------------------------------------------- Glicko-2")[0])
con = psycopg2.connect(host="/tmp", port=5439, user="laplace", dbname="laplace"); cur = con.cursor(); B = psycopg2.Binary
ATOM = {T0["id"][cp].tobytes(): cp for cp in range(len(T0))}
def parse_path(ewkb):
    b = bytes(ewkb); typ = struct.unpack_from("<I", b, 1)[0]; off = 5; n = 1
    if typ & 0xFF == 2: n = struct.unpack_from("<I", b, off)[0]; off += 4
    out = []
    for i in range(n):
        x, y, z, m = struct.unpack_from("<4d", b, off + 32 * i)
        pr = [struct.unpack("<Q", struct.pack("<d", v))[0] & ((1 << 52) - 1) for v in (x, y, z)]
        out.append(((pr[0] | pr[1] << 43 | pr[2] << 86).to_bytes(16, "little"), int(m)))
    return out
def decode(i):
    if i in ATOM: return chr(ATOM[i])
    cur.execute("select st_asewkb(path) from physicality where entity = %s", (B(i),)); row = cur.fetchone()
    return "".join(decode(c) * r for c, r in parse_path(row[0])) if row else "?"
def q(sql, *a):
    t = time.time(); cur.execute(sql, a); return cur.fetchall(), (time.time() - t) * 1000
def name(i):
    rows, _ = q("select st_asewkb(path) from physicality where entity = %s", B(i))
    return None
SENSE, LANG = label("sense"), label("lang")
word_text = {}
def lbl(s): i = label(s); word_text[i] = s; return i
dog = lbl("dog")
# every recorded quantity read as recorded, combined only here: observed usages, then the witness's sense order, then standing
senses, ms = q("""select c.object, s.rating, s.deviation, s.matches, w.members,
                         (select coalesce(sum(o.count), 0) from occurrence o where o.claim = c.id) as used,
                         (select d.position from ordinal d where d.claim = c.id and d.witness = %s) as ord
                  from claim c join consensus s on s.claim = c.id join witness_set w on w.id = s.witnesses
                  where c.subject = %s and c.predicate = %s
                  order by used desc, ord nulls last, s.rating desc""", B(label("Open English WordNet 2025+")), B(dog), B(label("eng")))
print(f"probe 1 — 'dog' up to its concepts ({ms:.1f} ms):")
ili_of = {}
for line in open("/vault/Data/CILI/ili-map-pwn30.tab"):
    i, s_ = line.split(); ili_of[label(i)] = i
for obj, r, rd, n, members, used, od in senses:
    print(f"   {ili_of.get(bytes(obj), '?'):<8} used {used:>3}  order {od or '-':>2}  rating {r:6.0f} ± {rd:3.0f}  matches {n}  witnesses: {', '.join(m.split(': ')[-1] for m in members)}")
langs = ["deu", "fra", "spa", "jpn", "ita", "fin", "pol", "arb"]
L = {lg: label(lg) for lg in langs}
target = senses[0][0]
rows, ms = q("""select c.subject, c.predicate, s.rating, s.deviation, s.matches from claim c
                join consensus s on s.claim = c.id
                where c.object = %s and c.predicate = any(%s) order by s.rating desc""",
             B(bytes(target)), [B(L[x]) for x in langs])
print(f"\nprobe 2 — {ili_of.get(bytes(target))} down into other languages ({ms:.1f} ms):")
inv = {v: k for k, v in L.items()}; best = {}
# recover lemma text by decoding the subject's path through entity paths is possible; for display, look up OMW text we just labeled
text_of = {v: k for k, v in LABEL.items()}
for subj, lang, r, rd, n in rows:
    best.setdefault(inv[bytes(lang)], []).append(f"{decode(bytes(subj))} {r:.0f}±{rd:.0f}")
for lg in langs: print(f"   {lg}: " + ", ".join(best.get(lg, [])[:5]))
