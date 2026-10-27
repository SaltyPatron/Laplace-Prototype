"""Do independent model families agree on what is true? (prototype measurement) A random sample of the models'
[token, near, token] claims is graded against the curated web: same concept (both words lexicalize one ILI),
hypernym/hyponym (their concepts are related), or translation (the second word lexicalizes the first's concept in
another language). Grouped by how many independent model families attest the claim."""
import collections, random, sys, time, psycopg2, psycopg2.extras
psycopg2.extras.register_uuid()
T = time.time(); say = lambda m: print(f"[{time.time()-T:6.1f}s] {m}", flush=True)
con = psycopg2.connect(host="localhost", port=5432, user="laplace", dbname="laplace"); cur = con.cursor()
q = lambda sql, *a: (cur.execute(sql, a), cur.fetchall())[1]
ID = {w: q("SELECT laplace_text_id(%s)", w)[0][0] for w in ("near", "eng", "hypernym", "hyponym")}
SPACE = q("SELECT laplace_cp_id(32)")[0][0]
eng = collections.defaultdict(set)
for w, i in q("SELECT subject, object FROM claim WHERE predicate = %s", ID["eng"]): eng[w].add(i)
hyp = set((a, b) for a, b in q("SELECT subject, object FROM claim WHERE predicate IN (%s, %s)", ID["hypernym"], ID["hyponym"]))
other = collections.defaultdict(set)                                    # ILI -> words lexicalizing it in other languages
ilis = {i for s in eng.values() for i in s}
for w, i in q("SELECT subject, object FROM claim WHERE predicate <> %s AND object = ANY (%s)", ID["eng"], list(ilis)): other[i].add(w)
say(f"attested: {len(eng):,} English words, {len(hyp):,} hypernym links, {sum(len(v) for v in other.values()):,} other-language lexicalizations")
lineage = {w: l for w, l in q("SELECT id, lineage FROM witness WHERE kind = 'model'")}
rows = q("""SELECT c.subject, c.object, s.witnesses FROM claim c TABLESAMPLE SYSTEM (5) REPEATABLE (11)
            JOIN standing s ON s.claim = c.id WHERE c.predicate = %s LIMIT 300000""", ID["near"])
toks = {x for a, b, _ in rows for x in (a, b)}
word = {}
for e, ids in q("""SELECT entity, laplace_vertex_ids(path) FROM physicality WHERE tier = 3 AND entity = ANY (%s)""", list(toks)):
    if len(ids) == 2 and SPACE in ids: word[e] = [x for x in ids if x != SPACE][0]
w_of = lambda t: word.get(t, t)
stats = collections.defaultdict(collections.Counter)
for a, b, ws in rows:
    fam = len({lineage[w] for w in ws if w in lineage}); A, B = w_of(a), w_of(b)
    if A == B: continue
    ia, ib = eng.get(A, set()), eng.get(B, set())
    kind = "same concept" if ia & ib else "hypernym/hyponym" if any((x, y) in hyp for x in ia for y in ib) else \
           "translation" if any(B in other.get(x, ()) for x in ia) else "none"
    stats[fam][kind] += 1; stats[fam]["all"] += 1; stats[fam]["graded"] += bool(ia)
print(f"\n  {'families':>8} {'claims':>9} {'with an English word':>21} {'same concept':>13} {'hypernym':>9} {'translation':>12} {'attested':>9}")
for f in sorted(stats):
    s = stats[f]; g = max(s["graded"], 1)
    att = s["same concept"] + s["hypernym/hyponym"] + s["translation"]
    print(f"  {f:>8} {s['all']:>9,} {s['graded']:>21,} {100*s['same concept']/g:12.2f}% {100*s['hypernym/hyponym']/g:8.2f}% {100*s['translation']/g:11.2f}% {100*att/g:8.2f}%")
