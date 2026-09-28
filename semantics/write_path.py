"""The consensus write path under hub load (prototype measurement). A stream of attestations over a million claims,
Zipf-distributed so a few hub claims receive most of them, is written in batches in two ways:
  in place: append the batch to a ledger and update each touched claim's standing (one set-based UPDATE per batch)
  ledger:   append only; a standing is read by aggregating the claim's ledger rows at query time
Reported: rows per second per batch, heap and dead-row growth, and the read cost of a hot and a cold claim."""
import io, math, time, uuid, numpy as np, psycopg2
con = psycopg2.connect(host="localhost", port=5432, user="laplace", dbname="laplace"); con.autocommit = True; cur = con.cursor()
C, B, NB, S = 1_000_000, 100_000, 50, 1.1
rng = np.random.default_rng(3); claims = [str(uuid.UUID(bytes=rng.bytes(16))) for _ in range(C)]
cur.execute("""DROP TABLE IF EXISTS wp_ledger, wp_standing;
               CREATE TABLE wp_ledger (seq bigint GENERATED ALWAYS AS IDENTITY, claim uuid NOT NULL, score real NOT NULL);
               CREATE INDEX ON wp_ledger (claim);
               CREATE TABLE wp_standing (claim uuid PRIMARY KEY, n int NOT NULL, total real NOT NULL) WITH (fillfactor = 80)""")
buf = io.StringIO(); buf.write("".join(f"{c}\t0\t0\n" for c in claims)); buf.seek(0); cur.copy_expert("COPY wp_standing FROM STDIN", buf)
cur.execute("VACUUM ANALYZE wp_standing")
size = lambda t: (cur.execute("SELECT pg_table_size(%s), (SELECT n_dead_tup FROM pg_stat_user_tables WHERE relname = %s)", (t, t)), cur.fetchone())[1]
t_led = t_upd = 0.0
for b in range(NB):
    ranks = np.minimum(rng.zipf(S, B), C) - 1; scores = rng.random(B)
    buf = io.StringIO(); buf.write("".join(f"{claims[r]}\t{s:.3f}\n" for r, s in zip(ranks, scores))); buf.seek(0)
    t = time.time(); cur.execute("CREATE TEMP TABLE IF NOT EXISTS wp_batch (claim uuid, score real); TRUNCATE wp_batch")
    cur.copy_expert("COPY wp_batch FROM STDIN", buf); cur.execute("INSERT INTO wp_ledger (claim, score) SELECT claim, score FROM wp_batch")
    t_led += time.time() - t; t = time.time()
    cur.execute("""UPDATE wp_standing s SET n = s.n + b.k, total = s.total + b.t
                   FROM (SELECT claim, count(*) k, sum(score) t FROM wp_batch GROUP BY claim) b WHERE b.claim = s.claim""")
    touched = cur.rowcount; t_upd += time.time() - t
    if b in (0, 9, 24, 49):
        sh, dh = size("wp_standing"); sl, _ = size("wp_ledger")
        print(f"batch {b+1:>2}: ledger {sl/1e6:7.1f} MB ({B*(b+1)/t_led:,.0f} rows/s)  standing {sh/1e6:6.1f} MB, {dh or 0:,} dead rows, "
              f"{touched:,} claims touched this batch ({B*(b+1)/t_upd:,.0f} attestations/s through updates)", flush=True)
hot, cold = claims[0], claims[C // 2]
for name, c in (("hot claim", hot), ("cold claim", cold)):
    cur.execute("SELECT count(*) FROM wp_ledger WHERE claim = %s", (c,)); k = cur.fetchone()[0]
    ts = []
    for _ in range(5): t = time.time(); cur.execute("SELECT n, total FROM wp_standing WHERE claim = %s", (c,)); cur.fetchone(); ts.append(time.time() - t)
    tl = []
    for _ in range(5): t = time.time(); cur.execute("SELECT count(*), sum(score) FROM wp_ledger WHERE claim = %s", (c,)); cur.fetchone(); tl.append(time.time() - t)
    print(f"{name}: {k:,} ledger rows; read the standing {1000*min(ts):.2f} ms, aggregate the ledger {1000*min(tl):.2f} ms")
cur.execute("VACUUM wp_standing"); sh, dh = size("wp_standing"); print(f"after VACUUM: standing {sh/1e6:.1f} MB")
