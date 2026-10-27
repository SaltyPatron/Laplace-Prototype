"""Continuation as geometry (prototype): 'what follows the capital of'. The phrase is a trajectory, a path of its
constituents' IDs, like every stored physicality. One GIN lookup finds containers holding every constituent; each
container's path is cut into windows of the phrase's length, and a window (anchored where the phrase's rarest constituent
sits, so only plausible windows are built) whose discrete Fréchet distance to the
phrase is 0 is the phrase itself; the vertex after it is a continuation. No text is decoded to answer."""
import sys, time, collections
sys.argv = [sys.argv[0]]
exec(open("queries.py").read().split("# 1. lookup and counts")[0])
from psycopg2 import Binary as B
import struct
def id_xyz(i):
    v = int.from_bytes(i, "little"); pr = [v & ((1 << 43) - 1), (v >> 43) & ((1 << 43) - 1), v >> 86]
    return [struct.unpack("<d", struct.pack("<Q", (1021 << 52) | x))[0] for x in pr]
def line_wkt(ids): return "LINESTRING Z (" + ", ".join("%.17g %.17g %.17g" % tuple(id_xyz(i)) for i in ids) + ")"

def continuations(phrase_ids, how):
    arr = [B(i) for i in dict.fromkeys(phrase_ids)]; n = len(phrase_ids)
    if how == "native":                                    # laplace_follows: raw vertex bytes compared in C (SSE), no rows per vertex
        cur.execute("""select f, count(*) from physicality p, unnest(laplace_follows(st_asewkb(p.path), %s::bytea[])) f
                       where laplace_vertex_ids(st_asewkb(p.path)) @> %s::bytea[] group by f order by 2 desc""",
                    ([B(i) for i in phrase_ids], arr))
    elif how == "anchored":                                  # windows only where the rarest constituent sits, confirmed by Fréchet
        k = min(range(n), key=lambda j: rarity[phrase_ids[j]])
        sql = f"""
        with cand as (select entity, path from physicality where laplace_vertex_ids(st_asewkb(path)) @> %s::bytea[]),
        pts as (select c.entity, (d).path[1] as i, (d).geom as g from cand c, st_dumppoints(c.path) d),
        anchor as (select entity, i - {k} as s from pts where st_x(g) = %s and st_y(g) = %s and st_z(g) = %s),
        win as (select a.entity, a.s, st_makeline(array_agg(p.g order by p.i) filter (where p.i < a.s + {n})) as seg,
                       max(p.g::text) filter (where p.i = a.s + {n}) as nxt, count(*) filter (where p.i < a.s + {n}) as c
                from anchor a join pts p on p.entity = a.entity and p.i between a.s and a.s + {n} group by a.entity, a.s)
        select (laplace_vertex_ids(st_asewkb(nxt::geometry)))[1], count(*) from win
        where c = {n} and nxt is not null and st_frechetdistance(seg, st_geomfromtext(%s)) = 0 group by 1 order by 2 desc"""
        cur.execute(sql, (arr, *id_xyz(phrase_ids[k]), line_wkt(phrase_ids)))
    else:                                                  # every window compared to the phrase by discrete Fréchet distance
        sql = f"""
        with cand as (select entity, path from physicality where laplace_vertex_ids(st_asewkb(path)) @> %s::bytea[]),
        pts as (select c.entity, (d).path[1] as i, (d).geom as g from cand c, st_dumppoints(c.path) d),
        win as (select entity, i, st_makeline(array_agg(g) over w) as seg, lead(g, {n}) over (partition by entity order by i) as nxt,
                       count(*) over w as k
                from pts window w as (partition by entity order by i rows between current row and {n - 1} following))
        select (laplace_vertex_ids(st_asewkb(nxt)))[1], count(*) from win
        where k = {n} and nxt is not null and st_frechetdistance(seg, st_geomfromtext(%s)) = 0 group by 1 order by 2 desc"""
        cur.execute(sql, (arr, line_wkt(phrase_ids)))
    return [(bytes(a), c) for a, c in cur.fetchall()]

cur.execute("select id, occurrences from entity_stats where id = any(%s)", ([B(word_id(w)) for w in ("the", "capital", "of")] + [B(leaf(" "))],))
rarity = collections.defaultdict(lambda: 1e18, {bytes(i): float(o) for i, o in cur.fetchall()})    # occurrences: fewest anchors first
phrase = [word_id("the"), leaf(" "), word_id("capital"), leaf(" "), word_id("of"), leaf(" ")]
for how in ("every window", "anchored", "native", "native", "native"):
    t = time.time(); res = continuations(phrase, how); ms = (time.time() - t) * 1000
    print(f"{how:<13} {sum(c for _, c in res):,} continuations, {len(res)} distinct  ({ms:.0f} ms)")
print("  " + ", ".join(f"{text(k)} ({v})" for k, v in res[:12]))
cur.execute("select count(*) from physicality where laplace_vertex_ids(st_asewkb(path)) @> %s::bytea[]", ([B(i) for i in dict.fromkeys(phrase)],))
print(f"  containers holding every constituent: {cur.fetchone()[0]:,}")
