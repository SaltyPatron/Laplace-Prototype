"""Pick the training games to analyse: for every player of the held-out 10% who also played before
the cut, their most recent training game, preferring games that cover two such players. Ties break
on the game's content hash, so the selection does not depend on file or arrival order.
usage: select_games.py out.jsonl"""
import sys, json, pool
G = pool.otb_classical(); cut = int(len(G) * 0.9); train, test = G[:cut], G[cut:]
need = {p for g in test for p in g[1:3]} & {p for g in train for p in g[1:3]}
last = {}
for g in train:                                            # sorted by (date, gid): the last wins
    for p in g[1:3]:
        if p in need: last[p] = g
chosen = {}
for g in train:                                            # games that are the latest for both players first
    if last.get(g[1]) is g and last.get(g[2]) is g: chosen[g[6]] = g
covered = {p for g in chosen.values() for p in g[1:3]}
for p in sorted(need - covered):
    g = last[p]; chosen[g[6]] = g
with open(sys.argv[1], "w") as f:
    for gid in sorted(chosen):
        g = chosen[gid]; f.write(json.dumps({"gid": gid, "date": str(g[0].date()), "white": g[1], "black": g[2], "moves": g[8]}) + "\n")
print(f"{len(need)} held-out players seen before the cut; {len(chosen)} games chosen")
