"""Sub-structure dedup of annotation layers: each dependency subtree (a head and all its descendants,
in sentence order) yields a layer sequence of (UPOS, deprel) pairs. Count distinct vs total, by size."""
import glob, time, blake3, collections
files = sorted(glob.glob("/vault/Data/UD-Treebanks/ud-treebanks-v2.17/*/*.conllu"))
t0 = time.time(); total = collections.Counter(); distinct = collections.defaultdict(set); nsent = 0
def flush(toks):
    kids = collections.defaultdict(list)
    for i, (upos, rel, head) in toks.items(): kids[head].append(i)
    memo = {}
    def span(i):
        if i in memo: return memo[i]
        s = [i] + [j for k in kids[i] for j in span(k)]; memo[i] = s; return s
    for i in toks:
        s = sorted(span(i)); n = len(s)
        if n < 2: continue
        key = blake3.blake3("\x1f".join(f"{toks[j][0]}|{toks[j][1]}" for j in s).encode()).digest()[:12]
        b = "2" if n == 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9-16" if n <= 16 else "17+"
        total[b] += 1; distinct[b].add(key)
for f in files:
    toks = {}
    for line in open(f, encoding="utf-8", errors="replace"):
        if line == "\n":
            if toks: flush(toks); nsent += 1
            toks = {}
        elif not line.startswith("#"):
            c = line.split("\t")
            if len(c) > 7 and c[0].isdigit() and c[6].isdigit():
                toks[int(c[0])] = (c[3], c[7].split(":")[0], int(c[6]))
print(f"{nsent:,} sentences in {time.time()-t0:.0f}s\n  subtree size   subtrees      distinct   share that must be stored")
T = D = 0
for b in ["2", "3-4", "5-8", "9-16", "17+"]:
    T += total[b]; D += len(distinct[b])
    print(f"  {b:>6}      {total[b]:>11,}  {len(distinct[b]):>11,}   {100*len(distinct[b])/total[b]:5.1f}%")
print(f"  {'all':>6}      {T:>11,}  {D:>11,}   {100*D/T:5.1f}%")
