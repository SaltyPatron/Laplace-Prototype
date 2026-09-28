"""Annotation layers as content: how much do UD sentence-level layers deduplicate?
For every sentence in every UD v2.17 treebank, build the UPOS sequence and the (deprel, head-offset)
sequence, hash each like a composition (BLAKE3 over its element IDs), and count distinct vs total."""
import glob, sys, time, blake3, collections
files = sorted(glob.glob("/vault/Data/UD-Treebanks/ud-treebanks-v2.17/*/*.conllu"))
t0 = time.time(); sentences = tokens = 0
seen = {"text": set(), "upos": set(), "deprel": set(), "upos+deprel": set()}
langs = set()
def h(seq): return blake3.blake3("\x1f".join(seq).encode()).digest()[:16]
for i, f in enumerate(files):
    langs.add(f.split("/")[-2].split("-")[0].replace("UD_", ""))
    text, upos, dep = None, [], []
    for line in open(f, encoding="utf-8", errors="replace"):
        if line.startswith("# text = "): text = line[9:].rstrip("\n")
        elif line == "\n":
            if upos:
                sentences += 1; tokens += len(upos)
                if text is not None: seen["text"].add(h([text]))
                seen["upos"].add(h(upos)); seen["deprel"].add(h(dep)); seen["upos+deprel"].add(h([a + "|" + b for a, b in zip(upos, dep)]))
            text, upos, dep = None, [], []
        elif not line.startswith("#"):
            c = line.split("\t")
            if len(c) > 7 and c[0].isdigit():
                upos.append(c[3])
                head = int(c[6]) if c[6].isdigit() else 0
                dep.append(f"{c[7].split(':')[0]}:{head - int(c[0]) if head else 0}")
    if (i + 1) % 100 == 0: print(f"[{time.time()-t0:5.0f}s] {i+1}/{len(files)} files, {sentences:,} sentences", flush=True)
print(f"\n{len(files)} files, {len(langs)} languages, {sentences:,} sentences, {tokens:,} tokens  ({time.time()-t0:.0f}s)")
print(f"one row per token per layer would be {2*tokens:,} records for these two layers")
for k, s in seen.items():
    print(f"  {k:<12} distinct {len(s):>10,}  of {sentences:>10,}  ({100*len(s)/sentences:5.1f}% of sentences need a new layer entity)")
