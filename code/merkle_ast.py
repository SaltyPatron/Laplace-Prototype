"""Merkle AST demo: parse PostgreSQL and CPython sources with tree-sitter; every syntax subtree becomes a
content-addressed node (ID = hash of its children's IDs; gaps between children are text constituents;
the node KIND is not hashed — it is the grammar's attestation). Checks exact recomposition and measures sharing."""
import sys, time, collections, pathlib, blake3
import tree_sitter, tree_sitter_c, tree_sitter_python
sys.setrecursionlimit(100000)
LANG = {".c": tree_sitter.Language(tree_sitter_c.language()), ".h": tree_sitter.Language(tree_sitter_c.language()),
        ".py": tree_sitter.Language(tree_sitter_python.language())}
parsers = {ext: tree_sitter.Parser(lang) for ext, lang in LANG.items()}
T0 = time.time()
seen = {}                                  # node id -> expanded token count
total = collections.Counter(); distinct = collections.Counter(); kinds = collections.defaultdict(set)
def leaf(b): return blake3.blake3(b"L" + b).digest()[:16]       # stand-in for the codepoint composition of this text
def build(node, src):
    """Returns (id, tokens, bytes) for node; composition = children with the gaps between them."""
    if node.child_count == 0:
        b = src[node.start_byte:node.end_byte]; return leaf(b), 1, b
    parts, pos, toks, out = [], node.start_byte, 0, []
    for ch in node.children:
        if ch.start_byte > pos: g = src[pos:ch.start_byte]; parts.append(leaf(g)); out.append(g)
        cid, ct, cb = build(ch, src); parts.append(cid); toks += ct; out.append(cb); pos = ch.end_byte
    if node.end_byte > pos: g = src[pos:node.end_byte]; parts.append(leaf(g)); out.append(g)
    nid = parts[0] if len(parts) == 1 else blake3.blake3(b"".join(parts)).digest()[:16]
    b = b"".join(out)
    bucket = "1-2" if toks <= 2 else "3-8" if toks <= 8 else "9-32" if toks <= 32 else "33-128" if toks <= 128 else "129+"
    total[bucket] += 1
    if nid not in seen: seen[nid] = toks; distinct[bucket] += 1
    kinds[nid].add(node.type)
    return nid, toks, b
files = sorted(p for root in ("/vault/Data/code-authority/postgres", "/vault/Data/code-authority/cpython")
               for p in pathlib.Path(root).rglob("*") if p.suffix in LANG and p.is_file())
exact = bad = 0; trunks = set(); nbytes = 0
for i, f in enumerate(files):
    src = f.read_bytes(); nbytes += len(src)
    tree = parsers[f.suffix].parse(src); root = tree.root_node
    rid, _, b = build(root, src)
    head, tail = src[:root.start_byte], src[root.end_byte:]
    ok = head + b + tail == src; exact += ok; bad += not ok
    trunks.add(rid)
    if (i + 1) % 1000 == 0: print(f"[{time.time()-T0:5.0f}s] {i+1}/{len(files)} files", flush=True)
print(f"\n{len(files):,} files ({nbytes/1e6:.0f} MB) in {time.time()-T0:.0f}s; recomposed exactly: {exact:,}, mismatched: {bad}; distinct file trunks: {len(trunks):,}")
print("  subtree size (tokens)   subtrees     distinct   share that is new")
T = D = 0
for b in ["1-2", "3-8", "9-32", "33-128", "129+"]:
    T += total[b]; D += distinct[b]
    print(f"  {b:>10}           {total[b]:>11,}  {distinct[b]:>10,}   {100*distinct[b]/max(total[b],1):5.1f}%")
print(f"  {'all':>10}           {T:>11,}  {D:>10,}   {100*D/T:5.1f}%")
multi = sum(1 for k in kinds.values() if len(k) > 1)
print(f"  distinct subtrees attested under more than one node kind: {multi:,} (the kind is an attestation, not identity)")
