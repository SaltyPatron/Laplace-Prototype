"""Storage comparison on the 26 test PNGs: rows of pixel references (current) vs a 2D quadtree
(image -> quadrants -> ... -> 8x8 patches) with perf-cache pixel literals packed 8 per vertex,
deduplicated at every level across all images, with run-length on identical siblings."""
import glob, struct, tempfile, subprocess, sys, numpy as np, blake3
sys.argv = ["x"]; exec(open("png_recipe.py").read().split("files = sorted(")[0])
V = 32; ENT = 16 + 32                                    # bytes per path vertex; per entity row (ID + coordinate)
def h(*parts): return blake3.blake3(b"".join(parts)).digest()[:16]
stored = {}                                              # node id -> vertices in its path
def patch(block):                                        # 8x8 (or smaller edge) block of pixels -> leaf composition
    flat = block.reshape(-1, block.shape[-1]); vals = [bytes(p) for p in flat]
    runs = [1]; lits = [vals[0]]
    for v in vals[1:]:
        if v == lits[-1]: runs[-1] += 1
        else: lits.append(v); runs.append(1)
    nid = h(b"P", bytes([block.shape[0], block.shape[1]]), *vals)       # pure content: the pixel values in order
    if nid not in stored: stored[nid] = max(1, -(-len(lits) // 8))      # 8 literal pixels per vertex, runs in M
    return nid
def quad(img, y0, x0, size):
    hgt, wid = img.shape[:2]
    if y0 >= hgt or x0 >= wid: return None
    if size <= 8: return patch(img[y0:min(y0 + size, hgt), x0:min(x0 + size, wid)])
    s = size // 2; kids = [quad(img, y0 + dy, x0 + dx, s) for dy in (0, s) for dx in (0, s)]
    kids = [k for k in kids if k is not None]
    if len(set(kids)) == 1 and len(kids) == 4: nid = h(b"Q4", kids[0])   # four identical quadrants: one vertex, run 4
    else: nid = h(b"Q", *kids)
    if nid not in stored: stored[nid] = len(set(kids)) if len(set(kids)) == 1 else len(kids)
    return nid
total_png = 0
for f in sorted(glob.glob("/vault/Data/test-data/images/*.png")):
    b = open(f, "rb").read(); total_png += len(b); meta = []; idat = []
    for t, d in chunks(b): (idat.append(d) if t == b"IDAT" else meta.append(d))
    w, hh, depth, ctype = struct.unpack(">IIBB", meta[0][:10]); ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]; z = b"".join(idat)
    with tempfile.TemporaryDirectory() as td:
        open(f"{td}/d", "wb").write(z[2:-4]); subprocess.run([PF, "decode", f"{td}/d", f"{td}/u", f"{td}/x"], check=True); un = open(f"{td}/u", "rb").read()
    px, _ = unfilter(un, hh, w * ch, ch); img = px.reshape(hh, w, ch)
    size = 8
    while size < max(w, hh): size *= 2
    quad(img, 0, 0, size)
nodes = len(stored); verts = sum(stored.values()); qt = verts * V + nodes * ENT
print(f"original PNGs:                       {total_png:>11,} bytes")
print(f"rows of pixel references (before):   {45512320:>11,} bytes  (98.5x)")
print(f"quadtree + packed pixel literals:    {qt:>11,} bytes  ({qt/total_png:.1f}x)   nodes {nodes:,}, vertices {verts:,}")
