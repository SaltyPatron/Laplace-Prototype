"""Do natural photos deduplicate? COCO val2017 (5,000 JPEGs): quantized DCT coefficients (the content a
byte-exact JPEG recipe stores) -> per-component quadtree over 8x8 blocks, deduplicated at every level across
all photos. Leaf blocks store nonzero coefficients as packed literals (6-bit position + 11-bit value, 208
bits per vertex). Reports distinct share per level and storage vs the original JPEG bytes (raw and LZ4).

Usage: python3 jpeg_dedup.py [number_of_photos]      (default: all 5,000)
"""
import glob, sys, time, collections, numpy as np, blake3, jpeglib, lz4.frame

files = sorted(glob.glob("/vault/Data/COCO/val2017/*.jpg"))
if len(sys.argv) > 1: files = files[:int(sys.argv[1])]
V, ENT = 32, 48                                   # bytes per path vertex; per entity row (ID + coordinate)
stored = {}; leafbuf = []
level_tot = collections.Counter(); level_new = collections.Counter()
file_ids = set(); jpeg_bytes = 0; dups = 0; T0 = time.time()

def leaf(blk):
    b = blk.astype(np.int16).tobytes(); nid = blake3.blake3(b"B" + b).digest()[:16]
    level_tot["8x8 block"] += 1
    if nid not in stored:
        nz = np.flatnonzero(blk); stored[nid] = max(1, -(-len(nz) * 17 // 208)); level_new["8x8 block"] += 1
        leafbuf.append(nz.astype(np.uint8).tobytes() + blk.reshape(-1)[nz].astype(np.int16).tobytes())
    return nid

def quad(C, y0, x0, size):
    if y0 >= C.shape[0] or x0 >= C.shape[1]: return None
    if size == 1: return leaf(C[y0, x0])
    s = size // 2
    kids = [k for k in (quad(C, y0 + dy, x0 + dx, s) for dy in (0, s) for dx in (0, s)) if k is not None]
    same = len(kids) == 4 and len(set(kids)) == 1
    nid = blake3.blake3((b"Q4" + kids[0]) if same else (b"Q" + b"".join(kids))).digest()[:16]
    name = f"{8 * size}px region"; level_tot[name] += 1
    if nid not in stored: stored[nid] = 1 if same else len(kids); level_new[name] += 1
    return nid

for i, f in enumerate(files):
    raw = open(f, "rb").read(); jpeg_bytes += len(raw)
    fid = blake3.blake3(raw).digest()[:16]
    if fid in file_ids: dups += 1; continue
    file_ids.add(fid)
    im = jpeglib.read_dct(f)
    for C in (im.Y, im.Cb, im.Cr):
        if C is None: continue
        size = 1
        while size < max(C.shape[0], C.shape[1]): size *= 2
        quad(C, 0, 0, size)
    if (i + 1) % 250 == 0:
        print(f"[{time.time()-T0:5.0f}s] {i+1}/{len(files)} photos; unique nodes {len(stored):,}", flush=True)

nodes = len(stored); verts = sum(stored.values()); logical = nodes * ENT + verts * V
leaf_raw = b"".join(leafbuf); leaf_lz4 = len(lz4.frame.compress(leaf_raw, compression_level=9))
print(f"\n{len(files):,} photos, {jpeg_bytes/1e6:.0f} MB of JPEG; exact duplicate files {dups}; {time.time()-T0:.0f}s")
print("  level              total      distinct    new")
for k in ["8x8 block"] + [f"{8 * 2**j}px region" for j in range(1, 12)]:
    if level_tot[k]: print(f"  {k:<16} {level_tot[k]:>11,} {level_new[k]:>11,}  {100*level_new[k]/level_tot[k]:5.1f}%")
print(f"  storage, logical (entity rows + path vertices): {logical/1e6:8.1f} MB = {logical/jpeg_bytes:.2f}x the JPEGs")
print(f"  unique leaf coefficients: raw {len(leaf_raw)/1e6:.1f} MB -> LZ4 {leaf_lz4/1e6:.1f} MB")
