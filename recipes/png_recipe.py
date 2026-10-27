"""PNG recipe (prototype): decompose a PNG into a metadata tree and a pixel tree with real Laplace IDs,
then recompose it and require byte-exact equality with the original file.
  metadata: every chunk (IHDR fields, palette, text…), IDAT split sizes, zlib header, per-row filter types
  content : channel value (digit composition, e.g. 255 = [2,5,5]) -> pixel -> row -> image
  reconstruction: preflate diff (lets the deflate stream be re-created bit for bit)"""
import struct, zlib, subprocess, tempfile, os, sys, glob, time, pathlib, collections
import numpy as np, blake3
ROOT = pathlib.Path(__file__).resolve().parent.parent
T0 = np.fromfile(ROOT / "tier0/tier0.bin", dtype=[("id", "V16"), ("m", "<i8", (4,)), ("hilbert", "<u8"), ("rank", "<u4"), ("pad", "<u4")])
def comp(ids): return ids[0] if len(ids) == 1 else blake3.blake3(b"".join(ids)).digest()[:16]
NUM = [comp([T0["id"][ord(c)].tobytes() for c in str(v)]) for v in range(256)]      # channel values 0..255
PF = str(ROOT / "recipes/pf")
def chunks(b):
    assert b[:8] == b"\x89PNG\r\n\x1a\n"; i = 8
    while i < len(b):
        n = struct.unpack(">I", b[i:i + 4])[0]; t = b[i + 4:i + 8]; yield t, b[i + 8:i + 8 + n]; i += 12 + n
def unfilter(data, h, stride, bpp):
    out = np.zeros((h, stride), np.uint8); filt = []; p = 0; prev = np.zeros(stride, np.int32)
    for y in range(h):
        f = data[p]; row = np.frombuffer(data, np.uint8, stride, p + 1).astype(np.int32); p += 1 + stride; filt.append(f)
        cur = np.zeros(stride, np.int32)
        if f == 0: cur = row
        elif f == 2: cur = (row + prev) & 255
        else:
            for x in range(stride):
                a = cur[x - bpp] if x >= bpp else 0; b_ = prev[x]; c = prev[x - bpp] if x >= bpp else 0
                if f == 1: pr = a
                elif f == 3: pr = (a + b_) >> 1
                else:
                    pa, pb, pc = abs(b_ - c), abs(a - c), abs(a + b_ - 2 * c)
                    pr = a if pa <= pb and pa <= pc else (b_ if pb <= pc else c)
                cur[x] = (row[x] + pr) & 255
        out[y] = cur; prev = cur
    return out, filt
def refilter(px, filt, bpp):
    h, stride = px.shape; parts = []; prev = np.zeros(stride, np.int32)
    for y in range(h):
        cur = px[y].astype(np.int32); f = filt[y]
        left = np.concatenate([np.zeros(bpp, np.int32), cur[:-bpp]]); ul = np.concatenate([np.zeros(bpp, np.int32), prev[:-bpp]])
        if f == 0: r = cur
        elif f == 1: r = cur - left
        elif f == 2: r = cur - prev
        elif f == 3: r = cur - ((left + prev) >> 1)
        else:
            pa, pb, pc = np.abs(prev - ul), np.abs(left - ul), np.abs(left + prev - 2 * ul)
            pr = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, ul)); r = cur - pr
        parts.append(bytes([f]) + (r & 255).astype(np.uint8).tobytes()); prev = cur
    return b"".join(parts)
def ingest(path, seen, stats):
    b = open(path, "rb").read(); meta = []; idat = []
    for t, d in chunks(b):
        if t == b"IDAT": idat.append(d); meta.append((t, None, len(d)))
        else: meta.append((t, d, len(d)))
    ihdr = next(d for t, d, _ in meta if t == b"IHDR"); w, h, depth, ctype = struct.unpack(">IIBB", ihdr[:10])
    assert depth == 8 and ctype in (0, 2, 4, 6), "prototype handles 8-bit gray/RGB/gray-alpha/RGBA"
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]; z = b"".join(idat)
    with tempfile.TemporaryDirectory() as td:
        open(f"{td}/d", "wb").write(z[2:-4]); subprocess.run([PF, "decode", f"{td}/d", f"{td}/u", f"{td}/x"], check=True)
        un, diff = open(f"{td}/u", "rb").read(), open(f"{td}/x", "rb").read()
    px, filt = unfilter(un, h, w * ch, ch)
    # ---- content tree: channel value -> pixel -> row -> image (real Laplace IDs)
    pix = px.reshape(h, w, ch); rows = []
    for y in range(h):
        pids = [comp([NUM[v] for v in pix[y, x]]) for x in range(w)]
        for p in pids: stats["pixels"] += 1; stats["distinct_pixels"].add(p)
        rid = comp(pids); rows.append(rid); stats["rows"] += 1; stats["distinct_rows"].add(rid)
    image = comp(rows); stats["images"].add(image)
    # ---- recompose: pixels -> refilter -> preflate re-encode -> zlib wrap -> IDAT split -> chunks with CRCs
    un2 = refilter(px, filt, ch)
    with tempfile.TemporaryDirectory() as td:
        open(f"{td}/u", "wb").write(un2); open(f"{td}/x", "wb").write(diff)
        subprocess.run([PF, "encode", f"{td}/u", f"{td}/x", f"{td}/d"], check=True); deflate = open(f"{td}/d", "rb").read()
    z2 = z[:2] + deflate + struct.pack(">I", zlib.adler32(un2)); out = [b"\x89PNG\r\n\x1a\n"]; k = 0
    for t, d, n in meta:
        if t == b"IDAT": d = z2[k:k + n]; k += n
        out.append(struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d)))
    rebuilt = b"".join(out)
    return rebuilt == b, len(b), len(z), len(diff), w, h, ch, collections.Counter(filt)
files = sorted(glob.glob("/vault/Data/test-data/images/*.png")); stats = {"pixels": 0, "rows": 0, "distinct_pixels": set(), "distinct_rows": set(), "images": set()}
t0 = time.time(); ok = 0; tot_file = tot_z = tot_diff = 0
for f in files:
    e, nb, nz, nd, w, h, ch, fc = ingest(f, stats, stats); ok += e; tot_file += nb; tot_z += nz; tot_diff += nd
    print(f"  {os.path.basename(f)[:44]:<44} {w}x{h}x{ch}  {'EXACT' if e else 'MISMATCH'}  diff {nd:>6,} B ({100*nd/nz:4.1f}% of compressed)", flush=True)
print(f"\n{ok}/{len(files)} PNGs recomposed byte for byte  ({time.time()-t0:.0f}s)")
print(f"pixels {stats['pixels']:,}: distinct {len(stats['distinct_pixels']):,} ({100*len(stats['distinct_pixels'])/stats['pixels']:.2f}%); "
      f"rows {stats['rows']:,}: distinct {len(stats['distinct_rows']):,}; distinct images {len(stats['images'])}")
print(f"reconstruction data: {tot_diff:,} B = {100*tot_diff/tot_z:.1f}% of all compressed image data")
