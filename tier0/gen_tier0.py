"""Generate the Laplace tier-0 table (prototype).

For every one of the 1,114,112 code points:
  - DUCET total order from the local Unicode 17.0.0 uca/allkeys.txt (explicit, Hangul via jamo,
    @implicitweights, catch-all; L1|L2|L3 non-ignorable, ties by code point)       -> rank
  - H1 placement: Super-Fibonacci points (n = 1,114,112) walked in 4D Hilbert order
    (Skilling, 16 bits per axis over [-1,1]^4); DUCET rank r takes the r-th point     -> coordinate
  - exact fixed-point coordinate: m = round(x * 2^53) as int64, nudged toward zero until
    m.m <= 2^106 exactly, so every point is on or inside the glome in exact arithmetic
  - ID: first 16 bytes of BLAKE3 over the code point's generalized UTF-8 bytes
    (surrogates encoded as 3-byte sequences) - pure content, at most 4 bytes
  - Hilbert value of the coordinate (same 16-bit grid)
Writes tier0.bin (64-byte records indexed by code point) and fingerprint.txt.
"""
import hashlib, sys, pathlib
import numpy as np, blake3
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import time
_t = time.time()
def say(m): print(f'[{time.time()-_t:6.1f}s] {m}', flush=True)
say('building DUCET keys from allkeys.txt and the UCD ...')
from ducet_order import sk, N          # noqa: E402
from nfd17 import nfd                    # noqa: E402
# UCA identical level: ties on L1|L2|L3 are broken by the canonical decomposition (NFD), then code point
order = sorted(range(N), key=lambda cp: (sk[cp], nfd(cp), cp))
say('DUCET total order built')

n = N
rank = np.empty(n, dtype=np.int64); rank[np.array(order)] = np.arange(n)
ORDER_SHA = hashlib.sha256(b"".join(cp.to_bytes(3, "big") for cp in order)).hexdigest()
say(f'order sha256 {ORDER_SHA}')

def hilbert4(X, bits=16):
    X = X.copy(); D = X.shape[1]; M = np.uint64(1 << (bits - 1)); Q = M
    while Q > 1:
        Pm = Q - np.uint64(1)
        for i in range(D):
            m = (X[:, i] & Q) != 0
            X[m, 0] ^= Pm
            t = (X[~m, 0] ^ X[~m, i]) & Pm
            X[~m, 0] ^= t; X[~m, i] ^= t
        Q >>= np.uint64(1)
    for i in range(1, D): X[:, i] ^= X[:, i - 1]
    t = np.zeros(len(X), dtype=np.uint64); Q = M
    while Q > 1:
        m = (X[:, D - 1] & Q) != 0
        t[m] ^= Q - np.uint64(1); Q >>= np.uint64(1)
    for i in range(D): X[:, i] ^= t
    h = np.zeros(len(X), dtype=np.uint64)
    for b in range(bits - 1, -1, -1):
        for i in range(D):
            h = (h << np.uint64(1)) | ((X[:, i] >> np.uint64(b)) & np.uint64(1))
    return h

def grid16(x):
    return np.clip(((x + 1) / 2 * 65536), 0, 65535).astype(np.uint64)

phi, psi = np.sqrt(2.0), 1.533751168755204288118041
s = np.arange(n, dtype=np.float64) + 0.5; t = s / n
S = np.stack([np.sqrt(t) * np.sin(2 * np.pi * s / phi), np.sqrt(t) * np.cos(2 * np.pi * s / phi),
              np.sqrt(1 - t) * np.sin(2 * np.pi * s / psi), np.sqrt(1 - t) * np.cos(2 * np.pi * s / psi)], 1)
say('placing codepoints (Super-Fibonacci in Hilbert order) ...')
walk = np.argsort(hilbert4(grid16(S)), kind="stable")          # points in Hilbert order
X = S[walk][rank]                                              # point per code point

say('fixing coordinates to the exact grid ...')
# exact fixed-point coordinates with norm <= 1
SCALE = 1 << 53
Mx = np.rint(X * SCALE).astype(np.int64)
LIMIT = 1 << 106
nudged = 0
for cp in range(n):
    m = [int(v) for v in Mx[cp]]
    while sum(v * v for v in m) > LIMIT:
        j = max(range(4), key=lambda k: abs(m[k]))
        m[j] -= 1 if m[j] > 0 else -1
        nudged += 1
    Mx[cp] = m
say(f'nudged {nudged} ulps inward')

def utf8_general(cp):
    if cp < 0x80: return bytes([cp])
    if cp < 0x800: return bytes([0xC0 | cp >> 6, 0x80 | cp & 0x3F])
    if cp < 0x10000: return bytes([0xE0 | cp >> 12, 0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F])
    return bytes([0xF0 | cp >> 18, 0x80 | cp >> 12 & 0x3F, 0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F])

say('hashing leaf IDs ...')
rec = np.zeros(n, dtype=[("id", "V16"), ("m", "<i8", (4,)), ("hilbert", "<u8"), ("rank", "<u4"), ("pad", "<u4")])
rec["id"] = np.frombuffer(b"".join(blake3.blake3(utf8_general(cp)).digest()[:16] for cp in range(n)), dtype="V16")
rec["m"] = Mx
rec["hilbert"] = hilbert4(grid16(Mx.astype(np.float64) / SCALE))
rec["rank"] = rank
out = pathlib.Path(__file__).parent / "tier0.bin"
rec.tofile(out)
assert len(np.unique(rec["id"].view("S16"))) == n, "leaf ID collision"
fp = hashlib.sha256(out.read_bytes()).hexdigest()
(out.parent / "fingerprint.txt").write_text(
    f"tier0.bin sha256 {fp}\nducet order sha256 {ORDER_SHA}\nunicode 17.0.0 allkeys 17.0.0\n"
    f"records {n} x 64 bytes\nnudged ulps {nudged}\n")
print(open(out.parent / "fingerprint.txt").read())
