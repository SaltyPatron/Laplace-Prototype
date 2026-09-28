"""Canonical decomposition (NFD) of single code points from the Unicode 17.0.0 UnicodeData.txt,
including algorithmic Hangul decomposition and canonical ordering by combining class."""
UD = "/vault/Data/UCD/Public/UCD/latest/ucd/UnicodeData.txt"
_dec, _ccc = {}, {}
for line in open(UD, encoding="utf-8"):
    f = line.split(";")
    cp = int(f[0], 16); _ccc[cp] = int(f[3])
    if f[5] and not f[5].startswith("<"):
        _dec[cp] = [int(x, 16) for x in f[5].split()]
SBase, LBase, VBase, TBase, LCount, VCount, TCount = 0xAC00, 0x1100, 0x1161, 0x11A7, 19, 21, 28
NCount, SCount = VCount * TCount, LCount * VCount * TCount
def _full(cp):
    if SBase <= cp < SBase + SCount:
        s = cp - SBase; out = [LBase + s // NCount, VBase + (s % NCount) // TCount]
        if s % TCount: out.append(TBase + s % TCount)
        return out
    if cp in _dec: return [x for d in _dec[cp] for x in _full(d)]
    return [cp]
def nfd(cp):
    seq = _full(cp)
    # canonical ordering: stable sort runs of non-starters by combining class
    i = 0
    while i < len(seq):
        if _ccc.get(seq[i], 0) == 0: i += 1; continue
        j = i
        while j < len(seq) and _ccc.get(seq[j], 0) != 0: j += 1
        seq[i:j] = sorted(seq[i:j], key=lambda c: _ccc.get(c, 0)); i = j
    return tuple(seq)
