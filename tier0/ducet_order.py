import contextlib, io; _q = io.StringIO()
with contextlib.redirect_stdout(_q):
    #!/usr/bin/env python3
    """Compute UCA/DUCET and UCDXML statistics for Laplace tier-0 research.
    
    Reads (read-only) the local Unicode 17.0.0 data:
      uca/allkeys.txt, ucdxml/ucd.all.flat.xml
    Prints JSON with counts. Deterministic; no network.
    """
    import json, re, sys, hashlib
    import xml.etree.ElementTree as ET
    from collections import Counter, defaultdict
    
    ROOT = "/vault/Data/UCD/Public/UCD/latest"
    ALLKEYS = f"{ROOT}/uca/allkeys.txt"
    XML = f"{ROOT}/ucdxml/ucd.all.flat.xml"
    NS = "{http://www.unicode.org/ns/2003/ucd/1.0}"
    N = 0x110000
    
    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    
    # ---------------- UCD XML ----------------
    elem_count = Counter(); elem_cps = Counter()
    kind = [None] * N          # element kind per code point
    gc = [None] * N; blk = [None] * N; uideo = [False] * N; dt = [None] * N
    attr_sets = {}
    for ev, el in ET.iterparse(XML, events=("end",)):
        tag = el.tag.replace(NS, "")
        if tag in ("char", "reserved", "noncharacter", "surrogate"):
            a = el.attrib
            if "cp" in a:
                lo = hi = int(a["cp"], 16)
            elif "first-cp" in a:
                lo, hi = int(a["first-cp"], 16), int(a["last-cp"], 16)
            else:
                el.clear(); continue  # e.g. <char> children of named-sequences? (not expected)
            elem_count[tag] += 1; elem_cps[tag] += hi - lo + 1
            attr_sets.setdefault(tag, set()).update(k for k in a if not k.startswith("k"))
            for cp in range(lo, hi + 1):
                if kind[cp] is not None:
                    raise SystemExit(f"duplicate coverage {cp:04X}")
                kind[cp] = tag; gc[cp] = a.get("gc"); blk[cp] = a.get("blk")
                uideo[cp] = a.get("UIdeo") == "Y"; dt[cp] = a.get("dt")
            el.clear()
        elif tag == "repertoire":
            el.clear()
    uncovered = sum(1 for k in kind if k is None)
    gc_counts = Counter(gc)
    
    # ---------------- allkeys ----------------
    entries = []; implicit_ranges = []; version = None
    ce_re = re.compile(r"\[([.*])([0-9A-F]{4})\.([0-9A-F]{4})\.([0-9A-F]{4})\]")
    with open(ALLKEYS, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("@version"):
                version = line.split()[1]; continue
            if line.startswith("@implicitweights"):
                m = re.match(r"@implicitweights\s+([0-9A-F]+)\.\.([0-9A-F]+);\s*([0-9A-F]+)", line)
                implicit_ranges.append((int(m[1], 16), int(m[2], 16), int(m[3], 16))); continue
            lhs, rhs = line.split(";", 1)
            cps = tuple(int(x, 16) for x in lhs.split())
            ces = tuple((v == "*", int(p, 16), int(s, 16), int(t, 16)) for v, p, s, t in ce_re.findall(rhs))
            entries.append((cps, ces))
    
    single = {c[0]: ces for c, ces in entries if len(c) == 1}
    multi = [(c, ces) for c, ces in entries if len(c) > 1]
    multi_len = Counter(len(c) for c, _ in multi)
    contraction_starters = sorted({c[0] for c, _ in multi})
    starters_without_single = [cp for cp in contraction_starters if cp not in single]
    
    # classify every code point by how UCA weights it
    def implicit_class(cp):
        for lo, hi, base in implicit_ranges:
            if lo <= cp <= hi and kind[cp] == "char":
                # UTS #10 10.1.3: BBBB counts from the first range that shares this AAAA base,
                # so supplements (e.g. Tangut Supplement, base FB00) sort after the main block
                first = min(l for l, h, b in implicit_ranges if b == base)
                return "siniform", ((base, 0x20, 2), (((cp - first) | 0x8000), 0, 0))
        if uideo[cp]:
            core = blk[cp] in ("CJK", "CJK_Compat_Ideographs")  # UCDXML short block names
            aaaa = (0xFB40 if core else 0xFB80) + (cp >> 15)
            return ("han_core" if core else "han_other"), ((aaaa, 0x20, 2), (((cp & 0x7FFF) | 0x8000), 0, 0))
        aaaa = 0xFBC0 + (cp >> 15)
        return "unassigned_or_other", ((aaaa, 0x20, 2), (((cp & 0x7FFF) | 0x8000), 0, 0))
    
    cls = Counter(); cls_by_kind = defaultdict(Counter); han_aaaa = Counter()
    hangul_syllables = 0
    keys = []
    for cp in range(N):
        if cp in single:
            c = "explicit"
            key = tuple((p, s, t) for _, p, s, t in single[cp])
        elif 0xAC00 <= cp <= 0xD7A3:
            # Hangul syllables: canonically decomposed to jamo (UCA step 1, NFD), weights of jamo
            c = "hangul_via_nfd"; hangul_syllables += 1
            S = cp - 0xAC00; L = 0x1100 + S // 588; V = 0x1161 + (S % 588) // 28; T = 0x11A7 + S % 28
            seq = [L, V] + ([T] if T != 0x11A7 else [])
            key = tuple((p, s, t) for j in seq for _, p, s, t in single[j])
        else:
            c, key = implicit_class(cp)
            if c.startswith("han"):
                han_aaaa[f"{key[0][0]:04X}"] += 1
        cls[c] += 1; cls_by_kind[kind[cp]][c] += 1
        keys.append(key)
    
    # decomposable (canonical) chars lacking explicit entries (would be handled via NFD)
    canon_decomp_no_entry = sum(1 for cp in range(N) if dt[cp] == "can" and cp not in single and not (0xAC00 <= cp <= 0xD7A3))
    
    # ties: code points whose full (non-identical-level) CE key equals another's
    groups = Counter(keys)
    tied_cps = sum(v for v in groups.values() if v > 1)
    tie_groups = sum(1 for v in groups.values() if v > 1)
    largest_tie = max(groups.values())
    completely_ignorable = sum(1 for k in keys if all(w == (0, 0, 0) for w in k) or len(k) == 0)
    primary_ignorable = sum(1 for k in keys if all(w[0] == 0 for w in k))
    variable = sum(1 for cp in single if single[cp] and single[cp][0][0])
    distinct_primary_first = len({k[0][0] for k in keys if k})
    
    out = {
        "files": {"allkeys.txt": sha256(ALLKEYS), "ucd.all.flat.xml": sha256(XML)},
        "ucdxml": {
            "element_counts": dict(elem_count), "codepoints_covered": dict(elem_cps),
            "total_covered": sum(elem_cps.values()), "uncovered": uncovered,
            "gc_counts": dict(sorted(gc_counts.items())),
            "non_unihan_attributes_on_reserved": len(attr_sets.get("reserved", ())),
            "non_unihan_attributes_on_char": len(attr_sets.get("char", ())),
        },
        "allkeys": {
            "version": version, "implicitweights_lines": [f"{a:04X}..{b:04X};{c:04X}" for a, b, c in implicit_ranges],
            "total_entries": len(entries), "single_codepoint_entries": len(single),
            "multi_codepoint_entries": len(multi), "multi_by_length": dict(multi_len),
            "distinct_contraction_starters": len(contraction_starters),
            "contraction_starters_without_single_entry": [f"{c:04X}" for c in starters_without_single],
            "entries_with_multiple_CEs(expansions, single cp)": sum(1 for v in single.values() if len(v) > 1),
            "single_entries_marked_variable": variable,
            "explicit_single_by_element_kind": {k: v["explicit"] for k, v in cls_by_kind.items() if v["explicit"]},
        },
        "all_codepoints_classification": dict(cls),
        "classification_by_element_kind": {k: dict(v) for k, v in cls_by_kind.items()},
        "han_implicit_AAAA_leads": dict(sorted(han_aaaa.items())),
        "canonical_decomposables_without_explicit_entry_excl_hangul": canon_decomp_no_entry,
        "ties": {"codepoints_in_tie_groups": tied_cps, "tie_groups": tie_groups, "largest_tie_group": largest_tie,
                 "distinct_L1-L3_keys": len(groups),
                 "completely_ignorable_codepoints": completely_ignorable,
                 "primary_ignorable_codepoints": primary_ignorable},
    }
    json.dump(out, sys.stdout, indent=1)
    print()
    
    # ---------------- deterministic total order (non-ignorable, L1|L2|L3 sort key, then code point) ----------------
    def sortkey(key):
        l1 = tuple(w[0] for w in key if w[0]); l2 = tuple(w[1] for w in key if w[1]); l3 = tuple(w[2] for w in key if w[2])
        return (l1, l2, l3)
    sk = [sortkey(k) for k in keys]
    order = sorted(range(N), key=lambda cp: (sk[cp], cp))
