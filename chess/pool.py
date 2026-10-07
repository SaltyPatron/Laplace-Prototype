"""The over-the-board classical pool of The Week in Chess, parsed exactly as ratings.py parses it.
Games are deduplicated by a content hash of players, date, round and moves; the pool is every TWIC
game whose event name says neither blitz nor rapid. Each game keeps its moves for engine analysis."""
import glob, os, re, blake3, datetime as dt
CHESS = os.environ.get("CHESS_DIR", "/vault/Data/Games/Chess")
TAG = re.compile(r'^\[(\w+) "(.*)"\]')
RESULT = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
def games(path):
    head, moves = {}, []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = TAG.match(line)
        if m:
            if moves: yield head, "".join(moves); head, moves = {}, []
            head[m.group(1)] = m.group(2)
        elif line.strip(): moves.append(line)
    if head: yield head, "".join(moves)
def when(h):
    d = (h.get("UTCDate") or h.get("Date", "")).replace("?", "1"); t = h.get("UTCTime", "12:00:00")
    try: return dt.datetime.strptime(d + " " + t, "%Y.%m.%d %H:%M:%S")
    except ValueError: return None
def elo(v): return int(v) if v.isdigit() else None
def otb_classical():
    """[(date, white, black, score, white_elo, black_elo, gid_hex, issue, movetext)] sorted by (date, gid)."""
    out, seen = [], set()
    for f in sorted(glob.glob(os.path.join(CHESS, "twic", "*.pgn"))):
        issue = os.path.basename(f)[:-4]
        for h, mv in games(f):
            if h.get("Result") not in RESULT or not h.get("White") or not h.get("Black"): continue
            gid = blake3.blake3((h.get("White", "") + h.get("Black", "") + h.get("Date", "") + h.get("Round", "") + mv).encode()).digest()[:16]
            if gid in seen: continue
            seen.add(gid)
            e = h.get("Event", "").lower()
            if "blitz" in e or "rapid" in e: continue
            w = when(h)
            if not w: continue
            out.append((w, h["White"], h["Black"], RESULT[h["Result"]], elo(h.get("WhiteElo", "")), elo(h.get("BlackElo", "")), gid.hex(), issue, mv))
    out.sort(key=lambda g: (g[0], g[6]))
    return out
