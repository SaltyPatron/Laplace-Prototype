"""Engine analysis for intrinsic ratings (Regan & Haworth 2011), deterministic and per game.
For every turn after move 8 that is not a repetition, Stockfish (Threads=1, Hash=16, fixed nodes,
ucinewgame before each game) evaluates the top K moves with MultiPV K; when the played move is not
among them it is searched alone (searchmoves) with the same node budget. Each game's analysis is a
pure function of its moves, so games can be analysed in any order, on any number of processes.
usage: analyze.py ENGINE NODES MULTIPV WORKERS games.jsonl out.jsonl"""
import sys, json, subprocess, os, io, chess, chess.pgn, concurrent.futures as cf
ENGINE, NODES, K, WORKERS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
SRC, OUT = sys.argv[5], sys.argv[6]
MATE = 10000
if os.name == "nt":                                         # yield the CPU to anything else on the machine
    import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
else: os.nice(19)
class UCI:
    def __init__(self):
        flags = 0x08000040 if os.name == "nt" else 0           # CREATE_NO_WINDOW | IDLE_PRIORITY_CLASS
        self.p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, creationflags=flags)
        self.send("uci"); self.wait("uciok")
        for k, v in (("Threads", 1), ("Hash", 16), ("MultiPV", K)): self.send(f"setoption name {k} value {v}")
        self.send("isready"); self.wait("readyok")
    def send(self, s): self.p.stdin.write(s + "\n")
    def wait(self, tok):
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("engine died")
            if l.startswith(tok): return l
    def go(self, moves, extra=""):
        self.send("position startpos moves " + " ".join(moves) if moves else "position startpos")
        self.send(f"go nodes {NODES}{extra}")
        lines = {}
        while True:
            l = self.p.stdout.readline()
            if l.startswith("bestmove"): break
            t = l.split()
            if "multipv" in t and "pv" in t and "score" in t and "lowerbound" not in t and "upperbound" not in t:
                i = int(t[t.index("multipv") + 1]); s = t[t.index("score") + 1]; v = int(t[t.index("score") + 2])
                cp = v if s == "cp" else (MATE - abs(v)) * (1 if v > 0 else -1)
                lines[i] = (t[t.index("pv") + 1], cp)        # the last report for each line is its final one
        return [lines[i] for i in sorted(lines)]
eng = None
def analyse(rec):
    global eng
    if eng is None: eng = UCI()
    gid, movetext = rec["gid"], rec["moves"]
    board = chess.Board(); played = []
    g = chess.pgn.read_game(io.StringIO(movetext))
    eng.send("ucinewgame"); eng.send("isready"); eng.wait("readyok")
    seen = {}; turns = []
    for ply, mv in enumerate(g.mainline_moves()):
        key = board._transposition_key(); seen[key] = seen.get(key, 0) + 1
        if ply >= 16 and seen[key] == 1 and board.legal_moves.count() > 1:
            top = eng.go(played)
            u = mv.uci(); pv = dict(top)
            if u not in pv: pv_played = eng.go(played, " searchmoves " + u); pc = pv_played[0][1] if pv_played else None
            else: pc = pv[u]
            turns.append({"ply": ply, "legal": board.legal_moves.count(), "top": top, "played": u, "cp": pc})
        board.push(mv); played.append(mv.uci())
    return {"gid": gid, "turns": turns}
if __name__ == "__main__":
    todo = [json.loads(l) for l in open(SRC)]
    done = set()
    if os.path.exists(OUT): done = {json.loads(l)["gid"] for l in open(OUT)}
    todo = [r for r in todo if r["gid"] not in done]
    print(f"{len(todo)} games to analyse ({len(done)} already done)", flush=True)
    with open(OUT, "a") as out, cf.ProcessPoolExecutor(WORKERS) as ex:
        for n, res in enumerate(ex.map(analyse, todo, chunksize=1)):
            out.write(json.dumps(res) + "\n"); out.flush()
            if n % 50 == 0: print(n, flush=True)
