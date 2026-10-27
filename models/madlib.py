"""MadLib interviews (prototype). Templates are frequent three-word contexts observed in the Gutenberg corpus ("the
capital of ___"). For each: Laplace's own answer, the exact continuations counted inside the stored trajectories
(laplace_follows), and a model's answer, its next-token distribution for the same context. Reports how often the model's
first choice is Laplace's most observed continuation and how often it is in the model's top five.

Usage: python3 madlib.py model_dir n_templates"""
import collections, glob, re, sys, time, math
import torch, psycopg2
import json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_min import Llama
MD, N = sys.argv[1], int(sys.argv[2]); torch.set_num_threads(12)
T = time.time(); say = lambda m: print(f"[{time.time()-T:6.1f}s] {m}", flush=True)
# templates: the most frequent three-word contexts that end a phrase mid-sentence
cnt = collections.Counter(); WORD = re.compile(r"[A-Za-z]+")
for f in sorted(glob.glob("/vault/Data/ProjectGutenberg/text/*.txt"))[:60]:
    w = WORD.findall(open(f, encoding="utf-8", errors="replace").read())
    for i in range(len(w) - 3): cnt[(w[i], w[i + 1], w[i + 2])] += 1
STOP = {"the", "a", "an", "of", "and", "to", "in", "that", "it", "he", "she", "i", "was", "is", "his", "her", "you"}
templates = [t for t, c in cnt.most_common(20000) if t[2].lower() in STOP and not all(x.lower() in STOP for x in t)][:N]
say(f"{len(templates)} templates, e.g. " + " | ".join(" ".join(t) + " ___" for t in templates[:6]))
con = psycopg2.connect(host="localhost", port=5432, user="laplace", dbname="laplace"); cur = con.cursor()
cur.execute("SET enable_parallel_append = off")
def observed(t):
    ids = ", ".join(f"laplace_text_id({x!r})" for x in (t[0], " ", t[1], " ", t[2], " "))
    keys = ", ".join(f"laplace_text_id({x!r})" for x in (t[0], " ", t[1], t[2]))
    cur.execute(f"""SELECT f, count(*) FROM physicality p, unnest(laplace_follows(p.path, ARRAY[{ids}])) f
                    WHERE laplace_vertex_ids(p.path) @> ARRAY[{keys}] GROUP BY f ORDER BY 2 DESC LIMIT 20""")
    return cur.fetchall()
model = Llama(MD); vocab = json.load(open(MD + "/tokenizer.json"))["model"]["vocab"]; inv = {i: t for t, i in vocab.items()}
cur.execute("SELECT 1")
def wid(w): cur.execute("SELECT laplace_text_id(%s)", (w,)); return cur.fetchone()[0]
top1 = in5 = n = 0; tq = tm = 0.0; examples = []
for t in templates:
    a = time.time(); obs = observed(t); tq += time.time() - a
    if not obs: continue
    a = time.time()
    pieces = ["\u2581" + x for x in t]
    if not all(pc in vocab for pc in pieces): continue                       # whole-word tokens only, for now
    logits = model.logits([1] + [vocab[pc] for pc in pieces])
    lp = torch.log_softmax(logits, -1); best = torch.topk(lp, 30)
    words = []
    for i in best.indices.tolist():
        s = inv[i]
        if s.startswith("▁") and s[1:].isalpha(): words.append(s[1:])
        if len(words) == 5: break
    tm += time.time() - a
    ids = [wid(w) for w in words]; obs_top = obs[0][0]
    n += 1; top1 += bool(ids) and ids[0] == obs_top; in5 += obs_top in ids
    if len(examples) < 8:
        cur.execute("SELECT %s::text", (obs_top,)); examples.append((" ".join(t), words[:3], obs[0][1], sum(c for _, c in obs)))
say(f"interviewed {n} templates: Laplace's counting {tq:.1f} s ({1000*tq/n:.1f} ms each), model {tm:.1f} s ({1000*tm/n:.1f} ms each)")
print(f"\n  model's first choice = Laplace's most observed continuation: {100*top1/n:.1f}%")
print(f"  Laplace's most observed continuation in the model's top five: {100*in5/n:.1f}%")
for ctx, ws, c, tot in examples: print(f"   {ctx} ___   model: {', '.join(ws)}   (Laplace's top continuation seen {c} of {tot} times)")
