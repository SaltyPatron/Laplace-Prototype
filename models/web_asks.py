"""The web asks, the model answers (prototype experiment). Questions come from what Laplace already holds: every
English word pair the curated web attests as the same concept or as hypernym and hyponym. The model answers each
with one dot product of its own embeddings; no row is swept. Reported: how well the answers separate attested pairs
from random pairs (AUC), and the time per answer."""
import collections, json, random, struct, sys, time, numpy as np, torch, psycopg2, psycopg2.extras
from safetensors import safe_open
psycopg2.extras.register_uuid(); torch.set_num_threads(12)
D = "/vault/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
T = time.time(); say = lambda m: print(f"[{time.time()-T:6.1f}s] {m}", flush=True)
con = psycopg2.connect(host="localhost", port=5432, user="laplace", dbname="laplace"); cur = con.cursor()
q = lambda sql, *a: (cur.execute(sql, a), cur.fetchall())[1]
ID = {w: q("SELECT laplace_text_id(%s)", w)[0][0] for w in ("eng", "hypernym", "hyponym")}
# the model's tokens, as entities (the ingester's own map); single-token words only, bare or space-led
raw = open("/vault/Data/LaplaceResearch/models/vocabmaps/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0_snapshots_fe8a4ea1ffedaf415f4da2f062534de366a451e6_tokenizer.json.vocabmap", "rb").read()
tok_ent = {struct.unpack_from("<I", raw, k)[0]: struct.unpack_from("16s", raw, k + 4)[0] for k in range(0, len(raw), 20)}
vocab = json.load(open(D + "/tokenizer.json"))["model"]["vocab"]
word_tok = {}                                                        # word entity -> token index (prefer the space-led form)
import uuid
for t, i in vocab.items():
    if t.startswith("▁") and t[1:].isalpha():
        w = q("SELECT laplace_text_id(%s)", t[1:])[0][0] if False else None
words = [t[1:] for t in vocab if t.startswith("▁") and t[1:].isalpha() and len(t) > 2]
cur.execute("SELECT w, laplace_text_id(w) FROM unnest(%s::text[]) w", (words,))
wid = {e: vocab["▁" + w] for w, e in cur.fetchall()}
with safe_open(D + "/model.safetensors", "pt") as h: E = h.get_tensor("model.embed_tokens.weight").float()
E = torch.nn.functional.normalize(E, dim=1)
say(f"{len(wid):,} English words are single space-led tokens")
# the web's questions: word pairs sharing a concept, and word pairs whose concepts are hypernym/hyponym
eng = collections.defaultdict(set); by_ili = collections.defaultdict(set)
for w, i in q("SELECT subject, object FROM claim WHERE predicate = %s AND subject = ANY (%s)", ID["eng"], list(wid)):
    eng[w].add(i); by_ili[i].add(w)
same = {(a, b) for ws in by_ili.values() for a in ws for b in ws if a != b}
hyp = set()
for x, y in q("SELECT subject, object FROM claim WHERE predicate IN (%s, %s) AND subject = ANY (%s)", ID["hypernym"], ID["hyponym"], list(by_ili)):
    for a in by_ili.get(x, ()):
        for b in by_ili.get(y, ()):
            if a != b: hyp.add((a, b))
rng = random.Random(5); W = list(wid)
rand = {(rng.choice(W), rng.choice(W)) for _ in range(max(len(same), len(hyp)))}
def answer(pairs):
    a = torch.tensor([wid[x] for x, _ in pairs]); b = torch.tensor([wid[y] for _, y in pairs])
    t = time.perf_counter(); s = (E[a] * E[b]).sum(1); return s.numpy(), time.perf_counter() - t
def auc(pos, neg):
    x = np.concatenate([pos, neg]); r = x.argsort().argsort() + 1
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
sp, ts = answer(list(same)); hp, th = answer(list(hyp)); rp, tr = answer(list(rand))
n = len(same) + len(hyp) + len(rand)
print(f"\n  questions: {len(same):,} same-concept pairs, {len(hyp):,} hypernym pairs, {len(rand):,} random pairs")
print(f"  answered in {1000*(ts+th+tr):.1f} ms total: {1e9*(ts+th+tr)/n:.0f} ns per answer, one dot product each")
print(f"  same concept vs random: AUC {auc(sp, rp):.3f}   (mean cosine {sp.mean():.3f} vs {rp.mean():.3f})")
print(f"  hypernym     vs random: AUC {auc(hp, rp):.3f}   (mean cosine {hp.mean():.3f} vs {rp.mean():.3f})")
