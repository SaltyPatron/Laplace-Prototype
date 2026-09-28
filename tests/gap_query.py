"""Gap query: every [Captain, ' ', ?] inside one book, with the gap's fillers counted."""
import sys, time, collections
sys.argv = [sys.argv[0]]
exec(open(__import__("pathlib").Path(__file__).with_name("queries.py")).read().split("# 1. lookup and counts")[0])     # reuse helpers only
t = time.time()
trunk = bytes(q("select trunk from source where origin like %s", "%/moby_dick.txt")[0][0][0])
paras = [c for c, _ in paths([trunk])[trunk]]                            # file -> paragraphs
book_sentences = set()
for i in range(0, len(paras), 20000):
    for pid, p in paths(paras[i:i + 20000]).items():
        if p is None: continue                                           # a paragraph that collapsed to a codepoint
        book_sentences.update(c for c, _ in p)
    book_sentences.update(x for x in paras[i:i + 20000])                 # paragraphs that collapsed to one sentence
cap = word_id("Captain"); sp = leaf(" ")
cands = [bytes(r[0]) for r in containers(cap)[0]]
in_book = [c for c in cands if c in book_sentences]
fill = collections.Counter()
for c in in_book:
    seq = expanded(c)
    for i in range(len(seq) - 2):
        if seq[i] == cap and seq[i + 1] == sp: fill[text(seq[i + 2])] += 1
print(f"Moby Dick: {len(book_sentences):,} sentences; 'Captain' containers corpus-wide {len(cands):,}, in this book {len(in_book):,}  ({(time.time()-t)*1000:.0f} ms)")
print("Captain ___ :", ", ".join(f"{k} ({v})" for k, v in fill.most_common(25)))
