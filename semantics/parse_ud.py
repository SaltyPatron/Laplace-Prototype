"""Dependency annotation of the WSD corpora by an outside witness (Stanza, UD English): each token of SemCor and the
five evaluation sets gets a UPOS, a head, and a deprel. Input is pre-tokenized with the corpora's own tokens, so every
annotation lines up with an instance id. Output: one TSV per corpus: sentence id, token index, token id or '-', form,
UPOS, head index (0 = root), deprel.

Usage: python3 parse_ud.py bench              (CPU vs GPU on the first 500 SemCor sentences)
       python3 parse_ud.py [cpu|gpu]           (parse everything)"""
import sys, time, xml.etree.ElementTree as ET, stanza, torch
W = "/vault/Data/WSD/WSD_Evaluation_Framework"; OUT = "/vault/Data/LaplaceResearch/semantics/parses"
CORPORA = {"semcor": f"{W}/Training_Corpora/SemCor/semcor.data.xml"}
for n in ("semeval2007", "senseval2", "senseval3", "semeval2013", "semeval2015"):
    CORPORA[n] = f"{W}/Evaluation_Datasets/{n}/{n}.data.xml"
def sentences(path):
    for s in ET.parse(path).getroot().iter("sentence"):
        toks = [(t.get("id") or "-", t.text) for t in s]
        if toks: yield s.get("id"), toks
def pipe(dev):
    return stanza.Pipeline("en", dir="/vault/Data/LaplaceResearch/models/stanza", processors="tokenize,pos,lemma,depparse",
                           tokenize_pretokenized=True, use_gpu=(dev == "gpu"), verbose=False, download_method=None)
def parse(nlp, sents):
    doc = nlp([[w.replace(" ", "_") for _, w in toks] for _, toks in sents])
    for (sid, toks), sent in zip(sents, doc.sentences):
        for i, (w, (tid, form)) in enumerate(zip(sent.words, toks), 1):
            yield f"{sid}\t{i}\t{tid}\t{form}\t{w.upos}\t{w.head}\t{w.deprel}\n"
if sys.argv[1] == "bench":
    sents = list(sentences(CORPORA["semcor"]))[:500]; ntok = sum(len(t) for _, t in sents)
    for dev in ("cpu", "gpu"):
        if dev == "gpu" and not torch.cuda.is_available(): continue
        nlp = pipe(dev); list(parse(nlp, sents[:20]))                       # warm up
        t = time.time(); rows = list(parse(nlp, sents)); dt = time.time() - t
        print(f"{dev}: {len(sents)} sentences, {ntok:,} tokens in {dt:.1f} s = {len(sents)/dt:.0f} sentences/s, {ntok/dt:,.0f} tokens/s", flush=True)
else:
    nlp = pipe(sys.argv[1])
    for name, path in CORPORA.items():
        t = time.time(); sents = list(sentences(path))
        with open(f"{OUT}/{name}.ud.tsv", "w") as f:
            for k in range(0, len(sents), 2000): f.writelines(parse(nlp, sents[k:k + 2000]))
        print(f"{name}: {len(sents):,} sentences in {time.time()-t:.0f} s", flush=True)
