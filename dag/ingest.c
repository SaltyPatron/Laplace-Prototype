/* Laplace storage prototype: ingest UTF-8 text files into a content-addressed Merkle DAG.
 *
 * Tier 0 comes from tier0/tier0.bin (1,114,112 x 64-byte records: BLAKE3-128 ID, exact
 * fixed-point coordinate m[4] with value = m / 2^53, Hilbert value, DUCET rank).
 *
 * Per file (plain text, lossless):
 *   file      = [paragraph ...]            tier 5 (the file trunk; a .txt has no header fields)
 *   paragraph = [sentence ...]             tier 4 (ends at a blank line)
 *   sentence  = [word segment ...]         tier 3 (UAX #29 sentence boundaries)
 *   segment   = [grapheme | codepoint ...] tier 2 (UAX #29 word boundaries; spaces, punctuation too)
 *   grapheme  = [codepoint ...]            tier 1 (only when a grapheme has more than one codepoint)
 * Segmentation is tailored for hard-wrapped text: ICU sees a copy in which single line breaks
 * inside a paragraph are spaces; boundaries are applied to the original bytes, so every byte,
 * including every line break, stays in the content.
 *
 * Identity: ID = BLAKE3(concatenated 16-byte child IDs, repeats included)[:16] - pure content.
 * A composition with one child is that child (one-child collapse).
 * Real coordinate: exact vertex average of the children's fixed-point coordinates, truncated
 * toward zero (so nothing can leave the glome). Trajectory: child references with run lengths.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <unicode/ubrk.h>
#include <unicode/utext.h>
#include "blake3.h"

#define NCP 1114112u
typedef struct { uint8_t id[16]; int64_t m[4]; uint64_t hilbert; uint32_t rank, pad; } T0;
typedef struct { uint64_t ref; uint32_t run; } Vtx;   /* ref < NCP: codepoint; else NCP + composition index */
typedef struct { uint8_t id[16]; int64_t m[4]; uint64_t vstart; uint32_t vcount, len; uint8_t tier; } Comp;

static const T0 *t0;
static Comp *comps; static uint64_t ncomp, capcomp;
static Vtx *verts; static uint64_t nvert, capvert;
static uint64_t *table; static uint64_t tcap;          /* open addressing: stores comp index + 1 */
static uint64_t stat_new[6], stat_hit[6], stat_refs[6], stat_mismatch;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *xrealloc(void *p, size_t n){ p = realloc(p, n); if (!p) { perror("realloc"); exit(1); } return p; }

static const uint8_t *ref_id(uint64_t r){ return r < NCP ? t0[r].id : comps[r - NCP].id; }
static const int64_t *ref_m(uint64_t r){ return r < NCP ? t0[r].m : comps[r - NCP].m; }
static int ref_tier(uint64_t r){ return r < NCP ? 0 : comps[r - NCP].tier; }

static uint64_t hkey(const uint8_t *id){ uint64_t k; memcpy(&k, id, 8); return k; }
static void table_grow(void){
    uint64_t ncap = tcap ? tcap * 2 : (1u << 20), *nt = calloc(ncap, 8);
    for (uint64_t i = 0; i < tcap; i++) if (table[i]) {
        uint64_t h = hkey(comps[table[i] - 1].id) & (ncap - 1);
        while (nt[h]) h = (h + 1) & (ncap - 1);
        nt[h] = table[i];
    }
    free(table); table = nt; tcap = ncap;
}

/* Build (or find) the composition for children[0..n). Returns a reference. */
static uint64_t node(const uint64_t *ch, uint32_t n, uint8_t tier){
    if (n == 1) return ch[0];                                   /* one-child collapse */
    uint8_t id[32]; blake3_hasher h; blake3_hasher_init(&h);
    for (uint32_t i = 0; i < n; i++) blake3_hasher_update(&h, ref_id(ch[i]), 16);
    blake3_hasher_finalize(&h, id, 16);
    if ((ncomp + 1) * 2 > tcap) table_grow();
    uint64_t s = hkey(id) & (tcap - 1);
    while (table[s]) {
        Comp *c = &comps[table[s] - 1];
        if (!memcmp(c->id, id, 16)) {                           /* same content: children must match */
            uint32_t k = 0, bad = (c->len != n);
            for (uint32_t v = 0; v < c->vcount && !bad; v++)
                for (uint32_t r = 0; r < verts[c->vstart + v].run && !bad; r++, k++)
                    bad = (k >= n || verts[c->vstart + v].ref != ch[k]);
            if (bad) stat_mismatch++;
            stat_hit[tier]++;
            return NCP + (table[s] - 1);
        }
        s = (s + 1) & (tcap - 1);
    }
    if (ncomp == capcomp) { capcomp = capcomp ? capcomp * 2 : (1u << 20); comps = xrealloc(comps, capcomp * sizeof *comps); }
    Comp *c = &comps[ncomp];
    memcpy(c->id, id, 16); c->tier = tier; c->len = n; c->vstart = nvert; c->vcount = 0;
    __int128 sum[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < n; i++) {
        const int64_t *m = ref_m(ch[i]);
        for (int d = 0; d < 4; d++) sum[d] += m[d];
        if (c->vcount && verts[nvert - 1].ref == ch[i]) { verts[nvert - 1].run++; continue; }
        if (nvert == capvert) { capvert = capvert ? capvert * 2 : (1u << 22); verts = xrealloc(verts, capvert * sizeof *verts); }
        verts[nvert++] = (Vtx){ ch[i], 1 }; c->vcount++;
    }
    for (int d = 0; d < 4; d++) c->m[d] = (int64_t)(sum[d] / (__int128)n);   /* C division truncates toward zero */
    table[s] = ++ncomp;
    stat_new[tier]++;
    return NCP + (ncomp - 1);
}

/* ---- UTF-8 ---- */
static int utf8_next(const uint8_t *s, size_t n, size_t *i, uint32_t *cp){
    uint8_t b = s[*i];
    if (b < 0x80) { *cp = b; *i += 1; return 1; }
    int len = (b >= 0xF0) ? 4 : (b >= 0xE0) ? 3 : (b >= 0xC0) ? 2 : 0;
    if (!len || *i + len > n) return 0;
    uint32_t c = b & (0x7F >> len);
    for (int k = 1; k < len; k++) { if ((s[*i + k] & 0xC0) != 0x80) return 0; c = (c << 6) | (s[*i + k] & 0x3F); }
    *cp = c; *i += len; return c < NCP;
}
static size_t utf8_put(uint32_t cp, uint8_t *o){
    if (cp < 0x80) { o[0] = cp; return 1; }
    if (cp < 0x800) { o[0] = 0xC0 | cp >> 6; o[1] = 0x80 | (cp & 0x3F); return 2; }
    if (cp < 0x10000) { o[0] = 0xE0 | cp >> 12; o[1] = 0x80 | (cp >> 6 & 0x3F); o[2] = 0x80 | (cp & 0x3F); return 3; }
    o[0] = 0xF0 | cp >> 18; o[1] = 0x80 | (cp >> 12 & 0x3F); o[2] = 0x80 | (cp >> 6 & 0x3F); o[3] = 0x80 | (cp & 0x3F); return 4;
}

/* ---- boundaries ---- */
typedef struct { int32_t *b; size_t n, cap; } Bounds;
static void bounds_of(UBreakIteratorType ty, UText *ut, Bounds *out){
    UErrorCode e = U_ZERO_ERROR; UBreakIterator *bi = ubrk_open(ty, "", NULL, 0, &e);
    ubrk_setUText(bi, ut, &e); if (U_FAILURE(e)) { fprintf(stderr, "ICU: %s\n", u_errorName(e)); exit(1); }
    out->n = 0;
    for (int32_t p = ubrk_first(bi); p != UBRK_DONE; p = ubrk_next(bi)) {
        if (out->n == out->cap) { out->cap = out->cap ? out->cap * 2 : 4096; out->b = xrealloc(out->b, out->cap * 4); }
        out->b[out->n++] = p;
    }
    ubrk_close(bi);
}

typedef struct { uint64_t *v; size_t n, cap; } Refs;
static void push(Refs *r, uint64_t x){ if (r->n == r->cap) { r->cap = r->cap ? r->cap * 2 : 256; r->v = xrealloc(r->v, r->cap * 8); } r->v[r->n++] = x; }

/* ---- recomposition ---- */
static uint8_t *rbuf; static size_t rlen, rcap;
static void expand(uint64_t r){
    if (r < NCP) {
        if (rlen + 4 > rcap) { rcap = rcap ? rcap * 2 : (1u << 24); rbuf = xrealloc(rbuf, rcap); }
        rlen += utf8_put((uint32_t)r, rbuf + rlen); return;
    }
    const Comp *c = &comps[r - NCP];
    for (uint32_t v = 0; v < c->vcount; v++) for (uint32_t k = 0; k < verts[c->vstart + v].run; k++) expand(verts[c->vstart + v].ref);
}

/* ---- EWKB hex for COPY ---- */
static void hexbytes(FILE *f, const void *p, size_t n){ static const char *H = "0123456789abcdef"; const uint8_t *b = p; for (size_t i = 0; i < n; i++) { fputc(H[b[i] >> 4], f); fputc(H[b[i] & 15], f); } }
static void put_u32(FILE *f, uint32_t v){ hexbytes(f, &v, 4); }
static void put_d(FILE *f, double v){ hexbytes(f, &v, 8); }
static void id_point(const uint8_t *id, double xyz[3]){                     /* 43 + 43 + 42 bits into [1,2) mantissas */
    unsigned __int128 v = 0; for (int i = 15; i >= 0; i--) v = (v << 8) | id[i];
    uint64_t part[3] = { (uint64_t)(v & (((unsigned __int128)1 << 43) - 1)), (uint64_t)((v >> 43) & (((unsigned __int128)1 << 43) - 1)), (uint64_t)(v >> 86) };
    for (int k = 0; k < 3; k++) { uint64_t bits = (1023ull << 52) | part[k]; memcpy(&xyz[k], &bits, 8); }
}

/* Skilling Hilbert index, 4 dims x 16 bits over [-1,1]^4 (same grid as tier-0 generation). */
static uint64_t hilbert(const int64_t m[4]){
    uint64_t X[4];
    for (int d = 0; d < 4; d++) { double x = (double)m[d] / 9007199254740992.0, g = (x + 1) / 2 * 65536; X[d] = g < 0 ? 0 : g > 65535 ? 65535 : (uint64_t)g; }
    for (uint64_t Q = 1u << 15; Q > 1; Q >>= 1) { uint64_t P = Q - 1;
        for (int i = 0; i < 4; i++) { if (X[i] & Q) X[0] ^= P; else { uint64_t t = (X[0] ^ X[i]) & P; X[0] ^= t; X[i] ^= t; } } }
    for (int i = 1; i < 4; i++) X[i] ^= X[i - 1];
    uint64_t t = 0; for (uint64_t Q = 1u << 15; Q > 1; Q >>= 1) if (X[3] & Q) t ^= Q - 1;
    for (int i = 0; i < 4; i++) X[i] ^= t;
    uint64_t h = 0; for (int b = 15; b >= 0; b--) for (int i = 0; i < 4; i++) h = (h << 1) | ((X[i] >> b) & 1);
    return h;
}

int main(int argc, char **argv){
    if (argc < 4) { fprintf(stderr, "usage: ingest tier0.bin outdir file...\n"); return 2; }
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    if ((size_t)st.st_size != NCP * sizeof(T0)) { fprintf(stderr, "tier0.bin has wrong size\n"); return 1; }
    t0 = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    const char *out = argv[2]; int nfiles = argc - 3;
    uint64_t *trunks = calloc(nfiles, 8); double t_start = now(); size_t total_bytes = 0; int ok_files = 0, bad_files = 0;
    Bounds sb = {0}, wb = {0}, gb = {0}; Refs segs = {0}, sents = {0}, paras = {0}, g = {0};
    char path[4096]; snprintf(path, sizeof path, "%s/source.copy", out); FILE *fsrc = fopen(path, "w");

    for (int fi = 0; fi < nfiles; fi++) {
        const char *fn = argv[3 + fi]; double tf = now();
        FILE *f = fopen(fn, "rb"); fseek(f, 0, SEEK_END); size_t n = ftell(f); rewind(f);
        uint8_t *src = malloc(n + 1), *cpy = malloc(n + 1);
        if (fread(src, 1, n, f) != n) { perror(fn); return 1; } fclose(f);
        for (size_t i = 0; i < n; ) { uint32_t cp; if (!utf8_next(src, n, &i, &cp)) { fprintf(stderr, "  %s: invalid UTF-8 at byte %zu, skipped\n", fn, i); goto next; } }
        /* tailoring copy: single line breaks inside a paragraph become spaces (same byte length) */
        memcpy(cpy, src, n);
        for (size_t i = 0; i < n; ) {
            if (src[i] != '\n' && src[i] != '\r') { i++; continue; }
            size_t a = i, b = i + ((src[i] == '\r' && i + 1 < n && src[i + 1] == '\n') ? 2 : 1);
            size_t j = b; while (j < n && (src[j] == ' ' || src[j] == '\t')) j++;
            int next_blank = (j < n && (src[j] == '\n' || src[j] == '\r'));
            size_t k = a; while (k > 0 && (src[k - 1] == ' ' || src[k - 1] == '\t')) k--;
            int prev_blank = (k == 0 || src[k - 1] == '\n' || src[k - 1] == '\r');
            if (!next_blank && !prev_blank) for (size_t z = a; z < b; z++) cpy[z] = ' ';
            i = b;
        }
        UErrorCode e = U_ZERO_ERROR; UText *ut = utext_openUTF8(NULL, (const char *)cpy, n, &e);
        bounds_of(UBRK_SENTENCE, ut, &sb); bounds_of(UBRK_WORD, ut, &wb); bounds_of(UBRK_CHARACTER, ut, &gb);
        paras.n = 0; sents.n = 0;
        size_t wi = 0, gi = 0;
        for (size_t si = 0; si + 1 < sb.n; si++) {
            int32_t s0 = sb.b[si], s1 = sb.b[si + 1]; segs.n = 0;
            int32_t w0 = s0;
            while (w0 < s1) {
                while (wi < wb.n && wb.b[wi] <= w0) wi++;
                int32_t w1 = (wi < wb.n && wb.b[wi] < s1) ? wb.b[wi] : s1;   /* word boundaries clipped to the sentence */
                g.n = 0; int32_t g0 = w0;
                while (g0 < w1) {
                    while (gi < gb.n && gb.b[gi] <= g0) gi++;
                    int32_t g1 = (gi < gb.n && gb.b[gi] < w1) ? gb.b[gi] : w1;
                    Refs cps = {0}; size_t i = g0; uint32_t cp;
                    while (i < (size_t)g1) { utf8_next(src, n, &i, &cp); push(&cps, cp); }
                    push(&g, node(cps.v, cps.n, 1)); stat_refs[1] += cps.n > 1; free(cps.v);
                    g0 = g1;
                }
                push(&segs, node(g.v, g.n, 2)); stat_refs[2]++;
                w0 = w1;
            }
            push(&sents, node(segs.v, segs.n, 3)); stat_refs[3]++;
            /* paragraph ends when this sentence's trailing whitespace contains a blank line */
            int end_para = (si + 2 == sb.n);
            for (int32_t z = s1 - 1; z >= s0 && !end_para; z--) {
                if (src[z] != '\n' && src[z] != '\r' && src[z] != ' ' && src[z] != '\t') break;
                if ((src[z] == '\n' || src[z] == '\r') && cpy[z] != ' ') end_para = 1;
            }
            if (end_para) { push(&paras, node(sents.v, sents.n, 4)); stat_refs[4]++; sents.n = 0; }
        }
        if (sents.n) { push(&paras, node(sents.v, sents.n, 4)); stat_refs[4]++; }
        uint64_t trunk = node(paras.v, paras.n, 5); trunks[fi] = trunk;
        rlen = 0; expand(trunk);
        int exact = (rlen == n && !memcmp(rbuf, src, n));
        ok_files += exact; bad_files += !exact; total_bytes += n;
        { uint8_t sha[32]; blake3_hasher h; blake3_hasher_init(&h); blake3_hasher_update(&h, src, n); blake3_hasher_finalize(&h, sha, 32);
          fprintf(fsrc, "\\\\x"); hexbytes(fsrc, ref_id(trunk), 16); fprintf(fsrc, "\t%s\ttext\ttext/plain; charset=utf-8\t%zu\t\\\\x", fn, n); hexbytes(fsrc, sha, 32); fputc('\n', fsrc); }
        printf("[%7.1fs] %3d/%d %-58.58s %7.2f MB %6.2fs  recomposed %s  entities %llu\n", now() - t_start, fi + 1, nfiles,
               strrchr(fn, '/') ? strrchr(fn, '/') + 1 : fn, n / 1048576.0, now() - tf, exact ? "EXACT" : "MISMATCH", (unsigned long long)ncomp);
        fflush(stdout);
        utext_close(ut);
    next:
        free(src); free(cpy);
    }
    fclose(fsrc);

    /* occurrences (paths from sources down) and distinct parents, parents processed before children */
    double *occ_c = calloc(ncomp, 8), *occ_a = calloc(NCP, 8); uint64_t *par_c = calloc(ncomp, 8), *par_a = calloc(NCP, 8);
    for (int fi = 0; fi < nfiles; fi++) if (trunks[fi] >= NCP) occ_c[trunks[fi] - NCP] += 1;
    uint64_t *seen = NULL; size_t seen_cap = 0;
    for (uint64_t i = ncomp; i-- > 0; ) {
        const Comp *c = &comps[i];
        if (c->vcount > seen_cap) { seen_cap = c->vcount * 2; seen = xrealloc(seen, seen_cap * 8); }
        size_t ns = 0;
        for (uint32_t v = 0; v < c->vcount; v++) {
            uint64_t r = verts[c->vstart + v].ref; double add = occ_c[i] * verts[c->vstart + v].run;
            if (r < NCP) occ_a[r] += add; else occ_c[r - NCP] += add;
            int dup = 0; for (size_t k = 0; k < ns && !dup; k++) dup = (seen[k] == r);
            if (!dup && ns < 64) seen[ns++] = r; else if (!dup) { /* long lists: count all (approximation noted below) */ }
            if (!dup) { if (r < NCP) par_a[r]++; else par_c[r - NCP]++; }
        }
    }

    /* COPY files */
    snprintf(path, sizeof path, "%s/entity.copy", out); FILE *fe = fopen(path, "w");
    snprintf(path, sizeof path, "%s/physicality.copy", out); FILE *fp = fopen(path, "w");
    snprintf(path, sizeof path, "%s/entity_stats.copy", out); FILE *fs = fopen(path, "w");
    double xyz[3];
    for (uint64_t r = 0; r < NCP + ncomp; r++) {
        const uint8_t *id = ref_id(r); const int64_t *m = ref_m(r);
        fprintf(fe, "\\\\x"); hexbytes(fe, id, 16); fputc('\t', fe);
        fputs("01", fe); put_u32(fe, 0xC0000001u); for (int d = 0; d < 4; d++) put_d(fe, (double)m[d] / 9007199254740992.0);
        fprintf(fe, "\t%d\t%lld\n", ref_tier(r), (long long)(r < NCP ? t0[r].hilbert : hilbert(m)));
        fprintf(fp, "\\\\x"); hexbytes(fp, id, 16); fputc('\t', fp); fputs("01", fp);
        if (r < NCP) { id_point(id, xyz); put_u32(fp, 0xC0000001u); put_d(fp, xyz[0]); put_d(fp, xyz[1]); put_d(fp, xyz[2]); put_d(fp, 0); }
        else {
            const Comp *c = &comps[r - NCP];
            if (c->vcount == 1) { id_point(ref_id(verts[c->vstart].ref), xyz); put_u32(fp, 0xC0000001u); put_d(fp, xyz[0]); put_d(fp, xyz[1]); put_d(fp, xyz[2]); put_d(fp, verts[c->vstart].run); }
            else { put_u32(fp, 0xC0000002u); put_u32(fp, c->vcount);
                   for (uint32_t v = 0; v < c->vcount; v++) { id_point(ref_id(verts[c->vstart + v].ref), xyz); put_d(fp, xyz[0]); put_d(fp, xyz[1]); put_d(fp, xyz[2]); put_d(fp, verts[c->vstart + v].run); } }
        }
        fputc('\n', fp);
        double o = r < NCP ? occ_a[r] : occ_c[r - NCP]; uint64_t p = r < NCP ? par_a[r] : par_c[r - NCP];
        if (o > 0 || r >= NCP) { fprintf(fs, "\\\\x"); hexbytes(fs, id, 16); fprintf(fs, "\t%llu\t%.0f\n", (unsigned long long)p, o); }
    }
    fclose(fe); fclose(fp); fclose(fs);

    static const char *tn[6] = { "codepoint", "grapheme", "word segment", "sentence", "paragraph", "file" };
    printf("\n== summary: %d files, %.1f MB, %.1fs, recomposed exactly: %d, mismatched: %d, children-mismatch on ID hit: %llu\n",
           nfiles, total_bytes / 1048576.0, now() - t_start, ok_files, bad_files, (unsigned long long)stat_mismatch);
    printf("   %-13s %12s %14s %14s\n", "tier", "distinct", "new", "reused (hits)");
    for (int t = 1; t < 6; t++) printf("   %-13s %12llu %14llu %14llu\n", tn[t], (unsigned long long)stat_new[t], (unsigned long long)stat_new[t], (unsigned long long)stat_hit[t]);
    printf("   compositions %llu, vertices %llu (after run-length encoding)\n", (unsigned long long)ncomp, (unsigned long long)nvert);
    return bad_files ? 1 : 0;
}
