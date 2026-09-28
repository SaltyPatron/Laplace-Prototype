/* laplace_proto: decode entity IDs bit-packed into geometry ZM vertices (prototype of Laplace-postgres).
 * laplace_vertex_ids(ewkb bytea) -> bytea[]  distinct 16-byte entity IDs of a POINT ZM / LINESTRING ZM,
 * read from the X, Y, Z mantissas (43 + 43 + 42 bits). Used as a GIN expression index to find containers. */
#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "varatt.h"
#include "utils/builtins.h"
#include "mb/pg_wchar.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "blake3.h"
PG_MODULE_MAGIC;

/* ---- tier 0 perf-cache: tier0.bin memory-mapped once per backend, O(1) per codepoint ---- */
#define NCP 1114112u
typedef struct { unsigned char id[16]; int64 m[4]; uint64 hilbert; uint32 rank, pad; } T0;
static const T0 *t0 = NULL;
static const T0 *tier0(void){
    if (!t0) {
        const char *path = "/home/ahart/Projects/Laplace-Prototype/tier0/tier0.bin";
        int fd = open(path, O_RDONLY); struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0 || (size_t) st.st_size != NCP * sizeof(T0))
            ereport(ERROR, (errmsg("laplace: cannot map tier 0 at %s", path)));
        t0 = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0); close(fd);
    }
    return t0;
}
static bytea *id_bytea(const unsigned char *id){ bytea *b = palloc(VARHDRSZ + 16); SET_VARSIZE(b, VARHDRSZ + 16); memcpy(VARDATA(b), id, 16); return b; }

/* laplace_cp_id(int) -> bytea: the leaf ID of one codepoint */
PG_FUNCTION_INFO_V1(laplace_cp_id);
Datum laplace_cp_id(PG_FUNCTION_ARGS){
    int32 cp = PG_GETARG_INT32(0);
    if (cp < 0 || (uint32) cp >= NCP) ereport(ERROR, (errmsg("laplace_cp_id: %d is outside the codespace", cp)));
    PG_RETURN_BYTEA_P(id_bytea(tier0()[cp].id));
}

/* laplace_seq_id(text) -> bytea: ID of the composition of the text's codepoints, one-child collapse */
PG_FUNCTION_INFO_V1(laplace_seq_id);
Datum laplace_seq_id(PG_FUNCTION_ARGS){
    text *t = PG_GETARG_TEXT_PP(0); const unsigned char *s = (const unsigned char *) VARDATA_ANY(t);
    int len = VARSIZE_ANY_EXHDR(t), n = 0; const T0 *T = tier0();
    blake3_hasher h; blake3_hasher_init(&h); const unsigned char *first = NULL;
    for (int i = 0; i < len; ) {
        int l = pg_utf_mblen(s + i); pg_wchar cp = utf8_to_unicode(s + i); i += l;
        if (n == 0) first = T[cp].id; blake3_hasher_update(&h, T[cp].id, 16); n++;
    }
    if (n == 0) PG_RETURN_NULL();
    if (n == 1) PG_RETURN_BYTEA_P(id_bytea(first));
    { unsigned char id[16]; blake3_hasher_finalize(&h, id, 16); PG_RETURN_BYTEA_P(id_bytea(id)); }
}

static void decode(const unsigned char *p, unsigned char *id){
    double v[3]; uint64 part[3]; memcpy(v, p, 24);
    for (int k = 0; k < 3; k++) { uint64 bits; memcpy(&bits, &v[k], 8); part[k] = bits & ((1ULL << 52) - 1); }
    unsigned __int128 x = (unsigned __int128)part[0] | ((unsigned __int128)part[1] << 43) | ((unsigned __int128)part[2] << 86);
    for (int i = 0; i < 16; i++) { id[i] = (unsigned char)(x & 0xFF); x >>= 8; }
}

PG_FUNCTION_INFO_V1(laplace_vertex_ids);
Datum laplace_vertex_ids(PG_FUNCTION_ARGS){
    bytea *w = PG_GETARG_BYTEA_PP(0); const unsigned char *p = (const unsigned char *)VARDATA_ANY(w);
    int len = VARSIZE_ANY_EXHDR(w); uint32 type, n = 1; int off = 5;
    if (len < 5 || p[0] != 1) ereport(ERROR, (errmsg("laplace_vertex_ids: expected little-endian EWKB")));
    memcpy(&type, p + 1, 4);
    if (type & 0x20000000) off += 4;                                   /* SRID present */
    if ((type & 0xFF) == 2) { memcpy(&n, p + off, 4); off += 4; }
    else if ((type & 0xFF) != 1) ereport(ERROR, (errmsg("laplace_vertex_ids: expected POINT or LINESTRING")));
    if (!(type & 0x80000000) || !(type & 0x40000000)) ereport(ERROR, (errmsg("laplace_vertex_ids: expected ZM")));
    Datum *out = palloc(sizeof(Datum) * n); int m = 0;
    for (uint32 i = 0; i < n; i++) {
        unsigned char id[16]; decode(p + off + 32 * i, id);
        bool dup = false; for (int k = 0; k < m && !dup; k++) dup = memcmp(VARDATA(DatumGetPointer(out[k])), id, 16) == 0;
        if (dup) continue;
        bytea *b = palloc(VARHDRSZ + 16); SET_VARSIZE(b, VARHDRSZ + 16); memcpy(VARDATA(b), id, 16); out[m++] = PointerGetDatum(b);
    }
    PG_RETURN_ARRAYTYPE_P(construct_array(out, m, BYTEAOID, -1, false, TYPALIGN_INT));
}

/* ---- continuation: laplace_follows(ewkb bytea, phrase bytea[]) -> bytea[]
 * Every place the phrase occurs as a run inside a LINESTRING ZM path (vertices expanded by their M run
 * lengths), the ID of the vertex that follows it. The phrase is encoded once into the same X/Y/Z bytes the
 * vertices store, so matching compares raw vertex bytes (24 per vertex, SSE) and only continuations are
 * decoded. A path is one trajectory; a phrase is a shorter one; a match is a window at Fréchet distance 0. */
#include <immintrin.h>
static void encode(const unsigned char *id, unsigned char *xyz){
    unsigned __int128 x = 0; for (int i = 15; i >= 0; i--) x = (x << 8) | id[i];
    uint64 part[3] = { (uint64)(x & ((1ULL << 43) - 1)), (uint64)((x >> 43) & ((1ULL << 43) - 1)), (uint64)(x >> 86) };
    for (int k = 0; k < 3; k++) { uint64 bits = (1021ULL << 52) | part[k]; memcpy(xyz + 8 * k, &bits, 8); }
}
static inline bool same_xyz(const unsigned char *a, const unsigned char *b){         /* 24 bytes: one SSE compare + one 8-byte */
    __m128i x = _mm_loadu_si128((const __m128i *)a), y = _mm_loadu_si128((const __m128i *)b);
    return _mm_movemask_epi8(_mm_cmpeq_epi8(x, y)) == 0xFFFF && memcmp(a + 16, b + 16, 8) == 0;
}
PG_FUNCTION_INFO_V1(laplace_follows);
Datum laplace_follows(PG_FUNCTION_ARGS){
    bytea *w = PG_GETARG_BYTEA_PP(0); ArrayType *pa = PG_GETARG_ARRAYTYPE_P(1);
    const unsigned char *p = (const unsigned char *)VARDATA_ANY(w); int len = VARSIZE_ANY_EXHDR(w);
    uint32 type, n = 1; int off = 5; Datum *pe; bool *pn; int np;
    deconstruct_array(pa, BYTEAOID, -1, false, TYPALIGN_INT, &pe, &pn, &np);
    if (len < 5 || p[0] != 1 || np == 0) PG_RETURN_NULL();
    memcpy(&type, p + 1, 4);
    if (type & 0x20000000) off += 4;
    if ((type & 0xFF) == 2) { memcpy(&n, p + off, 4); off += 4; } else PG_RETURN_NULL();
    unsigned char *ph = palloc(24 * np);                                  /* the phrase, as vertex bytes */
    for (int j = 0; j < np; j++) encode((const unsigned char *)VARDATA_ANY(DatumGetPointer(pe[j])), ph + 24 * j);
    /* expand the path by run length into a vertex index sequence */
    int total = 0; const unsigned char *v = p + off;
    for (uint32 i = 0; i < n; i++) { double m; memcpy(&m, v + 32 * i + 24, 8); total += m < 1 ? 1 : (int) m; }
    int *seq = palloc(sizeof(int) * (total + 1)), t = 0;
    for (uint32 i = 0; i < n; i++) { double m; memcpy(&m, v + 32 * i + 24, 8); for (int r = 0; r < (m < 1 ? 1 : (int) m); r++) seq[t++] = i; }
    Datum *out = palloc(sizeof(Datum) * (total + 1)); int k = 0;
    for (int s = 0; s + np < total; s++) {
        if (!same_xyz(v + 32 * seq[s], ph)) continue;                     /* anchor on the first vertex */
        int j = 1; while (j < np && same_xyz(v + 32 * seq[s + j], ph + 24 * j)) j++;
        if (j < np) continue;
        unsigned char id[16]; decode(v + 32 * seq[s + np], id); out[k++] = PointerGetDatum(id_bytea(id));
    }
    if (k == 0) PG_RETURN_NULL();
    PG_RETURN_ARRAYTYPE_P(construct_array(out, k, BYTEAOID, -1, false, TYPALIGN_INT));
}
