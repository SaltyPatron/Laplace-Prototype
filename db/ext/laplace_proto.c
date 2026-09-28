/* laplace_proto: decode entity IDs bit-packed into geometry ZM vertices (prototype of Laplace-postgres).
 * laplace_vertex_ids(ewkb bytea) -> bytea[]  distinct 16-byte entity IDs of a POINT ZM / LINESTRING ZM,
 * read from the X, Y, Z mantissas (43 + 43 + 42 bits). Used as a GIN expression index to find containers. */
#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "varatt.h"
PG_MODULE_MAGIC;

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
