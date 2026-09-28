# Laplace storage prototype

A local test bench for the Laplace storage layer (not Laplace-Engine). See the wiki's Research: Prototype page for results.

- `tier0/gen_tier0.py` — tier 0 table (DUCET order, H1 placement, exact fixed-point coordinates, BLAKE3-128 IDs, Hilbert values) and fingerprint
- `dag/ingest.c` — decomposer and ingestion (ICU 78 UAX #29, BLAKE3, integer centroids, run-length paths, recomposition check, COPY output)
- `db/` — schema, extension (`laplace_vertex_ids`), load and index scripts; PostgreSQL 18 cluster at /vault/Data/LaplacePrototype/pgdata, port 5439
- `tests/` — verification (`verify.py`), queries (`queries.py`, `gap_query.py`), UCD break conformance (`breaktest.c`)
- `rebuild.sh` — full rebuild: compile, ingest, load, index, verify, query

Start the database: `/usr/lib/postgresql/18/bin/pg_ctl -D /vault/Data/LaplacePrototype/pgdata -l /vault/Data/LaplacePrototype/pg.log start`
