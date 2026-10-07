# Laplace storage prototype

A local test bench for the Laplace storage layer (not Laplace-Engine). The results are on the wiki's [Research: Prototype](https://github.com/SaltyPatron/Laplace-Wiki/blob/main/Research/Prototype.md) page, published at <https://saltypatron.github.io/Laplace-Wiki/Research/Prototype/>.

- `tier0/gen_tier0.py` — tier 0 table (DUCET order, H1 placement, exact fixed-point coordinates, BLAKE3-128 IDs, Hilbert values) and fingerprint
- `dag/ingest.c` — decomposer and ingestion (ICU 78 UAX #29, BLAKE3, integer centroids, run-length paths, recomposition check, COPY output)
- `db/` — schema, extension (`laplace_vertex_ids`), load and index scripts; PostgreSQL 18 cluster at /vault/Data/LaplacePrototype/pgdata, port 5439
- `tests/` — verification (`verify.py`), queries (`queries.py`, `gap_query.py`), UCD break conformance (`breaktest.c`)
- `rebuild.sh` — full rebuild: compile, ingest, load, index, verify, query
- `chess/` — Glicko-2 ratings from observed games, per game mode, replayed in play order
- `chess/order_free.py` — order-free ratings: intrinsic move quality (`intrinsic.py`, Regan & Haworth 2011, from `analyze.py`'s Stockfish MultiPV analysis of games chosen by `select_games.py`) and Whole-History Rating (`whr.py`, Coulom 2008) with stated Elo and intrinsic estimates as separate observation channels, scored against Glicko-2 (`glicko2.py`) on TWIC classical (`pool.py`), with shuffled-order and late-arrival tests
- `semantics/` — word sense disambiguation by pull (WordNet highway + SemCor witness) on the Raganato et al. framework
- `code/` — Merkle AST over PostgreSQL and CPython with tree-sitter (exact recomposition, subtree sharing)
- `recipes/` — PNG recipe (pixel trees, byte-exact recomposition via preflate), quadtree storage, JPEG coefficient deduplication on COCO

Data lives outside the repository. What each measurement read is recorded with the measurement, under [Research](https://github.com/SaltyPatron/Laplace-Wiki/blob/main/Research/README.md) in Laplace-Wiki.

Start the database: `/usr/lib/postgresql/18/bin/pg_ctl -D /vault/Data/LaplacePrototype/pgdata -l /vault/Data/LaplacePrototype/pg.log start`
