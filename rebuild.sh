#!/bin/bash
# Full prototype rebuild: compile against ICU 78 (Unicode 17), ingest, load, index, verify, query.
set -euo pipefail
R=$HOME/Projects/Laplace-Prototype; O=/vault/Data/LaplacePrototype/out/gutenberg; I=$R/lib/icu78; B=/repos/src/blake3/c
P=(psql -h /tmp -p 5439 -U laplace -d laplace -v ON_ERROR_STOP=1 -q)
say(){ echo "[$(date +%T)] $*"; }
say "compiling ingest against ICU $(PKG_CONFIG_PATH=$I/lib/pkgconfig pkg-config --modversion icu-uc)"
gcc -O2 -march=native -Wall -Wno-unused-result -o $R/dag/ingest $R/dag/ingest.c -I$B $R/lib/libblake3.a -I$I/include -L$I/lib -licui18n -licuuc -licudata -Wl,--disable-new-dtags,-rpath,$I/lib
rm -rf $O && mkdir -p $O
say "ingesting 195 files"
stdbuf -oL $R/dag/ingest $R/tier0/tier0.bin $O /vault/Data/ProjectGutenberg/text/*.txt > $O/ingest.log
sed -n '/== summary/,$p' $O/ingest.log
say "recreating tables"
"${P[@]}" -c "drop table if exists entity_stats, physicality, source, entity cascade"
"${P[@]}" -f $R/db/schema.sql 2>/dev/null
say "loading"; $R/db/load.sh $O
say "indexing"; "${P[@]}" -f $R/db/indexes.sql | grep -c Time | xargs -I{} echo "  {} index steps done"
say "verifying"; python3 -u $R/tests/verify.py | tee $O/verify.log | grep -E 'PASS|FAIL|=='
say "queries"; python3 -u $R/tests/queries.py > $O/queries.log; python3 -u $R/tests/gap_query.py >> $O/queries.log
say "done"
