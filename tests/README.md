Gates A/B use the real modular Engine writer, a separate native reader, and a
private PostgreSQL 18 database with PostGIS and the candidate laplace extension.
The runner accepts only an empty `laplace_proof_*` database and never drops or
resets one. Set `max_prepared_transactions=0` to make any 2PC attempt fail.

Build `tests/CMakeLists.txt` with the local developer compiler, providing
`LAPLACE_INCLUDE_DIR`, `LAPLACE_LIBRARY`, `PostgreSQL_ROOT`, and `ICU_ROOT`.
On Windows, load `Laplace-Operations/laplace.env.ps1` and `$env:LAPLACE_DEVSHELL`
before CMake, and put the candidate Native DLL directory on `PATH`.

Run `tests/run-proofs.ps1` with `-Engine`, `-NativeProof`, `-Tier0`, `-Highway`,
`-Formats`, `-ConnInfo`, `-Psql`, and `-Scratch`. `-Formats` points at the canonical
Engine `recipes/formats` directory. The database must use the same candidate
Tier-0 and Highway paths in its `laplace.tier0` and `laplace.highway` settings.
The Windows fixture proof also uses `LAPLACE_GRAMMARS` from the Engine build.
`rebuild.sh` forwards these arguments to the same PowerShell runner.

The receipt records exact fixture hashes, database versions, nine byte-exact
reconstructions, persisted structural checks, actual language claims, unchanged
standings and zero-write file-trunk dedupe. Gate B rebuilds with absent corpus
and recipe roots, verifies all canonical slots and CSR endpoints, checks real
connectivity across the eight namespaces, and rejects corrupt/missing artifacts
before database access. There is no timing or multi-corpus performance gate.

Gate C remains the Sprint 3 full-corpus qualification. Historical exploratory
scripts are not used as acceptance checks by this entry point.
