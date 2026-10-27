# Laplace native proofs

This repository verifies the production Native, Engine and PostgreSQL implementations.
It contains the database round-trip and self-contained Highway proofs, plus native
model-export consumer checks. Historical measurements remain in
[Laplace-Wiki Research](https://saltypatron.github.io/Laplace-Wiki/Research/).

- [tests/README.md](tests/README.md) describes Gates A/B, their native reader,
  real Engine ingestion, fixture reconstruction and deduplication receipts.
- `rebuild.sh` forwards explicit binaries, caches and a test database to that runner.
- [models/README.md](models/README.md) describes native CPU model consumers,
  tensor/state verification and runtime measurements.
- `tier0/fingerprint.txt` identifies the canonical scalar cache.

Generate Tier-0 with the installed native Engine and existing local Unicode data:

```powershell
laplace tier0 -u E:\Data\Ingest\UCD -o E:\Data\Laplace\work\tier0\tier0.bin
```

The Native owner defines Unicode ordering, CORE-MATH projection, exact fixed-point
quantization and cache validation. The cache contains 1,112,064 Unicode scalar
records, sorted by content ID. The removed Python coordinate generators,
and independent prototype ingester/schema are available in Git history; they are
superseded by Native, Engine and the production PostgreSQL extension.

Historical experiments in `chess/`, `code/`, `recipes/`, `semantics/`, and the
Python model/test tools remain research references. Their presence does not imply
equivalent native implementation or completed acceptance. Gates A/B cover the
maintained storage and Highway proofs; full-corpus Gate C remains pending.

Local operation uses the host's configured PostgreSQL service. The proof runner
uses an explicitly named temporary database in an existing cluster; CI or the
operator owns its creation and removal. This repository does not provision a
separate PostgreSQL instance. Data and build artifacts live outside the repository.
