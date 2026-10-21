#!/usr/bin/env bash
# Non-destructive Gates A/B; pass explicit local binaries and scratch DB.
set -euo pipefail
exec pwsh -NoProfile -File "$(dirname "${BASH_SOURCE[0]}")/tests/run-proofs.ps1" "$@"
