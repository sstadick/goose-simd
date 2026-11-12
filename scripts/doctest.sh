#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Use the same goose-doc checkout as docs.sh.
exec python3 "$(dirname "${GOOSE_DOC:-../goose-doc/main.goose}")/doctest.py" \
  --symbols api.txt --preamble test/doc_preamble.goose --require-examples \
  "$@" simd_bytes.goose simd_int.goose
