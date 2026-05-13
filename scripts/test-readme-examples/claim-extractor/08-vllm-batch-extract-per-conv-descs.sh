#!/usr/bin/env bash
# Reproduces "Batch Processing — Multiple AI Service Descriptions" from README.claim-extractor.md.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras claim-extractor-vllm || true
require_gpu || true

log "running batch_extract with per-conversation ai_service_descriptions, model=$CLAIM_EXTRACTOR_MODEL"
cd "$REPO_ROOT"

OUTPUT="$(uv_run_python <<PY
from orbitals.claim_extractor import ClaimExtractor

if __name__ == "__main__":
    ce = ClaimExtractor(backend="vllm", model="${CLAIM_EXTRACTOR_MODEL}")

    ai_service_descriptions = [
        "You are a virtual assistant for Postal Service. You only answer questions about package tracking.",
        "You are a virtual assistant for an e-commerce platform. You help users manage their orders.",
    ]

    messages = [
        "Your package is in transit and will arrive on December 12, 2025.",
        "Order #42 has already shipped and cannot be cancelled anymore.",
    ]

    results = ce.batch_extract(
        messages,
        ai_service_descriptions=ai_service_descriptions,
    )

    print(f"len={len(results)}")
    for i, r in enumerate(results):
        n_claims = len(r.extractions.claims)
        n_intents = len(r.extractions.intents)
        print(f"[{i}] claims={n_claims} intents={n_intents}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "len=2"       "expected exactly 2 results from batch_extract"
assert_contains "$OUTPUT" "[0] claims=" "expected result 0 to be present"
assert_contains "$OUTPUT" "[1] claims=" "expected result 1 to be present"
log_pass "batch_extract with per-conversation ai_service_descriptions works"
