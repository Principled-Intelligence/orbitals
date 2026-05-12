#!/usr/bin/env bash
# Reproduces "Batch Processing — Single AI Service Description" from README.claim-extractor.md.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras claim-extractor-vllm || true
require_gpu || true

log "running batch_extract with a single ai_service_description, model=$CLAIM_EXTRACTOR_MODEL"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(backend="vllm", model="${CLAIM_EXTRACTOR_MODEL}")

ai_service_description = (
    "You are a virtual assistant for a parcel delivery service. "
    "You can only answer questions about package tracking."
)

messages = [
    "Your package is in transit and will arrive on December 12, 2025.",
    "Order #42 has already shipped and cannot be cancelled anymore.",
]

results = ce.batch_extract(
    messages,
    ai_service_description=ai_service_description,
)

print(f"len={len(results)}")
for i, r in enumerate(results):
    n_claims = len(r.extractions.claims)
    n_intents = len(r.extractions.intents)
    print(f"[{i}] claims={n_claims} intents={n_intents}")
    for claim in r.extractions.claims:
        print(f"[{i}][{claim.subtype}] {claim.content}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "len=2"    "expected exactly 2 results from batch_extract"
assert_contains "$OUTPUT" "[0] claims=" "expected result 0 to be present"
assert_contains "$OUTPUT" "[1] claims=" "expected result 1 to be present"
log_pass "batch_extract with single ai_service_description works"
