#!/usr/bin/env bash
# Reproduces "Single message as a dictionary" from README.claim-extractor.md.
# Dict input with {"role": "user", ...} — user messages can yield intents.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras claim-extractor-vllm || true
require_gpu || true

log "extracting from a single dict user message with model=$CLAIM_EXTRACTOR_MODEL"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(backend="vllm", model="${CLAIM_EXTRACTOR_MODEL}")

ai_service_description = (
    "You are a virtual assistant for a parcel delivery service. "
    "You can only answer questions about package tracking."
)

result = ce.extract(
    {
        "role": "user",
        "content": "Can I still cancel order #42 before it ships?",
    },
    ai_service_description=ai_service_description,
)

print("--Claims--")
for claim in result.extractions.claims:
    print(f"[{claim.subtype}] {claim.content}")

print("--Intents--")
for intent in result.extractions.intents:
    print(f"[Intent] {intent.content}")
print("--End--")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "--Claims--"  "expected --Claims-- header"
assert_contains "$OUTPUT" "--Intents--" "expected --Intents-- header"
assert_contains "$OUTPUT" "[Intent]"    "expected at least one Intent from a user message"
log_pass "user-dict input produces intents"
