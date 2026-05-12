#!/usr/bin/env bash
# Reproduces "Conversation as a list of dictionaries" from README.claim-extractor.md.
# A multi-turn list — extractions come only from the LAST message.
# In the README example the last message is from the assistant, so we expect claims (not intents).

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras claim-extractor-vllm || true
require_gpu || true

log "extracting from a multi-turn conversation (last message: assistant) with model=$CLAIM_EXTRACTOR_MODEL"
cd "$REPO_ROOT"

OUTPUT="$(uv_run_python <<PY
from orbitals.claim_extractor import ClaimExtractor

if __name__ == "__main__":
    ce = ClaimExtractor(backend="vllm", model="${CLAIM_EXTRACTOR_MODEL}")

    ai_service_description = (
        "You are a virtual assistant for a parcel delivery service. "
        "You can only answer questions about package tracking."
    )

    result = ce.extract(
        [
            {
                "role": "user",
                "content": "I ordered a package, tracking number 1234567890.",
            },
            {
                "role": "assistant",
                "content": "Your package is in transit and will arrive on December 12, 2025.",
            },
        ],
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
assert_contains "$OUTPUT" "[Factoid]"   "expected at least one Factoid from the assistant turn"
log_pass "multi-turn conversation produces claims on the last (assistant) turn"
