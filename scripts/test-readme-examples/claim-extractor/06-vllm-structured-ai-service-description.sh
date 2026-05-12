#!/usr/bin/env bash
# Reproduces "AI Service Description" structured form from README.claim-extractor.md.
# Uses orbitals.types.AIServiceDescription instead of a free-form string.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras claim-extractor-vllm || true
require_gpu || true

log "running structured AIServiceDescription example with model=$CLAIM_EXTRACTOR_MODEL"
cd "$REPO_ROOT"

OUTPUT="$(uv_run_python <<PY
from orbitals.claim_extractor import ClaimExtractor
from orbitals.types import AIServiceDescription

if __name__ == "__main__":
    ce = ClaimExtractor(backend="vllm", model="${CLAIM_EXTRACTOR_MODEL}")

    ai_service_description = AIServiceDescription(
        identity_role="Virtual assistant for a parcel delivery service.",
        context="Operates on the company website; users are customers checking on shipments.",
        knowledge_scope="Package tracking, delivery windows, redelivery options.",
        functionalities=["Tracking lookup", "Out-for-delivery notifications"],
        principles="Only answer questions related to package tracking.",
        website_url="https://example.com",
    )

    assistant_message = (
        "Your package with tracking number 1234567890 is currently in transit and "
        "is expected to be delivered on December 12, 2025. If you want, I can also "
        "notify you when it is out for delivery."
    )

    result = ce.extract(assistant_message, ai_service_description=ai_service_description)

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
assert_contains "$OUTPUT" "[Factoid]"   "expected at least one Factoid claim"
log_pass "structured AIServiceDescription accepted and produces extractions"
