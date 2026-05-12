#!/usr/bin/env bash
# Same scenario as 01, but with the Hugging Face backend
# (README.claim-extractor.md Initialization section mentions `backend="hf"`
# as an alternative to vllm).

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras claim-extractor-hf || true
require_gpu || true

log "running README.claim-extractor.md quickstart with hf backend, model=$CLAIM_EXTRACTOR_MODEL"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    backend="hf",
    model="${CLAIM_EXTRACTOR_MODEL}",
)

ai_service_description = """
You are a virtual assistant for a parcel delivery service.
You can only answer questions about package tracking.
"""

assistant_message = (
    "Your package with tracking number 1234567890 is currently in transit and "
    "is expected to be delivered on December 12, 2025. If you want, I can also "
    "notify you when it is out for delivery."
)

result = ce.extract(
    assistant_message,
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
assert_contains "$OUTPUT" "--Claims--"  "expected --Claims-- header in output"
assert_contains "$OUTPUT" "--Intents--" "expected --Intents-- header in output"
assert_contains "$OUTPUT" "[Factoid]"   "expected at least one Factoid claim"
log_pass "hf quickstart produces the expected output structure"
