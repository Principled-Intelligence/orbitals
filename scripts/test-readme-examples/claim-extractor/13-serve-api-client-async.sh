#!/usr/bin/env bash
# Reproduces the asynchronous Python SDK example from README.claim-extractor.md:
#
#     ce = AsyncClaimExtractor(backend="api", api_url="http://localhost:8000")
#     result = await ce.extract(...)

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras serving || true
require_gpu || true

trap_stop_server
start_server_if_needed

URL="$(server_url)"
log "calling /extract via AsyncClaimExtractor(backend=\"api\", api_url=$URL)"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
import asyncio
from orbitals.claim_extractor import AsyncClaimExtractor

async def main() -> None:
    ce = AsyncClaimExtractor(
        backend="api",
        api_url="${URL}",
    )

    result = await ce.extract(
        "Your package is in transit and will arrive on December 12, 2025.",
        ai_service_description="You are a virtual assistant for a parcel delivery service. You can only answer questions about package tracking.",
    )

    print("--Claims--")
    for claim in result.extractions.claims:
        print(f"[{claim.subtype}] {claim.content}")

    print("--Intents--")
    for intent in result.extractions.intents:
        print(f"[Intent] {intent.content}")
    print(f"Model: {result.model}")
    print("--End--")

asyncio.run(main())
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "--Claims--"  "expected --Claims-- header"
assert_contains "$OUTPUT" "[Factoid]"   "expected at least one Factoid claim via the async api backend"
assert_contains "$OUTPUT" "Model:"      "expected the response to carry a model name"
log_pass "async api client returns the expected output"
