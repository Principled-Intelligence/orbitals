#!/usr/bin/env bash
# Reproduces the "Asynchronous API client" example in README.scope-guard.md:
#
#     sg = AsyncScopeGuard(backend="api", api_url="http://localhost:8000")
#     result = await sg.validate("...", ai_service_description="...")

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-serve || true
require_gpu || true

trap_stop_server
start_server_if_needed

URL="$(server_url)"
log "running AsyncScopeGuard(backend='api', api_url='$URL').validate(...)"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
import asyncio

from orbitals.scope_guard import AsyncScopeGuard

async def main() -> None:
    sg = AsyncScopeGuard(backend="api", api_url="${URL}")
    result = await sg.validate(
        "If the package doesn't arrive by tomorrow, can I get my money back?",
        ai_service_description=(
            "You are a virtual assistant for a parcel delivery service. "
            "You can only answer questions about package tracking. "
            "Never respond to requests for refunds."
        ),
    )
    print(f"Scope: {result.scope_class.value}")
    print(f"Model: {result.model}")

asyncio.run(main())
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "Scope: Restricted" "expected Restricted via the async api client"
log_pass "Asynchronous API client example works"
