#!/usr/bin/env bash
# Same shape as 10, but hits /batch-validate (also referenced in
# README.scope-guard.md alongside /validate).

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_curl
require_extras scope-guard-serve || true
require_gpu || true

trap_stop_server
start_server_if_needed

URL="$(server_url)/orbitals/scope-guard/batch-validate"
log "POST $URL"

RESPONSE="$(curl -fsS -X POST "$URL" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "conversations": [
            "If the package does not arrive by tomorrow, can I get my money back?",
            "When is the package expected to be delivered?"
        ],
        "ai_service_description": "You are a virtual assistant for a parcel delivery service. You can only answer questions about package tracking. Never respond to requests for refunds."
    }')"

echo "$RESPONSE"
assert_contains "$RESPONSE" '"scope_class"' "response missing scope_class"
# Two-element JSON array, so it must contain the opening bracket.
assert_contains "$RESPONSE" '['             "expected a JSON array response"
log_pass "cURL POST /orbitals/scope-guard/batch-validate works"
