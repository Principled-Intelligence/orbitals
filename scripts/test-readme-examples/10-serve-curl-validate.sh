#!/usr/bin/env bash
# Reproduces "Direct HTTP Requests" from README.scope-guard.md:
#
#     curl -X 'POST' \
#         'http://localhost:8000/orbitals/scope-guard/validate' ...

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_curl
require_extras scope-guard-serve || true
require_gpu || true

trap_stop_server
start_server_if_needed

URL="$(server_url)/orbitals/scope-guard/validate"
log "POST $URL"

RESPONSE="$(curl -fsS -X POST "$URL" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "conversation": "If the package does not arrive by tomorrow, can I get my money back?",
        "ai_service_description": "You are a virtual assistant for a parcel delivery service. You can only answer questions about package tracking. Never respond to requests for refunds."
    }')"

echo "$RESPONSE"
assert_contains "$RESPONSE" '"scope_class"' "response missing scope_class"
assert_contains "$RESPONSE" '"evidences"'   "response missing evidences"
assert_contains "$RESPONSE" '"model"'       "response missing model"
assert_contains "$RESPONSE" '"time_taken"'  "response missing time_taken"
assert_contains "$RESPONSE" '"Restricted"'  "expected refund query to be Restricted"
log_pass "cURL POST /orbitals/scope-guard/validate matches README contract"
