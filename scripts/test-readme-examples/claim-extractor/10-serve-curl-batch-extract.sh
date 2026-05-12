#!/usr/bin/env bash
# Exercises the /batch-extract endpoint mentioned in README.claim-extractor.md.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_curl
require_extras claim-extractor-serve || true
require_gpu || true

trap_stop_server
start_server_if_needed

URL="$(server_url)/orbitals/claim-extractor/batch-extract"
log "POST $URL"

RESPONSE="$(curl -fsS -X POST "$URL" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "conversations": [
            "Your package is in transit and will arrive on December 12, 2025.",
            "Order #42 has already shipped and cannot be cancelled anymore."
        ],
        "ai_service_description": "You are a virtual assistant for a parcel delivery service. You can only answer questions about package tracking."
    }')"

echo "$RESPONSE"
assert_contains "$RESPONSE" '['             "response should be a JSON array"
assert_contains "$RESPONSE" '"extractions"' "response missing extractions"
assert_contains "$RESPONSE" '"claims"'      "response missing claims"
assert_contains "$RESPONSE" '"intents"'     "response missing intents"
assert_contains "$RESPONSE" '"model"'       "response missing model"
assert_contains "$RESPONSE" '"time_taken"'  "response missing time_taken"
log_pass "cURL POST /orbitals/claim-extractor/batch-extract works"
