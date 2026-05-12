#!/usr/bin/env bash
# Reproduces "Extracting from every turn of a conversation" from README.claim-extractor.md.
# The /extract-conversation endpoint returns a *list* of extractions — one per turn.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_curl
require_extras claim-extractor-serve || true
require_gpu || true

trap_stop_server
start_server_if_needed

URL="$(server_url)/orbitals/claim-extractor/extract-conversation"
log "POST $URL"

RESPONSE="$(curl -fsS -X POST "$URL" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "conversation": [
            {"role": "user", "content": "I ordered a package, tracking number 1234567890."},
            {"role": "assistant", "content": "Your package is in transit and will arrive on December 12, 2025."}
        ],
        "ai_service_description": "You are a virtual assistant for a parcel delivery service."
    }')"

echo "$RESPONSE"
assert_contains "$RESPONSE" '"extractions"' "response missing extractions"
assert_contains "$RESPONSE" '['             "extractions field should be a JSON array (one entry per turn)"
assert_contains "$RESPONSE" '"claims"'      "response missing claims"
assert_contains "$RESPONSE" '"intents"'     "response missing intents"
assert_contains "$RESPONSE" '"model"'       "response missing model"
assert_contains "$RESPONSE" '"time_taken"'  "response missing time_taken"
log_pass "cURL POST /orbitals/claim-extractor/extract-conversation matches README contract"
