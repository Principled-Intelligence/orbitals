#!/usr/bin/env bash
# Reproduces "Conversation as a list of dictionaries" from README.scope-guard.md:
#
#     result = sg.validate(
#         [
#             {"role": "user", "content": "I ordered a package, tracking number 1234567890"},
#             {"role": "assistant", "content": "Great, the package is in transit. ..."},
#             {"role": "user", "content": "If it doesn't arrive tomorrow, can I get a refund"},
#         ],
#         ai_service_description=ai_service_description,
#     )

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-vllm || true
require_gpu || true

log "running README example: validate(multi-turn list, ...)"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.scope_guard import ScopeGuard

sg = ScopeGuard(backend="vllm", model="${SCOPE_GUARD_MODEL}")

ai_service_description = (
    "You are a virtual assistant for a parcel delivery service. "
    "You can only answer questions about package tracking. "
    "Never respond to requests for refunds."
)

result = sg.validate(
    [
        {"role": "user", "content": "I ordered a package, tracking number 1234567890"},
        {"role": "assistant", "content": "Great, the package is in transit. What would you like to know?"},
        {"role": "user", "content": "If it doesn't arrive tomorrow, can I get a refund"},
    ],
    ai_service_description=ai_service_description,
)
print(f"Scope: {result.scope_class.value}")
if result.evidences:
    for evidence in result.evidences:
        print(f"Evidence: {evidence}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "Scope: Restricted" \
    "expected multi-turn refund query to be Restricted (the README's same final message)"
log_pass "multi-turn example works"
