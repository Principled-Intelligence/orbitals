#!/usr/bin/env bash
# Reproduces "User query as a string" from README.scope-guard.md:
#
#     result = sg.validate(
#         "When is my package scheduled to arrive?",
#         ai_service_description=ai_service_description
#     )

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-vllm || true
require_gpu || true

log "running README example: validate(string, ...)"
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
    "When is my package scheduled to arrive?",
    ai_service_description=ai_service_description,
)
print(f"Scope: {result.scope_class.value}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "Scope:" "string-input example should produce a Scope line"
log_pass "string-input example works"
