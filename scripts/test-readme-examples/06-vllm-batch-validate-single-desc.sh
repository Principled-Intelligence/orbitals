#!/usr/bin/env bash
# Reproduces "Single AI Service Description" from README.scope-guard.md:
#
#     queries = [
#         "If the package hasn't arrived by tomorrow, can I get my money back?",
#         "When is the package expected to be delivered?"
#     ]
#     result = sg.batch_validate(queries, ai_service_description=ai_service_description)

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-vllm || true
require_gpu || true

log "running README example: batch_validate(..., ai_service_description=...)"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.scope_guard import ScopeGuard

sg = ScopeGuard(backend="vllm", model="${SCOPE_GUARD_MODEL}")

ai_service_description = (
    "You are a virtual assistant for a parcel delivery service. "
    "You can only answer questions about package tracking. "
    "Never respond to requests for refunds."
)

queries = [
    "If the package hasn't arrived by tomorrow, can I get my money back?",
    "When is the package expected to be delivered?",
]
results = sg.batch_validate(queries, ai_service_description=ai_service_description)
print(f"len={len(results)}")
for i, r in enumerate(results):
    print(f"[{i}] scope={r.scope_class.value}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "len=2"          "batch_validate should return 2 results"
assert_contains "$OUTPUT" "[0] scope="     "first result missing"
assert_contains "$OUTPUT" "[1] scope="     "second result missing"
log_pass "batch_validate with single description works"
