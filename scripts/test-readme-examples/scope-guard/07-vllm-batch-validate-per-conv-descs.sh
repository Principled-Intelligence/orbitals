#!/usr/bin/env bash
# Reproduces "Multiple AI Service Descriptions" from README.scope-guard.md:
#
#     ai_service_descriptions = [...]
#     result = sg.batch_validate(queries, ai_service_descriptions=ai_service_descriptions)

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-vllm || true
require_gpu || true

log "running README example: batch_validate(..., ai_service_descriptions=[...])"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.scope_guard import ScopeGuard

sg = ScopeGuard(backend="vllm", model="${SCOPE_GUARD_MODEL}")

ai_service_descriptions = [
    "You are a virtual assistant for Postal Service. You only answer questions about package tracking. Never respond to refund requests.",
    "You are a virtual assistant for a Courier. You answer questions about package tracking. Never respond to refund requests.",
]
queries = [
    "If the package hasn't arrived by tomorrow, can I get my money back?",
    "When is the package expected to be delivered?",
]
results = sg.batch_validate(queries, ai_service_descriptions=ai_service_descriptions)
print(f"len={len(results)}")
for i, r in enumerate(results):
    print(f"[{i}] scope={r.scope_class.value}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "len=2"          "expected 2 results"
assert_contains "$OUTPUT" "[0] scope="     "first result missing"
assert_contains "$OUTPUT" "[1] scope="     "second result missing"
log_pass "batch_validate with per-conversation descriptions works"
