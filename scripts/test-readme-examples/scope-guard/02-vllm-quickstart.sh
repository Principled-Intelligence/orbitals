#!/usr/bin/env bash
# Reproduces the vLLM quickstart in README.scope-guard.md, including the
# expected output:
#     # Scope: Restricted
#     # Evidences:
#     #   - Never respond to requests for refunds.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-vllm || true
require_gpu || true

log "running README.scope-guard.md vLLM quickstart with model=$SCOPE_GUARD_MODEL"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.scope_guard import ScopeGuard

sg = ScopeGuard(
    backend="vllm",
    model="${SCOPE_GUARD_MODEL}",
)

ai_service_description = """
You are a virtual assistant for a parcel delivery service.
You can only answer questions about package tracking.
Never respond to requests for refunds.
"""

user_query = "If the package hasn't arrived by tomorrow, can I get my money back?"
result = sg.validate(user_query, ai_service_description=ai_service_description)

print(f"Scope: {result.scope_class.value}")
if result.evidences:
    print("Evidences:")
    for evidence in result.evidences:
        print(f"  - {evidence}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "Scope: Restricted" "expected the refund query to be classified Restricted"
assert_contains "$OUTPUT" "Evidences:" "expected at least one evidence in the output"
log_pass "vLLM quickstart matches README expected output"
