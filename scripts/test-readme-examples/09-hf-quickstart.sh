#!/usr/bin/env bash
# Hugging Face backend quickstart, mirroring the vLLM quickstart in
# README.scope-guard.md (same example but with backend="huggingface").
# README mentions: pip install orbitals[scope-guard-hf]

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-hf || true
require_gpu || true

log "running README quickstart with backend=huggingface, model=$SCOPE_GUARD_MODEL"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.scope_guard import ScopeGuard

sg = ScopeGuard(
    backend="hf",
    model="${SCOPE_GUARD_MODEL}",
)

ai_service_description = """
You are a virtual assistant for a parcel delivery service.
You can only answer questions about package tracking.
Never respond to requests for refunds.
"""

result = sg.validate(
    "If the package hasn't arrived by tomorrow, can I get my money back?",
    ai_service_description=ai_service_description,
)
print(f"Scope: {result.scope_class.value}")
if result.evidences:
    for evidence in result.evidences:
        print(f"  - {evidence}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "Scope: Restricted" \
    "expected refund query to be Restricted with the hf backend too"
log_pass "Hugging Face backend quickstart works"
