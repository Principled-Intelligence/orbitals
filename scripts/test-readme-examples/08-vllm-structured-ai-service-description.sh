#!/usr/bin/env bash
# Exercises the AIServiceDescription structured object recommended in the
# "AI Service Description" section of README.scope-guard.md.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-vllm || true
require_gpu || true

log "running README example: validate(..., ai_service_description=AIServiceDescription(...))"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<PY
from orbitals.scope_guard import ScopeGuard
from orbitals.types import AIServiceDescription

sg = ScopeGuard(backend="vllm", model="${SCOPE_GUARD_MODEL}")

svc = AIServiceDescription(
    identity_role="Virtual assistant for a parcel delivery service",
    context="Customer-facing chatbot for an e-commerce logistics company.",
    knowledge_scope="Package tracking, shipping status, delivery ETAs.",
    functionalities=["Track packages", "Estimate delivery time"],
    principles=[
        "Never respond to requests for refunds.",
        "Do not share customer PII.",
    ],
)

result = sg.validate(
    "If the package hasn't arrived by tomorrow, can I get my money back?",
    ai_service_description=svc,
)
print(f"Scope: {result.scope_class.value}")
if result.evidences:
    for ev in result.evidences:
        print(f"Evidence: {ev}")
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "Scope: Restricted" \
    "expected refund query to be Restricted under the structured description"
log_pass "structured AIServiceDescription example works"
