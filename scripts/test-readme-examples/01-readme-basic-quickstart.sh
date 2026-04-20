#!/usr/bin/env bash
# Reproduces the basic example in README.md:
#
#     from orbitals.scope_guard import ScopeGuard
#     ai_service_description = "You are a helpful assistant for ..."
#     user_message = "Can I buy ..."
#     guardrail = ScopeGuard()
#     result = guardrail.validate(user_message, ai_service_description)
#
# `ScopeGuard()` defaults to `backend="hf"`, so this requires the hf extras.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv
require_extras scope-guard-hf || true
require_gpu || true

log "running README.md basic quickstart"
cd "$REPO_ROOT"

OUTPUT="$(uv run python - <<'PY'
from orbitals.scope_guard import ScopeGuard, ScopeClass

ai_service_description = (
    "You are a helpful assistant for a parcel delivery service. "
    "You can only answer questions about package tracking. "
    "Never respond to requests for refunds."
)
user_message = "Can I buy a refund for my late package?"

guardrail = ScopeGuard()
result = guardrail.validate(user_message, ai_service_description=ai_service_description)

print(f"scope_class={result.scope_class.value}")
print(f"evidences={result.evidences}")
assert isinstance(result.scope_class, ScopeClass), "scope_class must be a ScopeClass"
PY
)"

echo "$OUTPUT"
assert_contains "$OUTPUT" "scope_class=" "Python output should contain scope_class="
log_pass "README.md basic quickstart works"
