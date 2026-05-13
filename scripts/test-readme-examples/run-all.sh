#!/usr/bin/env bash
# Top-level driver: runs every README-example suite in sequence.
#
# Tips:
#   - Set SKIP_SCOPE_GUARD=1 to skip the scope-guard suite
#   - Set SKIP_CLAIM_EXTRACTOR=1 to skip the claim-extractor suite
#
# All other SKIP_* variables (SKIP_HF, SKIP_VLLM, SKIP_SERVE, SKIP_INSTALL)
# are passed through to each suite's run-all.sh.

set -uo pipefail
source "$(dirname "$0")/lib-common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a SUITE_PASSED=() SUITE_FAILED=() SUITE_SKIPPED=()

run_suite() {
    local label="$1" script="$2"
    log "================ suite: $label ================"
    if bash "$script"; then
        SUITE_PASSED+=("$label")
    else
        SUITE_FAILED+=("$label")
    fi
}

skip_suite() {
    local label="$1" reason="$2"
    SUITE_SKIPPED+=("$label  ($reason)")
    log_warn "skipping suite $label: $reason"
}

# scope-guard
if [[ "${SKIP_SCOPE_GUARD:-0}" == "1" ]]; then
    skip_suite "scope-guard" "SKIP_SCOPE_GUARD=1"
else
    run_suite "scope-guard" "$SCRIPT_DIR/scope-guard/run-all.sh"
fi

# claim-extractor
if [[ "${SKIP_CLAIM_EXTRACTOR:-0}" == "1" ]]; then
    skip_suite "claim-extractor" "SKIP_CLAIM_EXTRACTOR=1"
else
    run_suite "claim-extractor" "$SCRIPT_DIR/claim-extractor/run-all.sh"
fi

# Summary
echo
log "============== OVERALL SUMMARY =============="
log_pass "suites passed:  ${#SUITE_PASSED[@]}"
for s in "${SUITE_PASSED[@]}"; do echo "    + $s"; done
log_warn "suites skipped: ${#SUITE_SKIPPED[@]}"
for s in "${SUITE_SKIPPED[@]}"; do echo "    - $s"; done
if (( ${#SUITE_FAILED[@]} > 0 )); then
    log_fail "suites failed: ${#SUITE_FAILED[@]}"
    for s in "${SUITE_FAILED[@]}"; do echo "    x $s"; done
    exit 1
fi
log_pass "all README example suites passed"
