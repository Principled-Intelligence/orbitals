#!/usr/bin/env bash
# Run every claim-extractor README example test in sequence.
#
# Tips:
#   - Set SKIP_INSTALL=1 to skip 00-install-all.sh
#   - Set SKIP_HF=1 to skip the (slow) hf backend example (02)
#   - Set SKIP_VLLM=1 to skip the vllm backend examples (01, 03-08)
#   - Set SKIP_SERVE=1 to skip the served examples (09-13)
#
# When the served examples are run via this driver, the server is started
# *once* and reused across all of them (much faster than restarting per script).

set -uo pipefail
source "$(dirname "$0")/lib.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a PASSED=() FAILED=() SKIPPED=()

run_one() {
    local script="$1"
    local label
    label="$(basename "$script")"
    log "================ $label ================"
    if bash "$script"; then
        PASSED+=("$label")
    else
        FAILED+=("$label")
    fi
}

skip_one() {
    local label="$1" reason="$2"
    SKIPPED+=("$label  ($reason)")
    log_warn "skipping $label: $reason"
}

# 00 - install
if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
    skip_one "00-install-all.sh" "SKIP_INSTALL=1"
else
    run_one "$SCRIPT_DIR/00-install-all.sh"
fi

# 01 - vllm quickstart
if [[ "${SKIP_VLLM:-0}" == "1" ]]; then
    skip_one "01-vllm-quickstart.sh" "SKIP_VLLM=1"
else
    run_one "$SCRIPT_DIR/01-vllm-quickstart.sh"
fi

# 02 - hf quickstart
if [[ "${SKIP_HF:-0}" == "1" ]]; then
    skip_one "02-hf-quickstart.sh" "SKIP_HF=1"
else
    run_one "$SCRIPT_DIR/02-hf-quickstart.sh"
fi

# 03-08 - vllm backend variations
if [[ "${SKIP_VLLM:-0}" == "1" ]]; then
    for script in "$SCRIPT_DIR"/0[3-8]-vllm-*.sh; do
        skip_one "$(basename "$script")" "SKIP_VLLM=1"
    done
else
    for script in "$SCRIPT_DIR"/0[3-8]-vllm-*.sh; do
        run_one "$script"
    done
fi

# 09-13 - served examples (share one server)
if [[ "${SKIP_SERVE:-0}" == "1" ]]; then
    for script in "$SCRIPT_DIR"/{09,1[0-3]}-serve-*.sh; do
        [ -f "$script" ] || continue
        skip_one "$(basename "$script")" "SKIP_SERVE=1"
    done
else
    log "================ shared server for served scripts ================"
    trap_stop_server
    if start_server_if_needed; then
        export CLAIM_EXTRACTOR_SERVER_URL
        CLAIM_EXTRACTOR_SERVER_URL="$(server_url)"
        for script in "$SCRIPT_DIR"/{09,1[0-3]}-serve-*.sh; do
            [ -f "$script" ] || continue
            run_one "$script"
        done
        stop_server || true
        unset CLAIM_EXTRACTOR_SERVER_URL
    else
        for script in "$SCRIPT_DIR"/{09,1[0-3]}-serve-*.sh; do
            [ -f "$script" ] || continue
            FAILED+=("$(basename "$script")  (server failed to start)")
        done
    fi
fi

# Summary
echo
log "============== SUMMARY =============="
log_pass "passed:  ${#PASSED[@]}"
for s in "${PASSED[@]}"; do echo "    + $s"; done
log_warn "skipped: ${#SKIPPED[@]}"
for s in "${SKIPPED[@]}"; do echo "    - $s"; done
if (( ${#FAILED[@]} > 0 )); then
    log_fail "failed:  ${#FAILED[@]}"
    for s in "${FAILED[@]}"; do echo "    x $s"; done
    exit 1
fi
log_pass "all claim-extractor README examples passed"
