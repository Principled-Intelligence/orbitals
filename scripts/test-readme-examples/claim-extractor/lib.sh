#!/usr/bin/env bash
# Shared helpers for the claim-extractor README example tests.
# Source this file from each script:  source "$(dirname "$0")/lib.sh"

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib-common.sh
source "$LIB_DIR/../lib-common.sh"

# --------------------------------------------------------------------------
# Defaults (overridable via environment)
# --------------------------------------------------------------------------
# Note: ports default to 8100/8101 (not 8000/8001) so that a running scope-guard
# server does not collide with these tests.
: "${CLAIM_EXTRACTOR_MODEL:=claim-extractor}"
: "${CLAIM_EXTRACTOR_SERVER_HOST:=0.0.0.0}"
: "${CLAIM_EXTRACTOR_SERVER_PORT:=8100}"
: "${CLAIM_EXTRACTOR_VLLM_PORT:=8101}"
: "${CLAIM_EXTRACTOR_SERVER_STARTUP_TIMEOUT:=600}"

export CLAIM_EXTRACTOR_MODEL CLAIM_EXTRACTOR_SERVER_HOST CLAIM_EXTRACTOR_SERVER_PORT
export CLAIM_EXTRACTOR_VLLM_PORT CLAIM_EXTRACTOR_SERVER_STARTUP_TIMEOUT

# --------------------------------------------------------------------------
# Server helpers (for the serve-based tests)
# --------------------------------------------------------------------------
_SERVER_PID=""
_SERVER_LOG=""

server_url() {
    if [ -n "${CLAIM_EXTRACTOR_SERVER_URL:-}" ]; then
        printf '%s\n' "$CLAIM_EXTRACTOR_SERVER_URL"
    else
        printf 'http://%s:%s\n' "127.0.0.1" "$CLAIM_EXTRACTOR_SERVER_PORT"
    fi
}

start_server_if_needed() {
    # If the user already exported a URL, don't start anything.
    if [ -n "${CLAIM_EXTRACTOR_SERVER_URL:-}" ]; then
        log "reusing existing server at $CLAIM_EXTRACTOR_SERVER_URL"
        return 0
    fi

    _SERVER_LOG="$(mktemp -t orbitals-serve.XXXXXX.log)"
    log "starting server in background (log: $_SERVER_LOG)"
    (
        cd "$REPO_ROOT" && \
        uv run orbitals claim-extractor serve "$CLAIM_EXTRACTOR_MODEL" \
            --host "$CLAIM_EXTRACTOR_SERVER_HOST" \
            --port "$CLAIM_EXTRACTOR_SERVER_PORT" \
            --vllm-port "$CLAIM_EXTRACTOR_VLLM_PORT" \
            >"$_SERVER_LOG" 2>&1
    ) &
    _SERVER_PID=$!

    log "waiting up to ${CLAIM_EXTRACTOR_SERVER_STARTUP_TIMEOUT}s for /docs to be available..."
    local elapsed=0 health_url
    health_url="$(server_url)/docs"
    while (( elapsed < CLAIM_EXTRACTOR_SERVER_STARTUP_TIMEOUT )); do
        if ! kill -0 "$_SERVER_PID" 2>/dev/null; then
            log_fail "server process exited prematurely"
            tail -n 200 "$_SERVER_LOG" >&2 || true
            return 1
        fi
        if curl -fsS -o /dev/null "$health_url" 2>/dev/null; then
            log "server is up at $(server_url)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    log_fail "server did not become healthy within ${CLAIM_EXTRACTOR_SERVER_STARTUP_TIMEOUT}s"
    tail -n 200 "$_SERVER_LOG" >&2 || true
    stop_server || true
    return 1
}

stop_server() {
    if [ -z "$_SERVER_PID" ]; then
        return 0
    fi
    log "stopping server (pid=$_SERVER_PID)"
    kill "$_SERVER_PID" 2>/dev/null || true
    wait "$_SERVER_PID" 2>/dev/null || true
    _SERVER_PID=""
}

trap_stop_server() {
    # Install a trap to ensure the server is stopped on exit.
    trap 'stop_server' EXIT INT TERM
}
