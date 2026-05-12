#!/usr/bin/env bash
# Shared helpers for the scope-guard README example tests.
# Source this file from each script:  source "$(dirname "$0")/lib.sh"

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib-common.sh
source "$LIB_DIR/../lib-common.sh"

# --------------------------------------------------------------------------
# Defaults (overridable via environment)
# --------------------------------------------------------------------------
: "${SCOPE_GUARD_MODEL:=scope-guard-q}"
: "${SCOPE_GUARD_SERVER_HOST:=0.0.0.0}"
: "${SCOPE_GUARD_SERVER_PORT:=8000}"
: "${SCOPE_GUARD_VLLM_PORT:=8001}"
: "${SCOPE_GUARD_SERVER_STARTUP_TIMEOUT:=600}"

export SCOPE_GUARD_MODEL SCOPE_GUARD_SERVER_HOST SCOPE_GUARD_SERVER_PORT
export SCOPE_GUARD_VLLM_PORT SCOPE_GUARD_SERVER_STARTUP_TIMEOUT

# --------------------------------------------------------------------------
# Server helpers (for the serve-based tests)
# --------------------------------------------------------------------------
_SERVER_PID=""
_SERVER_LOG=""

server_url() {
    if [ -n "${SCOPE_GUARD_SERVER_URL:-}" ]; then
        printf '%s\n' "$SCOPE_GUARD_SERVER_URL"
    else
        printf 'http://%s:%s\n' "127.0.0.1" "$SCOPE_GUARD_SERVER_PORT"
    fi
}

start_server_if_needed() {
    # If the user already exported a URL, don't start anything.
    if [ -n "${SCOPE_GUARD_SERVER_URL:-}" ]; then
        log "reusing existing server at $SCOPE_GUARD_SERVER_URL"
        return 0
    fi

    _SERVER_LOG="$(mktemp -t orbitals-serve.XXXXXX.log)"
    log "starting server in background (log: $_SERVER_LOG)"
    (
        cd "$REPO_ROOT" && \
        uv run orbitals scope-guard serve "$SCOPE_GUARD_MODEL" \
            --host "$SCOPE_GUARD_SERVER_HOST" \
            --port "$SCOPE_GUARD_SERVER_PORT" \
            --vllm-port "$SCOPE_GUARD_VLLM_PORT" \
            >"$_SERVER_LOG" 2>&1
    ) &
    _SERVER_PID=$!

    log "waiting up to ${SCOPE_GUARD_SERVER_STARTUP_TIMEOUT}s for /docs to be available..."
    local elapsed=0 health_url
    health_url="$(server_url)/docs"
    while (( elapsed < SCOPE_GUARD_SERVER_STARTUP_TIMEOUT )); do
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

    log_fail "server did not become healthy within ${SCOPE_GUARD_SERVER_STARTUP_TIMEOUT}s"
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
