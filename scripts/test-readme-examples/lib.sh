#!/usr/bin/env bash
# Shared helpers for README example tests.
# Source this file from each script:  source "$(dirname "$0")/lib.sh"

# --------------------------------------------------------------------------
# Resolve repository root (two levels up from this file: scripts/test-readme-examples/lib.sh)
# --------------------------------------------------------------------------
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LIB_DIR/../.." && pwd)"

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
# Logging
# --------------------------------------------------------------------------
if [ -t 1 ]; then
    _GREEN=$'\033[0;32m'
    _RED=$'\033[0;31m'
    _YELLOW=$'\033[0;33m'
    _BLUE=$'\033[0;34m'
    _RESET=$'\033[0m'
else
    _GREEN= _RED= _YELLOW= _BLUE= _RESET=
fi

log()      { printf '%s[%s]%s %s\n' "$_BLUE"   "$(date +%H:%M:%S)" "$_RESET" "$*"; }
log_pass() { printf '%s[PASS]%s %s\n'                "$_GREEN"  "$_RESET" "$*"; }
log_fail() { printf '%s[FAIL]%s %s\n'                "$_RED"    "$_RESET" "$*" >&2; }
log_warn() { printf '%s[WARN]%s %s\n'                "$_YELLOW" "$_RESET" "$*" >&2; }

# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------
assert_contains() {
    # assert_contains <haystack> <needle> [<message>]
    local haystack="$1" needle="$2" message="${3:-output should contain '$2'}"
    if [[ "$haystack" != *"$needle"* ]]; then
        log_fail "$message"
        printf '  --- output ---\n%s\n  --------------\n' "$haystack" >&2
        return 1
    fi
}

assert_status() {
    # assert_status <actual> <expected> [<message>]
    local actual="$1" expected="$2" message="${3:-expected status $2 but got $1}"
    if [[ "$actual" != "$expected" ]]; then
        log_fail "$message"
        return 1
    fi
}

# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
require_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        log_fail "uv is required but not installed (https://docs.astral.sh/uv/)"
        return 1
    fi
}

require_curl() {
    if ! command -v curl >/dev/null 2>&1; then
        log_fail "curl is required but not installed"
        return 1
    fi
}

require_extras() {
    # require_extras <extra-name> [<extra-name> ...]
    # Hint to the user that they need ./00-install-all.sh or specific extras.
    local missing=0
    for extra in "$@"; do
        if ! (cd "$REPO_ROOT" && uv run python -c "
import importlib, sys
for mod in {
    'scope-guard-vllm': ['vllm', 'transformers'],
    'scope-guard-hf':   ['transformers', 'accelerate'],
    'scope-guard-serve':['fastapi', 'uvicorn', 'vllm', 'transformers'],
    'serving':          ['fastapi', 'uvicorn'],
}.get('$extra', []):
    try:
        importlib.import_module(mod)
    except ImportError:
        sys.exit(f'missing module: {mod}')
" 2>/dev/null); then
            log_warn "extra '$extra' not installed (run ./00-install-all.sh, or 'uv sync --extra $extra')"
            missing=1
        fi
    done
    return $missing
}

require_gpu() {
    if ! (cd "$REPO_ROOT" && uv run python -c "
import sys
try:
    import torch
except ImportError:
    sys.exit('torch not installed')
sys.exit(0 if torch.cuda.is_available() else 'no CUDA-capable GPU detected')
" 2>/dev/null); then
        log_warn "no usable GPU detected; this script is likely to fail or be very slow"
        return 1
    fi
}

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
