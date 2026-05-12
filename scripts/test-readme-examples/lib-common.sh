#!/usr/bin/env bash
# Shared, domain-agnostic helpers for the README example tests.
#
# Sourced by the per-domain lib.sh files in scope-guard/ and claim-extractor/.
# Provides:
#   - REPO_ROOT
#   - log / log_pass / log_fail / log_warn
#   - assert_contains / assert_status
#   - require_uv / require_curl / require_extras / require_gpu
#
# Server helpers and env-var defaults live in each domain's lib.sh.

# --------------------------------------------------------------------------
# Resolve repository root (two levels up from this file)
# --------------------------------------------------------------------------
LIB_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LIB_COMMON_DIR/../.." && pwd)"

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
    'scope-guard-vllm':     ['vllm', 'transformers'],
    'scope-guard-hf':       ['transformers', 'accelerate'],
    'scope-guard-serve':    ['fastapi', 'uvicorn', 'vllm', 'transformers'],
    'claim-extractor-vllm': ['vllm', 'transformers'],
    'claim-extractor-hf':   ['transformers', 'accelerate'],
    'claim-extractor-serve':['fastapi', 'uvicorn', 'vllm', 'transformers'],
    'serving':              ['fastapi', 'uvicorn'],
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

uv_run_python() {
    # Run a Python script via temp file instead of stdin (`python -`).
    # vLLM's multiprocessing breaks with `python -` because child processes
    # attempt to re-import __main__ from the non-existent path "<stdin>".
    #
    # The script is launched in its own session (via `setsid`) so we can
    # SIGKILL any subprocesses that outlive the main interpreter. vLLM v1
    # spawns an out-of-process EngineCore worker that is occasionally
    # reparented to init while still holding GPU memory; if we don't reap
    # it, the next test can't acquire the GPU.
    local _tmpscript _pid _rc=0
    _tmpscript="$(mktemp /tmp/orbitals-test.XXXXXX.py)"
    cat > "$_tmpscript"
    setsid uv run python "$_tmpscript" &
    _pid=$!
    wait "$_pid" || _rc=$?
    # Kill anything still alive in the session we just created. `$_pid` is
    # the session leader's PID because non-interactive bash backgrounds
    # without setpgid, so setsid execs in place (no fork).
    pkill -KILL -s "$_pid" 2>/dev/null || true
    rm -f "$_tmpscript"
    return $_rc
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
