#!/usr/bin/env bash
# Install every optional extra so all README.claim-extractor.md examples can run.
# Equivalent to `pip install orbitals[claim-extractor-all]` from the README, but uses uv.

set -euo pipefail
source "$(dirname "$0")/lib.sh"

require_uv

log "syncing project with all extras (this may download several GB)"
cd "$REPO_ROOT"
uv sync --all-extras

log_pass "all extras installed"
