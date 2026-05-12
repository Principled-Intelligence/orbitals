# README example tests — `scope-guard`

End-to-end bash scripts that exercise the code examples shown in
[`README.md`](../../../README.md) and
[`README.scope-guard.md`](../../../README.scope-guard.md).

These complement the unit tests under [`tests/`](../../../tests/), which only mock
the heavy paths. The scripts in this directory actually load real models, start
the real FastAPI/vLLM server, and run real HTTP requests, so they require:

- A CUDA-capable GPU (the README models are 4B parameters)
- ~10 GB of disk for the Hugging Face download cache
- The `vllm`, `transformers`, `accelerate`, `fastapi`, `uvicorn` extras installed
  (`./00-install-all.sh` will do this)

The hosted-API example from `README.scope-guard.md`
(`ScopeGuard(backend="api", api_key="principled_1234")` against the cloud
endpoint) is **not** scripted here, because as the README notes the public
hosted endpoint is "coming soon". Once it's available, drop a script following
the same pattern as `12-serve-api-client-sync.sh` but pointing at the cloud URL
and authenticating with `PRINCIPLED_API_KEY`.

## Layout

| Script | What it covers |
|---|---|
| `00-install-all.sh` | `uv sync --all-extras` |
| `01-readme-basic-quickstart.sh` | The default-backend example in `README.md` |
| `02-vllm-quickstart.sh` | The vLLM quickstart in `README.scope-guard.md` |
| `03-vllm-input-format-string.sh` | `validate("...string...", ...)` |
| `04-vllm-input-format-dict.sh` | `validate({"role": "user", ...}, ...)` |
| `05-vllm-input-format-multi-turn.sh` | `validate([{...}, {...}, ...], ...)` |
| `06-vllm-batch-validate-single-desc.sh` | `batch_validate(..., ai_service_description=...)` |
| `07-vllm-batch-validate-per-conv-descs.sh` | `batch_validate(..., ai_service_descriptions=[...])` |
| `08-vllm-structured-ai-service-description.sh` | `AIServiceDescription` structured object |
| `09-hf-quickstart.sh` | Hugging Face backend quickstart |
| `10-serve-curl-validate.sh` | `curl POST /orbitals/scope-guard/validate` |
| `11-serve-curl-batch-validate.sh` | `curl POST /orbitals/scope-guard/batch-validate` |
| `12-serve-api-client-sync.sh` | `ScopeGuard(backend="api", api_url=...)` |
| `13-serve-api-client-async.sh` | `AsyncScopeGuard(backend="api", api_url=...)` |
| `run-all.sh` | Runs every script above in sequence |
| `lib.sh` | Shared helpers (logging, server start/stop) |

## Usage

Run the full battery:

```bash
./run-all.sh
```

Or run a single scenario:

```bash
./02-vllm-quickstart.sh
```

The "served" scripts (10–13) start their own server by default. To share a
single long-lived server across them (which is faster, since the model only
loads once), start one and export its URL:

```bash
# Terminal 1
uv run orbitals scope-guard serve scope-guard --port 8000

# Terminal 2
export SCOPE_GUARD_SERVER_URL=http://localhost:8000
./10-serve-curl-validate.sh
./11-serve-curl-batch-validate.sh
./12-serve-api-client-sync.sh
./13-serve-api-client-async.sh
```

## Configuration

These environment variables are honored by every script:

| Variable | Default | Purpose |
|---|---|---|
| `SCOPE_GUARD_MODEL` | `scope-guard-q` | Model alias passed to `ScopeGuard` |
| `SCOPE_GUARD_SERVER_HOST` | `0.0.0.0` | Host for the spun-up server |
| `SCOPE_GUARD_SERVER_PORT` | `8000` | Port for the spun-up server |
| `SCOPE_GUARD_VLLM_PORT` | `8001` | Port for the inner vLLM server |
| `SCOPE_GUARD_SERVER_URL` | _unset_ | If set, served scripts reuse this URL |
| `SCOPE_GUARD_SERVER_STARTUP_TIMEOUT` | `600` | Seconds to wait for `serve` to become healthy |
