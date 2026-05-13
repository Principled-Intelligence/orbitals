# README example tests — `claim-extractor`

End-to-end bash scripts that exercise the code examples shown in
[`README.claim-extractor.md`](../../../README.claim-extractor.md).

These complement the unit tests under [`tests/`](../../../tests/), which only
mock the heavy paths. The scripts in this directory actually load real models,
start the real FastAPI/vLLM server, and run real HTTP requests, so they require:

- A CUDA-capable GPU (the `claim-extractor-q` model is 4B parameters)
- ~10 GB of disk for the Hugging Face download cache
- The `vllm`, `transformers`, `accelerate`, `fastapi`, `uvicorn` extras installed
  (`./00-install-all.sh` will do this)

## Layout

| Script | What it covers |
|---|---|
| `00-install-all.sh` | `uv sync --all-extras` |
| `01-vllm-quickstart.sh` | The vLLM quickstart in `README.claim-extractor.md` |
| `02-hf-quickstart.sh` | Hugging Face backend (`backend="hf"`) |
| `03-vllm-input-format-string.sh` | `extract("...string...", ...)` — assistant message |
| `04-vllm-input-format-dict.sh` | `extract({"role": "user", ...}, ...)` |
| `05-vllm-input-format-conversation.sh` | `extract([{...}, {...}], ...)` |
| `06-vllm-structured-ai-service-description.sh` | `AIServiceDescription` structured object |
| `07-vllm-batch-extract-single-desc.sh` | `batch_extract(..., ai_service_description=...)` |
| `08-vllm-batch-extract-per-conv-descs.sh` | `batch_extract(..., ai_service_descriptions=[...])` |
| `09-serve-curl-extract.sh` | `curl POST /orbitals/claim-extractor/extract` |
| `10-serve-curl-batch-extract.sh` | `curl POST /orbitals/claim-extractor/batch-extract` |
| `11-serve-curl-extract-conversation.sh` | `curl POST /orbitals/claim-extractor/extract-conversation` |
| `12-serve-api-client-sync.sh` | `ClaimExtractor(backend="api", api_url=...)` |
| `13-serve-api-client-async.sh` | `AsyncClaimExtractor(backend="api", api_url=...)` |
| `run-all.sh` | Runs every script above in sequence |
| `lib.sh` | Suite-specific helpers (env vars, server start/stop); sources `../lib-common.sh` |

## Usage

Run the full battery:

```bash
./run-all.sh
```

Or run a single scenario:

```bash
./01-vllm-quickstart.sh
```

The "served" scripts (09–13) start their own server by default. To share a
single long-lived server across them (which is faster, since the model only
loads once), start one and export its URL:

```bash
# Terminal 1
uv run orbitals claim-extractor serve --port 8100

# Terminal 2
export CLAIM_EXTRACTOR_SERVER_URL=http://localhost:8100
./09-serve-curl-extract.sh
./10-serve-curl-batch-extract.sh
./11-serve-curl-extract-conversation.sh
./12-serve-api-client-sync.sh
./13-serve-api-client-async.sh
```

## Configuration

These environment variables are honored by every script:

| Variable | Default | Purpose |
|---|---|---|
| `CLAIM_EXTRACTOR_MODEL` | `claim-extractor` | Model alias passed to `ClaimExtractor` |
| `CLAIM_EXTRACTOR_SERVER_HOST` | `0.0.0.0` | Host for the spun-up server |
| `CLAIM_EXTRACTOR_SERVER_PORT` | `8100` | Port for the spun-up server (non-8000 to avoid colliding with scope-guard) |
| `CLAIM_EXTRACTOR_VLLM_PORT` | `8101` | Port for the inner vLLM server |
| `CLAIM_EXTRACTOR_SERVER_URL` | _unset_ | If set, served scripts reuse this URL |
| `CLAIM_EXTRACTOR_SERVER_STARTUP_TIMEOUT` | `600` | Seconds to wait for `serve` to become healthy |
