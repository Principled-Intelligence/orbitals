# README example tests

End-to-end bash scripts that exercise the code examples shown in the various
README files of this repo. The suite is split per-component:

| Suite | README it covers | Subdir |
|---|---|---|
| `scope-guard` | [`README.md`](../../README.md), [`README.scope-guard.md`](../../README.scope-guard.md) | [`scope-guard/`](./scope-guard/) |
| `claim-extractor` | [`README.claim-extractor.md`](../../README.claim-extractor.md) | [`claim-extractor/`](./claim-extractor/) |

These complement the unit tests under [`tests/`](../../tests/), which only mock
the heavy paths. The scripts here actually load real models, start the real
FastAPI/vLLM server, and run real HTTP requests, so they require:

- A CUDA-capable GPU (the README models are 4B parameters)
- ~10 GB of disk for the Hugging Face download cache
- The `vllm`, `transformers`, `accelerate`, `fastapi`, `uvicorn` extras installed
  (each suite's `00-install-all.sh` will do this)

## Layout

```
scripts/test-readme-examples/
├── README.md           # (you are here)
├── lib-common.sh       # domain-agnostic helpers (logging, asserts, require_*)
├── run-all.sh          # runs both suites in sequence
├── scope-guard/        # tests for README.md + README.scope-guard.md
└── claim-extractor/    # tests for README.claim-extractor.md
```

Each suite is self-contained: it has its own `lib.sh` (env-vars + server
helpers), its own `run-all.sh`, and its own numbered scripts.

## Usage

Run everything:

```bash
./run-all.sh
```

Run a single suite:

```bash
./scope-guard/run-all.sh
./claim-extractor/run-all.sh
```

Skip a suite from the top-level driver:

```bash
SKIP_SCOPE_GUARD=1     ./run-all.sh
SKIP_CLAIM_EXTRACTOR=1 ./run-all.sh
```

Run a single scenario:

```bash
./scope-guard/02-vllm-quickstart.sh
./claim-extractor/01-vllm-quickstart.sh
```

See each suite's `README.md` for the full per-suite script catalog and the
environment variables it honors.
