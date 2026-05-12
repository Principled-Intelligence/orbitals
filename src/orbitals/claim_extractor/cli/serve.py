import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx
import typer
import uvicorn

from orbitals.claim_extractor import ClaimExtractor

app = typer.Typer()


def setup_fastapi_logging():
    """Configure logging for the FastAPI server."""
    logging.basicConfig(
        level=logging.INFO,
        format="[FastAPI] %(levelname)s %(asctime)s %(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Set log level for common noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.command("serve")
def serve(
    vllm_model: str = typer.Argument(
        "claim-extractor", help="The model used for vLLM serving"
    ),
    skip_evidences: bool = typer.Option(True, help="Whether to skip evidences"),
    port: int = typer.Option(
        8000, "-p", "--port", help="The port to use for the server"
    ),
    host: str = typer.Option(
        "0.0.0.0", "-h", "--host", help="The host to use for the server"
    ),
    vllm_port: int = typer.Option(8001, help="The port to use for the vLLM server"),
    vllm_max_model_len: int = typer.Option(
        40_000, help="Maximum model length for vLLM"
    ),
    vllm_max_num_seqs: int = typer.Option(
        2, help="Maximum number of sequences for vLLM"
    ),
    vllm_gpu_memory_utilization: float = typer.Option(
        0.9, help="GPU memory utilization for vLLM"
    ),
    vllm_enable_prefix_caching: bool = typer.Option(
        True, help="Enable vLLM prefix caching"
    ),
    vllm_language_model_only: bool = typer.Option(
        True,
        help="Load only the language-model component of the underlying model (skips multimodal modules)",
    ),
    vllm_speculative_config: str = typer.Option(
        '{"num_speculative_tokens": 4, "method": "mtp"}',
        help="Speculative decoding config for vLLM as a JSON string. Pass an empty string to disable.",
    ),
    vllm_extra_args: str | None = typer.Option(
        None, help="Extra arguments to pass to the vLLM server"
    ),
    temperature: float = typer.Option(
        0.7, help="Sampling temperature for vLLM (0.0 = greedy)"
    ),
    frequency_penalty: float = typer.Option(
        0.0, help="frequency_penalty for vLLM sampling (additive logit penalty)"
    ),
    presence_penalty: float = typer.Option(
        1.5, help="presence_penalty for vLLM sampling (additive logit penalty)"
    ),
    repetition_penalty: float = typer.Option(
        1.0,
        help="repetition_penalty for vLLM sampling (multiplicative; 1.0 = off)",
    ),
    top_p: float = typer.Option(
        0.8, help="top_p (nucleus) sampling for vLLM (1.0 = off)"
    ),
    top_k: int = typer.Option(
        20, help="top_k sampling for vLLM (-1 = off)"
    ),
    min_p: float = typer.Option(
        0.0, help="min_p sampling for vLLM (0.0 = off)"
    ),
):
    vllm_model = ClaimExtractor.maybe_map_model(vllm_model)

    os.environ["CLAIM_EXTRACTOR_VLLM_MODEL"] = vllm_model
    os.environ["CLAIM_EXTRACTOR_VLLM_SERVING_URL"] = f"http://localhost:{vllm_port}"
    os.environ["CLAIM_EXTRACTOR_SKIP_EVIDENCES"] = str(1) if skip_evidences else str(0)
    os.environ["CLAIM_EXTRACTOR_TEMPERATURE"] = str(temperature)
    os.environ["CLAIM_EXTRACTOR_FREQUENCY_PENALTY"] = str(frequency_penalty)
    os.environ["CLAIM_EXTRACTOR_PRESENCE_PENALTY"] = str(presence_penalty)
    os.environ["CLAIM_EXTRACTOR_REPETITION_PENALTY"] = str(repetition_penalty)
    os.environ["CLAIM_EXTRACTOR_TOP_P"] = str(top_p)
    os.environ["CLAIM_EXTRACTOR_TOP_K"] = str(top_k)
    os.environ["CLAIM_EXTRACTOR_MIN_P"] = str(min_p)

    # Set up vLLM logging configuration
    vllm_logging_config = (
        Path(__file__).parent.parent / "serving" / "vllm_logging_config.json"
    )
    os.environ["VLLM_LOGGING_CONFIG_PATH"] = str(vllm_logging_config)

    # Start vLLM server
    vllm_cmd = [
        "vllm",
        "serve",
        vllm_model,
        "--max-model-len",
        str(vllm_max_model_len),
        "--max-num-seqs",
        str(vllm_max_num_seqs),
        "--port",
        str(vllm_port),
        "--gpu-memory-utilization",
        str(vllm_gpu_memory_utilization),
    ]
    if vllm_enable_prefix_caching:
        vllm_cmd.append("--enable-prefix-caching")
    if vllm_language_model_only:
        vllm_cmd.append("--language-model-only")
    if vllm_speculative_config:
        vllm_cmd.extend(["--speculative_config", vllm_speculative_config])
    if vllm_extra_args is not None:
        vllm_cmd.extend(shlex.split(vllm_extra_args))

    typer.echo(f"Starting vLLM server: {' '.join(vllm_cmd)}")
    vllm_process = subprocess.Popen(vllm_cmd, stdout=sys.stdout, stderr=sys.stderr)

    # Wait for vLLM server to be ready
    typer.echo(
        f"Waiting for vLLM server to be ready at http://localhost:{vllm_port}..."
    )
    max_retries = 60
    retry_delay = 5

    for i in range(max_retries):
        try:
            response = httpx.get(f"http://localhost:{vllm_port}/health", timeout=2.0)
            if response.status_code == 200:
                typer.echo("vLLM server is ready!")
                break
        except (httpx.RequestError, httpx.HTTPError):
            if i < max_retries - 1:
                typer.echo(f"vLLM server not ready yet, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                typer.echo("vLLM server failed to start in time", err=True)
                vllm_process.terminate()
                vllm_process.wait()
                raise typer.Exit(code=1)

    # Set up FastAPI logging
    setup_fastapi_logging()

    # Start uvicorn server with custom logging configuration
    typer.echo(f"Starting FastAPI server on {host}:{port}")

    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG  # type: ignore[attr-defined]
    log_config["formatters"]["default"]["fmt"] = (
        "[FastAPI] %(levelprefix)s %(asctime)s %(name)s] %(message)s"
    )
    log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config["formatters"]["access"]["fmt"] = (
        "[FastAPI] %(levelprefix)s %(asctime)s %(client_addr)s - "
        '"%(request_line)s" %(status_code)s'
    )
    log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

    try:
        uvicorn.run(
            "orbitals.claim_extractor.serving.main:app",
            host=host,
            port=port,
            log_config=log_config,
            log_level="info",
        )
    finally:
        # Clean up vLLM process on exit
        typer.echo("Shutting down vLLM server...")
        vllm_process.terminate()
        vllm_process.wait(timeout=10)
