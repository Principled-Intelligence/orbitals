<div align="left">
    <h1>
        <strong>ClaimExtractor</strong>
    </h1>
</div>

Given a conversation (or a single message) and — optionally — the specifications of an AI assistant, `claim-extractor` extracts the **claims** and **intents** expressed in the last message of the conversation.

* **Claims** — self-contained, decontextualized factual statements, labeled with one of four subtypes:
  * **Factoid** — verifiable facts about the world (dates, prices, procedures, URLs, …)
  * **Capability** — what the AI service can or cannot do (scope, features, limitations)
  * **User Assertion** — facts the user states about themselves (e.g., *"My ID expired"*)
  * **Unverifiable** — common knowledge, subjective claims, marketing language, visual/UI references
* **Intents** — explicit goals, requests, or actions the user wants to accomplish (extracted from user messages only)

`claim-extractor` is a building block for a wide range of downstream governance tasks: fact-checking, intent routing, response auditing, and hallucination detection. It is powered by open-weight Qwen-family models fine-tuned to extract atomic, self-contained claims and intents, released in two sizes:

| Model | Parameters | Hosting |
| :--- | :--- | :--- |
| [`claim-extractor-4B-q`](https://huggingface.co/principled-intelligence/claim-extractor-4B-q-2605) | 4B | Self-hosted (vLLM / HuggingFace) |
| [`claim-extractor-2B-q`](https://huggingface.co/principled-intelligence/claim-extractor-2B-q-2605) | 2B | Self-hosted (vLLM / HuggingFace) |

## Quickstart

The easiest way to get started with ClaimExtractor is to use our open models, which you can self-host on consumer-grade GPUs and use via the `vllm` or `hf` backends.

First, install `orbitals` and `claim-extractor`:

```bash
uv add 'orbitals[claim-extractor-vllm]'
# or if you prefer to use HuggingFace pipelines for inference instead of vLLM
# uv add 'orbitals[claim-extractor-hf]'
```

Then:

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    backend="vllm",  # or "hf" for huggingface
    model="claim-extractor-4B-q",    # for the 4B Qwen-family model
    # model="claim-extractor-2B-q",  # for the 2B Qwen-family model
)

ai_service_description = """
You are a virtual assistant for a parcel delivery service.
You can only answer questions about package tracking.
"""

assistant_message = (
    "Your package with tracking number 1234567890 is currently in transit and "
    "is expected to be delivered on December 12, 2025. If you want, I can also "
    "notify you when it is out for delivery."
)

result = ce.extract(
    assistant_message,
    ai_service_description=ai_service_description,
)

print("--Claims--")
for claim in result.extractions.claims:
    print(f"[{claim.subtype}] {claim.content}")

print("--Intents--")
for intent in result.extractions.intents:
    print(f"[Intent] {intent.content}")
print("--End--")

# --Claims--
# [Factoid] The tracking number of the user's package is 1234567890
# [Factoid] The user's package is expected to be delivered on December 12, 2025
# [Capability] The parcel delivery virtual assistant can notify the user when the package is out for delivery
# --Intents--
# --End--
```

## Usage

### Initialization

Initialize the `ClaimExtractor` object by picking a backend — `vllm` (recommended, faster) or `hf`:

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    backend="vllm",                   # or "hf"
    model="claim-extractor-4B-q",     # for the 4B Qwen-family model
    # model="claim-extractor-2B-q",   # for the 2B Qwen-family model
)
```

If you've already started a `claim-extractor` server (see [Serving](#serving-claimextractor-on-premise-or-on-your-infrastructure) below), use the `api` backend instead and point it at your server's URL:

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    backend="api",
    api_url="http://localhost:8000",
)
```

The `vllm` and `hf` backends accept sampling controls (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`) with defaults tuned for claim extraction; override them on the constructor if you need to. The matching server-side defaults are exposed via `orbitals claim-extractor serve --help`. The `api` backend does not take sampling controls — it forwards requests to a running server, which uses whatever sampling it was started with.

The `vllm` backend additionally enables the following vLLM engine flags by default — `orbitals claim-extractor serve` applies the same defaults automatically:

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `enable_prefix_caching` | `True` | Reuses KV-cache across requests that share a prompt prefix (the long system prompt is identical across every extraction). |
| `language_model_only` | `True` | Disables all multimodal inputs (sets per-prompt limits to 0 for every modality). |
| `speculative_config` | `{"num_speculative_tokens": 4, "method": "mtp"}` | Enables MTP speculative decoding for higher throughput. |

Override them via the constructor (`ClaimExtractor(backend="vllm", enable_prefix_caching=False)`) or, on the CLI, via `--no-vllm-enable-prefix-caching` / `--no-vllm-language-model-only` to disable the boolean flags, and `--vllm-speculative-config '<json>'` to change the speculative config (pass `--vllm-speculative-config ""` to disable it). From Python, pass `speculative_config=None` to disable speculative decoding entirely.

### Extractions

`ce.extract(...)` returns a `ClaimExtractorOutput` whose `extractions` field contains the extracted `claims` and `intents`:

```python
from orbitals.claim_extractor import ClaimExtractor, Claim, Intent

ce = ClaimExtractor(backend="vllm", model="claim-extractor-4B-q")
message = "Your package is in transit and will arrive on December 12, 2025."
ai_service_description = "You are a virtual assistant for a parcel delivery service."

result = ce.extract(message, ai_service_description=ai_service_description)

assert len(result.extractions.claims) == 0 or isinstance(result.extractions.claims[0], Claim)
assert len(result.extractions.intents) == 0 or isinstance(result.extractions.intents[0], Intent)

for claim in result.extractions.claims:
    # claim.subtype is one of: "Factoid", "Capability", "User Assertion", "Unverifiable"
    print(claim.subtype, claim.content)

for intent in result.extractions.intents:
    print(intent.content)
```

Each `Claim` and `Intent` carries a decontextualized `content` string. An `evidences` field is also present on every extraction (see [Evidences](#evidences) below), but the currently released model does not populate it — it will always be an empty list.

### Intents only

If you only care about the user's **intents** and don't need claims, set `intents_only=True`. This applies a stopping criterion that halts generation as soon as the intents are produced — the model never generates the (often much larger) claims section, so extraction is faster and cheaper. The returned `claims` list is always empty in this mode.

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(backend="vllm", model="claim-extractor-4B-q", intents_only=True)

result = ce.extract(
    {"role": "user", "content": "Can you book me an appointment for next Tuesday morning?"},
    ai_service_description=ai_service_description,
)

for intent in result.extractions.intents:
    print(intent.content)

assert result.extractions.claims == []
```

`intents_only` is supported on every backend (`vllm`, `hf`, `api`) and is available both as a constructor default and as a per-call override on `extract` / `batch_extract`:

```python
ce = ClaimExtractor(backend="vllm", model="claim-extractor-4B-q")  # default: full extraction

# override per call — extract only intents for this request
result = ce.extract(message, ai_service_description=ai_service_description, intents_only=True)
```

When serving, set the server-side default with the `--intents-only` flag on `orbitals claim-extractor serve`, or pass `"intents_only": true` on individual requests (see [Serving](#serving-claimextractor-on-premise-or-on-your-infrastructure) below).

### Claim subtypes

```python
# claim.subtype is a string literal. Valid values:
#   "Factoid"         — verifiable facts about the world
#   "Capability"      — what the AI service can or cannot do
#   "User Assertion"  — facts the user states about themselves
#   "Unverifiable"    — common knowledge / subjective / marketing / UI-references

if claim.subtype == "Capability":
    print("The assistant said it can do:", claim.content)
```

### Input Formats

The `extract` method is flexible and accepts various input formats.

#### Single message as a string

A raw string is interpreted as an **assistant** message — this is the most common case for fact-checking an AI response:

```python
result = ce.extract(
    "Your package is in transit and will arrive on December 12, 2025.",
    ai_service_description=ai_service_description,
)
```

#### Single message as a dictionary (OpenAI's API Message)

```python
result = ce.extract(
    {
        "role": "user",
        "content": "Can I still cancel order #42 before it ships?",
    },
    ai_service_description=ai_service_description,
)
```

#### Conversation as a list of dictionaries

```python
result = ce.extract(
    [
        {
            "role": "user",
            "content": "I ordered a package, tracking number 1234567890.",
        },
        {
            "role": "assistant",
            "content": "Your package is in transit and will arrive on December 12, 2025.",
        },
    ],
    ai_service_description=ai_service_description,
)
```

Claims and intents are extracted **only from the last message** of the conversation; earlier turns are used as context for decontextualization. The last message can be either a user or an assistant message:

- If the last message is from the **user**, both claims and intents may be extracted.
- If the last message is from the **assistant**, only claims may be extracted (assistant messages never produce intents).

If you need claims/intents extracted from every turn of a conversation, use the `/extract-conversation` endpoint (see [Extracting from every turn of a conversation](#extracting-from-every-turn-of-a-conversation) below).

### AI Service Description

The AI service description is **optional** for ClaimExtractor — you can pass `None` (the default) when you don't have one. However, providing a description meaningfully improves extraction quality:

1. It lets the model name the service explicitly in `Capability` claims.
2. It adds domain-specific specificity to every extraction.

You can provide it in two ways:

1. **As a single string** — a free-form description.
2. **As a structured object** — an `orbitals.types.AIServiceDescription` (**strongly recommended approach** for best performance):

```python
from orbitals.types import AIServiceDescription

message = "Your package is in transit and will arrive on December 12, 2025."
ai_service_description = AIServiceDescription(
    identity_role="Virtual assistant for a parcel delivery service.",
    context="Operates on the company website; users are customers checking on shipments.",
    knowledge_scope="Package tracking, delivery windows, redelivery options.",
    functionalities=["Tracking lookup", "Out-for-delivery notifications"],
    principles="Only answer questions related to package tracking.",
    website_url="https://example.com",
)

result = ce.extract(message, ai_service_description=ai_service_description)
```

Only `identity_role` and `context` are required; every other field is optional.

### Evidences

In addition to a decontextualized `content` string, every `Claim` and `Intent` is designed to carry a list of `evidences` — verbatim excerpts from the source message that support the extraction.

> **Note**: the currently released `claim-extractor-4B-q` and `claim-extractor-2B-q` models do **not** extract evidences — every `evidences` list will be empty. Evidence extraction is on the roadmap and will ship with a future model release. The `skip_evidences` flag on `ClaimExtractor` and the matching `--skip-evidences` serve flag are reserved for that future release; today they have no effect.

### Batch Processing

You can extract from multiple conversations at once using `batch_extract`.

#### Single AI Service Description

```python
messages = [
    "Your package is in transit and will arrive on December 12, 2025.",
    "Order #42 has already shipped and cannot be cancelled anymore.",
]

results = ce.batch_extract(
    messages,
    ai_service_description=ai_service_description,
)
```

#### Multiple AI Service Descriptions

```python
ai_service_descriptions = [
    "You are a virtual assistant for Postal Service. You only answer questions about package tracking.",
    "You are a virtual assistant for an e-commerce platform. You help users manage their orders.",
]

results = ce.batch_extract(
    messages,
    ai_service_descriptions=ai_service_descriptions,
)
```

## Serving ClaimExtractor on-premise or on your infrastructure

`claim-extractor` comes with built-in support for serving. For better performance, it consists of two components:

1. A **vLLM serving engine** that runs the model.
2. A **FastAPI server** that provides the end-to-end API interface, mapping input data to prompts, invoking the vLLM serving engine and returning the response to the user.

All of this is configured via the `orbitals claim-extractor serve` command:

```bash
# install the necessary packages
uv add 'orbitals[claim-extractor-serve]'

# start everything (defaults to "claim-extractor", which aliases to claim-extractor-4B-q)
orbitals claim-extractor serve --port 8000

# or pin a specific model
orbitals claim-extractor serve claim-extractor-4B-q --port 8000
# or use the smaller 2B model
# orbitals claim-extractor serve claim-extractor-2B-q --port 8000

# extract only intents by default (stops generation before claims)
# orbitals claim-extractor serve --intents-only --port 8000
```

`--intents-only` sets the server-side default; individual requests can still override it by passing `"intents_only": true` (or `false`) in the request body. See [Intents only](#intents-only).

Once the server is running, you can interact with it as follows:

#### 1. Direct HTTP Requests

Send requests to the `/orbitals/claim-extractor/extract` (or `/orbitals/claim-extractor/batch-extract`) endpoint using cURL or any HTTP client:

```bash
curl -X 'POST' \
    'http://localhost:8000/orbitals/claim-extractor/extract' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "conversation": "Your package is in transit and will arrive on December 12, 2025.",
        "ai_service_description": "You are a virtual assistant for a parcel delivery service. You can only answer questions about package tracking."
    }'
```

Response:

```json
{
    "extractions": {
        "intents": [],
        "claims": [
            {
                "subtype": "Factoid",
                "content": "The user's package is currently in transit",
                "evidences": []
            },
            {
                "subtype": "Factoid",
                "content": "The user's package is expected to arrive on December 12, 2025",
                "evidences": []
            }
        ]
    },
    "time_taken": 0.47,
    "model": "<model>",
    "usage": { "prompt_tokens": 1234, "completion_tokens": 120, "total_tokens": 1354 }
}
```

##### Extracting from every turn of a conversation

If you want claims/intents extracted from *every* message in a conversation (not just the last), use the `/extract-conversation` endpoint:

```bash
curl -X 'POST' \
    'http://localhost:8000/orbitals/claim-extractor/extract-conversation' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "conversation": [
            {"role": "user", "content": "I ordered a package, tracking number 1234567890."},
            {"role": "assistant", "content": "Your package is in transit and will arrive on December 12, 2025."}
        ],
        "ai_service_description": "You are a virtual assistant for a parcel delivery service."
    }'
```

Response:

```json
{
    "extractions": [
        { "intents": [/* ... */], "claims": [/* ... */] },
        { "intents": [],          "claims": [/* ... */] }
    ],
    "time_taken": 0.91,
    "model": "<model>",
    "usage": { "prompt_tokens": 2468, "completion_tokens": 240, "total_tokens": 2708 }
}
```

Unlike `/extract`, the `extractions` field is a **list** with one entry per message in the conversation, in order. Each entry contains the claims and intents extracted from the corresponding turn (using all prior turns as context).

#### 2. Python SDK

`claim-extractor` comes with built-in SDKs to invoke the server directly from Python (both sync and async).

**Synchronous API client:**

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    backend="api",
    api_url="http://localhost:8000",
)

result = ce.extract(
    "Your package is in transit and will arrive on December 12, 2025.",
    ai_service_description="You are a virtual assistant for a parcel delivery service. You can only answer questions about package tracking.",
)
```

**Asynchronous API client:**

```python
from orbitals.claim_extractor import AsyncClaimExtractor

ce = AsyncClaimExtractor(
    backend="api",
    api_url="http://localhost:8000",
)

result = await ce.extract(
    "Your package is in transit and will arrive on December 12, 2025.",
    ai_service_description="You are a virtual assistant for a parcel delivery service. You can only answer questions about package tracking.",
)
```

## FAQ

### vLLM is using too much GPU memory

If `claim-extractor` with the `vllm` backend (or `claim-extractor serve`) is consuming too much GPU memory, you can reduce the `gpu_memory_utilization` parameter with `ClaimExtractor(backend="vllm", gpu_memory_utilization=...)` (or set the `--vllm-gpu-memory-utilization` flag for `orbitals claim-extractor serve`). The default is 0.9 (90%), but you can lower it to free up GPU resources for other tasks.

### Getting Out of Memory (OOM) errors with vLLM

If you're experiencing OOM errors with the `vllm` backend (or when serving the model), you have two options:

1. **Lower `gpu_memory_utilization`**: Reduce the amount of GPU memory allocated to vLLM using the `gpu_memory_utilization` parameter or `--vllm-gpu-memory-utilization` flag.
2. **Reduce `max_model_len`**: Decrease the maximum model length using the `max_model_len` parameter or `--vllm-max-model-len` flag. **Note**: Be careful with this option, as the combined input and generated output must be shorter than the value you set.

## License

This project is licensed under the Apache 2.0 License.
