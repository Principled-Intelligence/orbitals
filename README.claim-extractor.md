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

Every extraction comes with **evidences**: verbatim excerpts from the source message that support the claim or intent. Evidences can be optionally disabled for lower-latency use cases.

`claim-extractor` is a building block for a wide range of downstream governance tasks: fact-checking, intent routing, response auditing, and hallucination detection. It is powered by specialized language models fine-tuned to extract atomic, self-contained claims and intents; different models are available with different deployment options:

| Model | Parameters | Hosting Options |
| :--- | :--- | :--- |
| claim-extractor-q | 4B | Self-hosted |
| claim-extractor-g | 4B | Self-hosted |
| claim-extractor-pro | ~ | Cloud-only |

> The open-weight checkpoints for `claim-extractor-q` and `claim-extractor-g` will be announced shortly. Reach out at [orbitals@principled-intelligence.com](mailto:orbitals@principled-intelligence.com) for early access.

## Quickstart with our open models

The easiest way to get started with ClaimExtractor is to use our open models, which you can self-host on consumer-grade GPUs and use via the `vllm` or `huggingface` backends.

First, install `orbitals` and `claim-extractor`:

```bash
pip install orbitals[claim-extractor-vllm]
# or if you prefer to use HuggingFace pipelines for inference instead of vLLM
# pip install orbitals[claim-extractor-hf]
```

Then:

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    backend="vllm",
    model="claim-extractor-q",    # for the Qwen-family model
    # model="claim-extractor-g",  # for the Gemma-family model
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

for claim in result.extractions.claims:
    print(f"[{claim.subtype}] {claim.content}")
    for evidence in claim.evidences:
        print(f"    evidence: {evidence}")

for intent in result.extractions.intents:
    print(f"[Intent] {intent.content}")

# [Factoid] The tracking number of the user's package is 1234567890
#     evidence: Your package with tracking number 1234567890 is currently in transit
# [Factoid] The user's package is expected to be delivered on December 12, 2025
#     evidence: expected to be delivered on December 12, 2025
# [Capability] The parcel delivery virtual assistant can notify the user when the package is out for delivery
#     evidence: I can also notify you when it is out for delivery.
```

## Quickstart with our hosted models

ClaimExtractor Pro is our most advanced model, available via API through our managed cloud hosting (get in touch with us for on-premise deployment).

> Open access to the hosted models is coming soon.
> Need access earlier? Contact us at [orbitals@principled-intelligence.com](mailto:orbitals@principled-intelligence.com).

Install plain `orbitals`:

```bash
pip install orbitals
```

Then:

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    backend="api",
    api_key="principled_1234",  # replace with your actual API key
)

ai_service_description = """
You are a virtual assistant for a parcel delivery service.
You can only answer questions about package tracking.
"""

result = ce.extract(
    "Your package is in transit and will arrive on December 12, 2025.",
    ai_service_description=ai_service_description,
)

for claim in result.extractions.claims:
    print(f"[{claim.subtype}] {claim.content}")
```

## Usage

### Initialization

Initialize the `ClaimExtractor` object by specifying the backend and model you want to use.

If you are using the self-hosted models, you can choose between the `vllm` and `huggingface` backends:

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    model="claim-extractor-q",        # for the Qwen-family model
    # model="claim-extractor-g",      # for the Gemma-family model
    backend="vllm",                   # or "huggingface"
)
```

If you are using the hosted models, use the `api` backend and provide your API key:

```python
from orbitals.claim_extractor import ClaimExtractor

ce = ClaimExtractor(
    backend="api",
    api_key="principled_1234",  # replace with your actual API key
)
```

### Extractions

`ce.extract(...)` returns a `ClaimExtractorOutput` whose `extractions` field contains the extracted `claims` and `intents`:

```python
from orbitals.claim_extractor import ClaimExtractor, Claim, Intent

ce = ClaimExtractor(backend="vllm", model="claim-extractor-q")
result = ce.extract(message, ai_service_description=ai_service_description)

assert isinstance(result.extractions.claims[0], Claim)
assert isinstance(result.extractions.intents[0], Intent)

for claim in result.extractions.claims:
    # claim.subtype is one of: "Factoid", "Capability", "User Assertion", "Unverifiable"
    print(claim.subtype, claim.content, claim.evidences)

for intent in result.extractions.intents:
    print(intent.content, intent.evidences)
```

Each `Claim` and `Intent` carries a decontextualized `content` string plus a list of verbatim `evidences` drawn from the source message.

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

### AI Service Description

The AI service description is **optional** for ClaimExtractor — you can pass `None` (the default) when you don't have one. However, providing a description meaningfully improves extraction quality:

1. It lets the model name the service explicitly in `Capability` claims.
2. It adds domain-specific specificity to every extraction.

You can provide it in two ways:

1. **As a single string** — a free-form description.
2. **As a structured object** — an `orbitals.types.AIServiceDescription` (**strongly recommended approach** for best performance).

### Skipping evidences

If you don't need verbatim evidences (and want to save latency and tokens), pass `skip_evidences=True` at init time or per-call:

```python
ce = ClaimExtractor(
    backend="vllm",
    model="claim-extractor-q",
    skip_evidences=True,
)

# or, override per-call:
result = ce.extract(message, ai_service_description=asd, skip_evidences=True)
```

When evidences are skipped, each `Claim` / `Intent` still has a `content` field but its `evidences` list will be empty.

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
pip install orbitals[claim-extractor-serve]

# start everything
orbitals claim-extractor serve claim-extractor-q --port 8000
```

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
                "evidences": ["Your package is in transit"]
            },
            {
                "subtype": "Factoid",
                "content": "The user's package is expected to arrive on December 12, 2025",
                "evidences": ["will arrive on December 12, 2025."]
            }
        ]
    },
    "time_taken": 0.47,
    "model": "<model>",
    "usage": { "prompt_tokens": 1234, "completion_tokens": 120, "total_tokens": 1354 }
}
```

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

If `claim-extractor` with the `vllm` backend (or `claim-extractor serve`) is consuming too much GPU memory, you can reduce the `gpu_memory_utilization` parameter in `VLLMClaimExtractor` (or set the `--vllm-gpu-memory-utilization` flag for `orbitals claim-extractor serve`). The default is 0.9 (90%), but you can lower it to free up GPU resources for other tasks.

### Getting Out of Memory (OOM) errors with vLLM

If you're experiencing OOM errors with the `vllm` backend (or when serving the model), you have two options:

1. **Lower `gpu_memory_utilization`**: Reduce the amount of GPU memory allocated to vLLM using the `gpu_memory_utilization` parameter or `--vllm-gpu-memory-utilization` flag.
2. **Reduce `max_model_len`**: Decrease the maximum model length using the `max_model_len` parameter or `--vllm-max-model-len` flag. **Note**: Be careful with this option, as the combined input and generated output must be shorter than the value you set.

## License

This project is licensed under the Apache 2.0 License.
