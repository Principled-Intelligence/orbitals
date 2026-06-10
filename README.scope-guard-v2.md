<div align="left">
    <h1>
        <strong>ScopeGuard V2</strong>
    </h1>
</div>

ScopeGuard V2 is the next generation of [ScopeGuard](README.scope-guard.md). Given the specifications of an AI service and a conversation, `scope-guard-v2` classifies the last user message into one of the following seven classes:

* **Directly Supported**: The query is clearly within the AI service's capabilities.
* **Potentially Supported**: The query could plausibly be handled by the AI service.
* **Predefined Answer**: The query matches a predefined scenario or FAQ with a specific, pre-written response.
* **Human Oversight**: The query requires human oversight, judgment, or expertise (e.g., it meets an escalation criterion).
* **Out of Scope**: The query is outside the AI service's defined role.
* **Restricted**: The query cannot be handled due to a specific constraint.
* **Chit Chat**: The query is a social interaction not related to the AI service's function.

On top of the classification, ScopeGuard V2 returns:

* **`evidences`**: verbatim spans from the AI service description supporting the decision (optional, can be skipped for lower latency)
* **`reasoning`**: a short explanation of why the class was chosen
* **`suggested_response`**: a ready-to-use response for the user when the query should not be processed (e.g., for Predefined Answer, Human Oversight, Out of Scope, Restricted, or Chit Chat)

> [!IMPORTANT]
> **Model availability**: The ScopeGuard V2 models are currently **private** and accessible **only via API** through Principled Intelligence's hosted service. An open-weight release will follow.
> Want access? Contact us at [orbitals@principled-intelligence.com](mailto:orbitals@principled-intelligence.com).

## Quickstart

Since ScopeGuard V2 is API-only for the moment, plain `orbitals` is all you need:

```bash
pip install orbitals
```

Then:

```python
from orbitals.scope_guard_v2 import ScopeGuardV2

sg = ScopeGuardV2(
    backend="api",
    api_key="principled_1234",  # replace with your actual API key
)

ai_service_description = """
You are a virtual assistant for a parcel delivery service.
You can only answer questions about package tracking.
Never respond to requests for refunds.
"""

user_query = "If the package hasn't arrived by tomorrow, can I get my money back?"
result = sg.validate(user_query, ai_service_description=ai_service_description)

print(f"Scope: {result.scope_class.value}")
print(f"Reasoning: {result.reasoning}")
if result.suggested_response:
    print(f"Suggested response: {result.suggested_response}")

# Scope: Restricted
# Reasoning: The request asks for a refund, which the service description explicitly forbids.
# Suggested response: I'm sorry, but I cannot help with refund requests. I can assist you with package tracking.
```

An asynchronous client is also available:

```python
from orbitals.scope_guard_v2 import AsyncScopeGuardV2

sg = AsyncScopeGuardV2(
    backend="api",
    api_key="principled_1234",  # replace with your actual API key
)

result = await sg.validate(user_query, ai_service_description=ai_service_description)
```

## Usage

### Scope Classes

The possible scope classes returned by `scope-guard-v2` are:

```python
from orbitals.scope_guard_v2 import ScopeClass

print(ScopeClass.DIRECTLY_SUPPORTED.value)    # "Directly Supported"
print(ScopeClass.POTENTIALLY_SUPPORTED.value) # "Potentially Supported"
print(ScopeClass.PREDEFINED_ANSWER.value)     # "Predefined Answer"
print(ScopeClass.HUMAN_OVERSIGHT.value)       # "Human Oversight"
print(ScopeClass.OUT_OF_SCOPE.value)          # "Out of Scope"
print(ScopeClass.RESTRICTED.value)            # "Restricted"
print(ScopeClass.CHIT_CHAT.value)             # "Chit Chat"
```

For example, you can check the scope class of a validation result as follows:

```python
result = sg.validate(user_query, ai_service_description=ai_service_description)

# Using the Enum member:
if result.scope_class == ScopeClass.RESTRICTED:
    print("The user query is restricted.")

# Or using the string value:
if result.scope_class.value == "Restricted":
    print("The user query is restricted.")
```

### Structured AI Service Description

While the AI service description can still be provided as a plain string, ScopeGuard V2 introduces `orbitals.types.AIServiceDescriptionV2`, a structured description that drives the new scope classes (**strongly recommended approach**):

```python
from orbitals.types import AIServiceDescriptionV2, PredefinedResponse

ai_service_description = AIServiceDescriptionV2(
    identity_role="Virtual assistant for a parcel delivery service.",
    context="Operates for an e-commerce logistics company; users are customers tracking their orders.",
    knowledge_scope="Package tracking, delivery times, and shipment statuses.",
    functionalities=[
        "Track packages by tracking number",
        "Provide expected delivery dates",
    ],
    constraints=[
        "Never respond to requests for refunds",
    ],
    predefined_responses=[
        PredefinedResponse(
            trigger="user asks for the support phone number",
            response="You can reach our support team at +1-800-555-0100.",
        ),
    ],
    escalation_criteria=[
        "The user reports a lost or damaged package",
        "The user threatens legal action",
    ],
    response_guidelines="Be concise and friendly.",
)

result = sg.validate(user_query, ai_service_description=ai_service_description)
```

* **`predefined_responses`** powers the **Predefined Answer** class: when a trigger matches, the suggested response contains the pre-written answer.
* **`escalation_criteria`** powers the **Human Oversight** class: when a criterion is met, the query is flagged for human review.
* **`constraints`** powers the **Restricted** class.

### Input Formats

The `validate` method accepts the same input formats as ScopeGuard V1:

```python
# User query as a string
result = sg.validate(
    "When is my package scheduled to arrive?",
    ai_service_description=ai_service_description,
)

# User query as a dictionary (OpenAI's API Message)
result = sg.validate(
    {"role": "user", "content": "When is my package scheduled to arrive?"},
    ai_service_description=ai_service_description,
)

# Conversation as a list of dictionaries
result = sg.validate(
    [
        {"role": "user", "content": "I ordered a package, tracking number 1234567890"},
        {"role": "assistant", "content": "Great, the package is in transit. What would you like to know?"},
        {"role": "user", "content": "If it doesn't arrive tomorrow, can I get a refund"},
    ],
    ai_service_description=ai_service_description,
)
```

With multi-turn conversations, the scope decision is computed exclusively for **the final user message**, with earlier turns considered only as conversational context.

### Skipping Evidences

If you don't need the supporting evidences, you can skip them to reduce latency and token usage, either per guard or per call:

```python
sg = ScopeGuardV2(backend="api", api_key="principled_1234", skip_evidences=True)

# or per call
result = sg.validate(
    user_query,
    ai_service_description=ai_service_description,
    skip_evidences=True,
)
```

### Default Safety Principles

Like V1, ScopeGuard V2 can optionally add a built-in set of general safety restrictions to the AI service description before classification. When the description is an `AIServiceDescriptionV2`, they are appended to its `constraints`; service-specific rules are overridden on conflict.

```python
sg = ScopeGuardV2(
    backend="api",
    api_key="principled_1234",
    include_default_safety_principles=True,
)

# or per call
result = sg.validate(
    user_query,
    ai_service_description=ai_service_description,
    include_default_safety_principles=True,
)
```

### Batch Processing

You can process multiple conversations at once using `batch_validate`, with either a single shared AI service description or one per conversation:

```python
queries = [
    "If the package hasn't arrived by tomorrow, can I get my money back?",
    "When is the package expected to be delivered?",
]

# Single AI Service Description
results = sg.batch_validate(queries, ai_service_description=ai_service_description)

# Multiple AI Service Descriptions (one per conversation)
results = sg.batch_validate(queries, ai_service_descriptions=[desc_1, desc_2])
```

## Self-hosting

The library already ships the `vllm`, `hf`, and serving backends for ScopeGuard V2 (`orbitals scope-guard-v2 serve`), but they require access to the model weights, which are **private for the moment**. Once the open-weight models are released, self-hosting will work exactly as it does for [ScopeGuard V1](README.scope-guard.md#serving-scopeguard-on-premise-or-on-your-infrastructure).

If you need on-premise deployment earlier, contact us at [orbitals@principled-intelligence.com](mailto:orbitals@principled-intelligence.com).

## License

This project is licensed under the Apache 2.0 License.
