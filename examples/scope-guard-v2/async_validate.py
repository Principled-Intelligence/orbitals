"""Validating concurrently with the async guard.

`AsyncScopeGuardV2` has the same methods as the sync one, awaited. Use it when the
guard sits in front of a chat service and several conversations are in flight at
once, so a slow validation does not block the others.

Note that the async guard registers fewer backends than the sync one: `api` for the
hosted service, and `vllm-api` for a vLLM server you are running yourself. There is
no async equivalent of the in process `vllm` and `hf` backends.

    python async_validate.py --backend api --api-key principled_1234
    python async_validate.py --backend vllm-api --model <model> --vllm-url http://localhost:8000
"""

import argparse
import asyncio
import time

from orbitals.scope_guard_v2 import AsyncScopeGuardV2
from orbitals.types import AIServiceDescriptionV2, PredefinedResponse

AI_SERVICE_DESCRIPTION = AIServiceDescriptionV2(
    identity_role="Customer support assistant for a mobile network operator.",
    context=(
        "Subscribers message from the operator's app about their plan, their bill, or "
        "coverage where they live."
    ),
    knowledge_scope=(
        "Mobile plans and their allowances, billing, roaming charges, coverage, and "
        "device settings for data and voicemail."
    ),
    functionalities=[
        "Explain what a subscriber's plan includes and how much of it is used",
        "Explain a charge on the latest bill",
        "Explain roaming rates for a given country",
        "Walk a subscriber through the data settings on their handset",
    ],
    constraints=[
        "Never change a plan, add an add on, or issue a credit",
        "Never unlock a handset or discuss unlocking it",
    ],
    predefined_responses=[
        PredefinedResponse(
            trigger="the subscriber reports their phone lost or stolen",
            response=(
                "Call 1800 for an immediate bar on the SIM. We can send a replacement "
                "SIM to your registered address the same day."
            ),
        ),
    ],
    escalation_criteria=[
        "The subscriber says charges appeared for a service they never signed up to",
    ],
    response_guidelines="Give the number or the setting, then one line of context.",
)

QUERIES = [
    "How much data have I got left this month?",
    "What will calls cost me in Morocco next week?",
    "My phone was taken on the metro this morning.",
    "There's a 9 euro charge for something called Media Pass that I never ordered.",
    "Can you unlock my handset so I can use another operator's SIM?",
    "No voicemail notifications since I changed my phone. Any ideas?",
]


async def main():
    args = parse_args()
    scope_guard = build_scope_guard(args)

    # One at a time, for comparison.
    started = time.time()
    for query in QUERIES:
        await scope_guard.validate(
            query, ai_service_description=AI_SERVICE_DESCRIPTION
        )
    sequential = time.time() - started

    # All at once.
    started = time.time()
    results = await asyncio.gather(
        *(
            scope_guard.validate(query, ai_service_description=AI_SERVICE_DESCRIPTION)
            for query in QUERIES
        )
    )
    concurrent = time.time() - started

    for query, result in zip(QUERIES, results):
        print(f"> {query}")
        print(f"  class:     {result.scope_class.value}")
        print(f"  reasoning: {result.reasoning}")
        print()

    print(f"{len(QUERIES)} queries one after another: {sequential:.2f}s")
    print(f"{len(QUERIES)} queries concurrently:     {concurrent:.2f}s")

    # batch_validate is awaited the same way, and lets the backend batch the work
    # rather than opening one request per conversation.
    results = await scope_guard.batch_validate(
        QUERIES,
        ai_service_description=AI_SERVICE_DESCRIPTION,
        output_fields=["scope_class"],
    )
    print()
    print("# batch_validate, class only")
    for query, result in zip(QUERIES, results):
        print(f"  {result.scope_class.value:<22} {query}")


def build_scope_guard(args) -> AsyncScopeGuardV2:
    if args.backend == "api":
        return AsyncScopeGuardV2(
            backend="api",
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
        )
    return AsyncScopeGuardV2(
        backend="vllm-api",
        model=args.model,
        vllm_serving_url=args.vllm_url,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["api", "vllm-api"], default="api")
    parser.add_argument(
        "--model", default=None, help="Model name. Required for vllm-api."
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--api-key", default=None, help="Defaults to $PRINCIPLED_API_KEY."
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="Base URL of your own vLLM server, for the vllm-api backend.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())


# Sample run, to be filled in from a real run against the 4B.
#
# > How much data have I got left this month?
#   class:     Directly Supported
#   ...
