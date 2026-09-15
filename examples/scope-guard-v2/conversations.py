"""The three shapes `validate` accepts for a conversation.

A bare string, a single message dict, and a list of messages. The last one is the
one worth understanding: in a multi turn conversation the decision is made about the
final user message only, and the earlier turns are context that helps interpret it.

The last two calls show why that matters. The same short question, "and if I cancel
instead?", lands in a different class depending on what came before it.

    python conversations.py --backend api --api-key principled_1234
    python conversations.py --backend vllm --model <scope-guard-v2-model>
"""

import argparse

from orbitals.scope_guard_v2 import ScopeGuardV2
from orbitals.types import AIServiceDescriptionV2, PredefinedResponse

AI_SERVICE_DESCRIPTION = AIServiceDescriptionV2(
    identity_role="Booking assistant for a short haul airline.",
    context=(
        "Passengers reach the assistant from the airline website, usually while "
        "holding a booking reference and looking at an upcoming flight."
    ),
    knowledge_scope=(
        "Flight times, baggage allowances, seat selection, check in, and the fare "
        "rules that apply to a booking."
    ),
    functionalities=[
        "Look up the times and status of a booked flight",
        "Explain the baggage allowance that applies to a fare",
        "Explain what changing a flight costs under the passenger's fare rules",
        "Explain the check in windows and what to do if one has closed",
    ],
    constraints=[
        "Never process a payment or a refund",
        "Never promise compensation for a delay or a cancellation",
    ],
    predefined_responses=[
        PredefinedResponse(
            trigger="the passenger asks to cancel a booking",
            response=(
                "Cancellations are handled in Manage Booking on the website. What you "
                "get back depends on your fare, and the page shows the amount before "
                "you confirm."
            ),
        ),
    ],
    escalation_criteria=[
        "The passenger says they are travelling with an unaccompanied minor or needs "
        "assistance the website cannot arrange",
    ],
    response_guidelines="Answer with the rule that applies, then the next step.",
)


def main():
    args = parse_args()
    scope_guard = build_scope_guard(args)

    # 1. A plain string.
    show(
        scope_guard,
        "a string",
        "How much hand luggage can I take on a Basic fare?",
    )

    # 2. A single message, in the shape OpenAI's API uses.
    show(
        scope_guard,
        "a message dict",
        {"role": "user", "content": "Has flight VY6218 left yet?"},
    )

    # 3. A conversation. Only the last user message is classified.
    show(
        scope_guard,
        "a conversation, after a question about changing a flight",
        [
            {"role": "user", "content": "I booked VY6218 for Thursday, reference QT4N9P."},
            {
                "role": "assistant",
                "content": "Found it. Thursday 09:40 to Barcelona. What would you like to know?",
            },
            {"role": "user", "content": "What would it cost me to move it to Friday?"},
            {
                "role": "assistant",
                "content": "On a Basic fare a change costs the fare difference plus a 40 euro fee.",
            },
            {"role": "user", "content": "And if I cancel instead?"},
        ],
    )

    # 4. The same final message with different context before it. The guard reads it
    #    as a question about the fare rules rather than a request to cancel.
    show(
        scope_guard,
        "a conversation, after a question about fare rules",
        [
            {"role": "user", "content": "I'm comparing your Basic and Flex fares."},
            {
                "role": "assistant",
                "content": "Happy to help. Flex allows free changes up to two hours before departure.",
            },
            {"role": "user", "content": "And if I cancel instead?"},
        ],
    )


def show(scope_guard, label, conversation):
    result = scope_guard.validate(
        conversation, ai_service_description=AI_SERVICE_DESCRIPTION
    )

    print(f"# {label}")
    if isinstance(conversation, list):
        print(f"  last user message: {conversation[-1]['content']}")
    print(f"  class:     {result.scope_class.value}")
    print(f"  reasoning: {result.reasoning}")
    if result.suggested_response:
        print(f"  response:  {result.suggested_response}")
    print()


def build_scope_guard(args) -> ScopeGuardV2:
    if args.backend == "api":
        return ScopeGuardV2(
            backend="api",
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
        )
    return ScopeGuardV2(backend=args.backend, model=args.model)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["api", "vllm", "hf"], default="api")
    parser.add_argument(
        "--model", default=None, help="Model name or path. Required for vllm and hf."
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--api-key", default=None, help="Defaults to $PRINCIPLED_API_KEY."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()


# Sample run, to be filled in from a real run against the 4B.
#
# # a string
#   class:     Directly Supported
#   reasoning: ...
