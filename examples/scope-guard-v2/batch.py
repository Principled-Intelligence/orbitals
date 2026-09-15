"""Validating many conversations in one call.

`batch_validate` takes a list of conversations and returns a list of results in the
same order. You can give it one description that applies to all of them, or one
description per conversation when you run several assistants from the same process.

The second case is the one that is easy to get wrong: `ai_service_descriptions` is a
different argument from `ai_service_description`, and the list has to be the same
length as the conversations.

    python batch.py --backend api --api-key principled_1234
    python batch.py --backend vllm --model <scope-guard-v2-model>
"""

import argparse

from orbitals.scope_guard_v2 import ScopeGuardV2
from orbitals.types import AIServiceDescriptionV2

# The intake assistant a claimant talks to first.
INTAKE = AIServiceDescriptionV2(
    identity_role="First line intake assistant for a home insurance claims team.",
    context=(
        "Policyholders open a claim through the insurer's app, often within hours of "
        "the damage happening."
    ),
    knowledge_scope=(
        "What a home policy covers, what evidence a claim needs, and how long each "
        "stage of a claim takes."
    ),
    functionalities=[
        "Explain what a home policy covers for water, fire, and theft damage",
        "List the photographs and documents a claim needs",
        "Give the current stage and expected timing of an open claim",
    ],
    constraints=[
        "Never approve, reject, or put a value on a claim",
        "Never advise the policyholder on what to say to a loss adjuster",
    ],
    escalation_criteria=[
        "The policyholder reports a fire, a flood, or anything that has made the home "
        "unsafe to live in",
    ],
    response_guidelines="Ask for one thing at a time.",
)

# The assistant that handles renewals, run by the same company.
RENEWALS = AIServiceDescriptionV2(
    identity_role="Renewals assistant for the same home insurance provider.",
    context="Policyholders arrive here in the weeks before their policy renews.",
    knowledge_scope="Renewal dates, premium changes, and cover options on a policy.",
    functionalities=[
        "Explain why a renewal premium has changed",
        "Compare the cover levels available at renewal",
        "Explain how to add or remove an optional cover",
    ],
    constraints=[
        "Never quote a price for a new policy",
        "Never discuss an open claim",
    ],
    response_guidelines="Be factual about numbers and avoid selling.",
)

CONVERSATIONS = [
    "A pipe burst under my kitchen sink last night. What do I need to send you?",
    "How long does a theft claim usually take once I've sent the crime reference?",
    "There's been a fire in the flat and we can't stay here tonight.",
    "Do you cover the shed as well as the house?",
]


def main():
    args = parse_args()
    scope_guard = build_scope_guard(args)

    print("# one description shared by every conversation")
    results = scope_guard.batch_validate(
        CONVERSATIONS, ai_service_description=INTAKE
    )
    for conversation, result in zip(CONVERSATIONS, results):
        print(f"> {conversation}")
        print(f"  class:     {result.scope_class.value}")
        print(f"  reasoning: {result.reasoning}")
        print()

    print("# one description per conversation")
    mixed = [
        "There's water coming through the ceiling from the flat above.",
        "Why has my premium gone up by 60 euros this year?",
    ]
    results = scope_guard.batch_validate(
        mixed, ai_service_descriptions=[INTAKE, RENEWALS]
    )
    for conversation, description, result in zip(mixed, [INTAKE, RENEWALS], results):
        print(f"> {conversation}")
        print(f"  assistant: {description.identity_role}")
        print(f"  class:     {result.scope_class.value}")
        print(f"  reasoning: {result.reasoning}")
        print()

    # Output field selection works on a batch too, and is worth using here: a batch
    # is where the saved generation adds up.
    results = scope_guard.batch_validate(
        CONVERSATIONS,
        ai_service_description=INTAKE,
        output_fields=["scope_class"],
    )
    print("# class only")
    for conversation, result in zip(CONVERSATIONS, results):
        print(f"  {result.scope_class.value:<22} {conversation}")


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
# # one description shared by every conversation
# > A pipe burst under my kitchen sink last night. What do I need to send you?
#   class:     Directly Supported
#   ...
