"""Scope classes in action, driven by a structured AI service description.

Runs one query per scope class against a retail bank support assistant, so you can
see which part of the description produced each answer. The interesting fields are
`constraints`, which produces Restricted, `predefined_responses`, which produces
Predefined Answer, and `escalation_criteria`, which produces Human Oversight.

Potentially Supported is the fussiest of the seven. It needs a query sitting next to
the description without being named by it, which no constraint, escalation criterion,
or predefined response claims first. The Apple Pay question is card and app business
that none of the functionalities cover, and it only stays in that gap because the
knowledge scope above is narrow.

    python structured_description.py --backend api --api-key principled_1234
    python structured_description.py --backend vllm --model <scope-guard-v2-model>
"""

import argparse

from orbitals.scope_guard_v2 import ScopeGuardV2
from orbitals.types import AIServiceDescriptionV2, PredefinedResponse

AI_SERVICE_DESCRIPTION = AIServiceDescriptionV2(
    identity_role="Support assistant inside the mobile app of a retail bank.",
    context=(
        "Users are personal banking customers. They open the assistant from the app, "
        "already signed in, usually to ask about a card payment they do not recognise "
        "or about a fee on their statement."
    ),
    # Deliberately tight. A broader scope, such as one that also named account
    # statements, pulls the Apple Pay query below into Directly Supported and the
    # Potentially Supported class stops being reachable here at all.
    knowledge_scope=(
        "Card payments, the fees charged on them, and the parts of the mobile app "
        "where a customer can see both."
    ),
    functionalities=[
        "Explain an individual card transaction on the customer's statement",
        "Break down the fees charged on an account over a given period",
        "Walk the customer through freezing or reissuing a card in the app",
        "Explain how long a transfer or a refund normally takes to arrive",
    ],
    constraints=[
        "Never give investment, tax, or legal advice",
        "Never move money, open or close accounts, or change a credit limit",
        "Never read out a full card number, even if the customer asks",
    ],
    predefined_responses=[
        PredefinedResponse(
            trigger="the customer says their card has been lost or stolen",
            response=(
                "Freeze the card from Cards > Freeze in the app, then call us on "
                "+44 800 555 0142 so we can send a replacement."
            ),
        ),
        PredefinedResponse(
            trigger="the customer asks how to raise a formal complaint",
            response=(
                "You can raise a complaint at bank.example/complaints. We reply within "
                "five working days."
            ),
        ),
    ],
    escalation_criteria=[
        "The customer says payments were made from their account without their knowledge",
        "The customer disputes a charge and asks for it to be reversed",
    ],
    response_guidelines=(
        "Be brief and concrete. Use the customer's own wording for amounts and dates."
    ),
)

# One query per scope class, in the order the classes are listed in the README.
QUERIES = [
    "Why was I charged 12 euros by SNCF last Tuesday?",
    "Can I add this card to Apple Pay?",
    "I think I left my card in a taxi last night.",
    "There are three payments to an electronics shop that I never made.",
    "Can you book me a dentist appointment for Friday?",
    "Should I move my savings into Tesla stock?",
    "Thanks, that was quicker than calling you.",
]


def main():
    args = parse_args()
    scope_guard = build_scope_guard(args)

    for query in QUERIES:
        result = scope_guard.validate(
            query, ai_service_description=AI_SERVICE_DESCRIPTION
        )

        print(f"> {query}")
        print(f"  class:     {result.scope_class.value}")
        print(f"  reasoning: {result.reasoning}")
        for evidence in result.evidences or []:
            print(f"  evidence:  {evidence}")
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
