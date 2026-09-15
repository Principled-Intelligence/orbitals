"""Choosing which fields the model emits.

The same query is validated four times under four different selections. Fields you
did not ask for come back as `None`, and the completion gets shorter each time you
drop one. `scope_class` is always included, so asking for it explicitly is the
cheapest call you can make.

A selection can be set once on the guard, or per call, and the per-call value wins.

    python output_fields.py --backend api --api-key principled_1234
    python output_fields.py --backend vllm --model <scope-guard-v2-model>
"""

import argparse

from orbitals.scope_guard_v2 import ScopeGuardV2
from orbitals.types import AIServiceDescriptionV2

AI_SERVICE_DESCRIPTION = AIServiceDescriptionV2(
    identity_role="Internal IT helpdesk assistant for the staff of a design agency.",
    context=(
        "Employees message the helpdesk from Slack when something on their laptop or "
        "in one of the company tools stops working."
    ),
    knowledge_scope=(
        "Company laptops, the VPN, single sign on, printers, and the licensed design "
        "and collaboration tools the agency pays for."
    ),
    functionalities=[
        "Walk an employee through resetting their single sign on password",
        "Diagnose common VPN connection failures",
        "Explain how to request a licence for a design tool",
        "Report a broken laptop and start a repair ticket",
    ],
    constraints=[
        "Never ask an employee to share a password or a one time code",
        "Never grant, extend, or revoke access to a system",
        "Never advise on anything related to payroll, contracts, or performance reviews",
    ],
    escalation_criteria=[
        "The employee describes something that looks like a security incident on their "
        "own machine, such as a device they no longer control or files they cannot open",
    ],
    response_guidelines="Write short steps the employee can follow without a call.",
)

USER_QUERY = "The VPN drops every time my laptop wakes from sleep. Any idea why?"

SELECTIONS = [
    ["scope_class"],
    ["reasoning", "scope_class"],
    ["reasoning", "scope_class", "suggested_response"],
    ["evidences", "reasoning", "scope_class", "suggested_response"],
]


def main():
    args = parse_args()
    scope_guard = build_scope_guard(args)

    print(f"> {USER_QUERY}")
    print()

    for selection in SELECTIONS:
        result = scope_guard.validate(
            USER_QUERY,
            ai_service_description=AI_SERVICE_DESCRIPTION,
            output_fields=selection,
        )

        print(f"requested: {selection}")
        print(f"  scope_class:        {result.scope_class.value}")
        print(f"  reasoning:          {result.reasoning}")
        print(f"  evidences:          {result.evidences}")
        print(f"  suggested_response: {result.suggested_response}")
        if result.usage:
            print(f"  completion tokens:  {result.usage.completion_tokens}")
        print()

    # A selection can also be fixed once, when every call in your service wants the
    # same shape. Anything passed to validate() still overrides it.
    cheap_guard = build_scope_guard(args, output_fields=["scope_class"])
    result = cheap_guard.validate(
        USER_QUERY, ai_service_description=AI_SERVICE_DESCRIPTION
    )
    print(f"guard-level selection:  {result.scope_class.value}")

    # `skip_evidences` predates output_fields and means every field except evidences.
    # Passing both is allowed, output_fields wins, and you get a DeprecationWarning.
    legacy_guard = build_scope_guard(args, skip_evidences=True)
    result = legacy_guard.validate(
        USER_QUERY, ai_service_description=AI_SERVICE_DESCRIPTION
    )
    print(f"skip_evidences=True:    evidences is {result.evidences}")


def build_scope_guard(args, **overrides) -> ScopeGuardV2:
    if args.backend == "api":
        return ScopeGuardV2(
            backend="api",
            model=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
            **overrides,
        )
    return ScopeGuardV2(backend=args.backend, model=args.model, **overrides)


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
# requested: ['scope_class']
#   scope_class:        Directly Supported
#   reasoning:          None
#   ...
