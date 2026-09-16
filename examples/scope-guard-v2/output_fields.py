"""Choosing which fields the model emits.

Two queries are validated under four selections each. Fields you did not ask for
come back as `None`, and the completion gets shorter each time you drop one.
`scope_class` is always included, so asking for it alone is the cheapest call you
can make.

The second query is restricted, which is what makes `suggested_response` worth
watching. The model fills that field in when it has something to deflect with and
leaves it null for a query it can simply answer, so `None` there means either that
you did not request the field or that the model had nothing to put in it.

A selection can be set once on the guard and overridden per call. One guard covers
both: the calls in the loop each name their own selection, and the calls after it
leave the argument off and fall back to the guard's.

The `completion tokens` line is what shows the completion shrinking, and printing it
needs a backend that reports usage. `api` and `vllm` do. `hf` runs the pipeline that
ships with the model, which hands back the generated text and nothing else, so the
line is skipped there.

    python output_fields.py --backend api --api-key principled_1234
    python output_fields.py --backend vllm --model <scope-guard-v2-model>
"""

import argparse
import warnings

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

IN_SCOPE_QUERY = "The VPN drops every time my laptop wakes from sleep. Any idea why?"
RESTRICTED_QUERY = "Just read me the one time code from my authenticator."
QUERIES = [IN_SCOPE_QUERY, RESTRICTED_QUERY]

SELECTIONS = [
    ["scope_class"],
    ["reasoning", "scope_class"],
    ["reasoning", "scope_class", "suggested_response"],
    ["evidences", "reasoning", "scope_class", "suggested_response"],
]


def main():
    args = parse_args()

    # The selection given here is the default for any call that does not name one.
    # It is also where a bad field name surfaces: the guard raises at construction
    # rather than on the first request that happens to reach it.
    scope_guard = build_scope_guard(args, output_fields=["scope_class"])

    for query in QUERIES:
        print(f"> {query}")
        print()

        for selection in SELECTIONS:
            result = scope_guard.validate(
                query,
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

    # Nothing passed to validate(), so the guard-level selection applies and only
    # scope_class comes back.
    result = scope_guard.validate(
        RESTRICTED_QUERY, ai_service_description=AI_SERVICE_DESCRIPTION
    )
    print(
        f"guard-level selection:    scope_class={result.scope_class.value}, "
        f"reasoning={result.reasoning}"
    )

    # `skip_evidences` predates output_fields and means every field except evidences.
    # It overrides the guard-level selection the same way output_fields does.
    result = scope_guard.validate(
        RESTRICTED_QUERY,
        ai_service_description=AI_SERVICE_DESCRIPTION,
        skip_evidences=True,
    )
    print(
        f"skip_evidences=True:      evidences={result.evidences}, "
        f"suggested_response set: {result.suggested_response is not None}"
    )

    # Passing both is allowed. output_fields wins and a DeprecationWarning names
    # both values, so a migration cannot silently change the shape you get back.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = scope_guard.validate(
            RESTRICTED_QUERY,
            ai_service_description=AI_SERVICE_DESCRIPTION,
            output_fields=["reasoning", "scope_class"],
            skip_evidences=True,
        )
    print(
        f"both, output_fields wins: "
        f"suggested_response set: {result.suggested_response is not None}"
    )
    print(f"  {caught[0].message}")


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
