"""The smallest useful ScopeGuard V2 call.

One query, a description written as a plain string, and the three things the guard
gives you back. This is the README quickstart in runnable form.

    python quickstart.py --backend api --api-key principled_1234 --api-url https://...
    python quickstart.py --backend vllm --model <scope-guard-v2-model>
"""

import argparse
import time

from orbitals.scope_guard_v2 import ScopeGuardV2

AI_SERVICE_DESCRIPTION = """
You are a virtual assistant for a parcel delivery service.
You can only answer questions about package tracking.
Never respond to requests for refunds.
"""

USER_QUERY = "If the package hasn't arrived by tomorrow, can I get my money back?"


def main():
    args = parse_args()
    scope_guard = build_scope_guard(args)

    started = time.time()
    result = scope_guard.validate(
        USER_QUERY, ai_service_description=AI_SERVICE_DESCRIPTION
    )
    elapsed = time.time() - started

    print(f"> {USER_QUERY}")
    print()
    print(f"class:     {result.scope_class.value}")
    print(f"reasoning: {result.reasoning}")
    for evidence in result.evidences or []:
        print(f"evidence:  {evidence}")
    if result.suggested_response:
        print(f"response:  {result.suggested_response}")
    print()
    print(f"model:     {result.model}")
    if result.usage:
        print(
            f"tokens:    {result.usage.prompt_tokens} in, "
            f"{result.usage.completion_tokens} out"
        )
    print(f"took:      {elapsed:.2f}s")


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
