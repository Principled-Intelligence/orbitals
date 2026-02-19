import argparse
import asyncio
import time

from orbitals.scope_guard import AsyncScopeGuard


async def main():
    args = parse_args()

    scope_guard = AsyncScopeGuard(
        backend="api",
        model=args.model,
        api_url=args.api_url,
        skip_evidences=args.skip_evidences,
    )

    user_question = (
        "If the package hasn't arrived by tomorrow, can I get my money back?"
    )
    ai_service_description = (
        "You are a virtual assistant for a parcel delivery service. "
        "You can only answer questions about package tracking. "
        "Never respond to requests for refunds."
    )

    start_time = time.time()
    result = await scope_guard.validate(
        user_question, ai_service_description=ai_service_description
    )
    end_time = time.time()

    print(f"# scope: {result.scope_class}")
    if result.evidences:
        print("# evidences:")
        for evidence in result.evidences:
            print(f"  * evidence: {evidence}")

    print(f"# model: {result.model}")
    print("# usage:")
    print(f"  * prompt tokens: {result.usage.prompt_tokens}")
    print(f"  * completion tokens: {result.usage.completion_tokens}")
    print(f"  * total tokens: {result.usage.total_tokens}")

    print(f"# time: {end_time - start_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("api_url", type=str)
    parser.add_argument("-s", "--skip-evidences", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
