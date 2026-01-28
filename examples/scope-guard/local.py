import argparse
import time

from orbitals.scope_guard import ScopeGuard


def main():
    args = parse_args()

    scope_guard = ScopeGuard(
        backend=args.backend, model=args.model, skip_evidences=args.skip_evidences
    )

    user_question = (
        "If the package hasn't arrived by tomorrow, can I get my money back?"
    )
    ai_service_description = (
        "You are a virtual assistant for a parcel delivery service. "
        "You can only answer questions about package tracking. "
        "Never respond to requests for refunds."
    )

    # mock guard to be sure the classifier is warmed up
    # using different user query and ai service description to avoid backend caching
    scope_guard.validate("What is your name?", "You are an helpful AI assistant.")

    start_time = time.time()
    result = scope_guard.validate(user_question, ai_service_description)
    end_time = time.time()

    print(f"# scope: {result.scope_class}")
    if result.evidences:
        print("# evidences:")
        for evidence in result.evidences:
            print(f"  * evidence: {evidence}")

    print(f"# time: {end_time - start_time:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", type=str, choices=["hf", "vllm"])
    parser.add_argument("model", type=str)
    parser.add_argument("-s", "--skip-evidences", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
