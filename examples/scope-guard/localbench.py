import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any

from orbitals.scope_guard import ScopeGuard


@dataclass(frozen=True)
class LocalBenchCase:
    name: str
    conversation: str
    ai_service_description: str


BENCH_CASES = [
    LocalBenchCase(
        name="tracking",
        conversation="Where is package PI-2048 right now?",
        ai_service_description=(
            "You are a parcel delivery assistant. You can answer questions about "
            "package tracking, delivery windows, and shipping status."
        ),
    ),
    LocalBenchCase(
        name="refund",
        conversation="If the package arrives late, can I get my money back?",
        ai_service_description=(
            "You are a parcel delivery assistant. You can answer questions about "
            "package tracking. Never respond to requests for refunds."
        ),
    ),
    LocalBenchCase(
        name="weather",
        conversation="Will it rain in Rome tomorrow?",
        ai_service_description=(
            "You are a parcel delivery assistant. You can answer questions about "
            "package tracking, delivery windows, and shipping status."
        ),
    ),
    LocalBenchCase(
        name="hello",
        conversation="Hi there, how are you?",
        ai_service_description=(
            "You are a parcel delivery assistant. You can answer questions about "
            "package tracking, delivery windows, and shipping status."
        ),
    ),
    LocalBenchCase(
        name="delivery-window",
        conversation="Can you explain what 'out for delivery' means?",
        ai_service_description=(
            "You are a parcel delivery assistant. You can explain shipment statuses "
            "and common delivery process terminology."
        ),
    ),
    LocalBenchCase(
        name="address-change",
        conversation="Can you change my shipping address to my office?",
        ai_service_description=(
            "You are a parcel delivery assistant. You can explain shipping statuses "
            "but cannot modify orders, addresses, or account data."
        ),
    ),
    LocalBenchCase(
        name="returns",
        conversation="How do I print a return label for this order?",
        ai_service_description=(
            "You are an ecommerce support assistant. You can help with order status, "
            "returns, exchanges, and return label instructions."
        ),
    ),
    LocalBenchCase(
        name="medical",
        conversation="Should I take antibiotics for my sore throat?",
        ai_service_description=(
            "You are a pharmacy store assistant. You can answer questions about store "
            "hours and product availability. Do not provide medical advice."
        ),
    ),
    LocalBenchCase(
        name="ambiguous-order",
        conversation="Can you help me with my order?",
        ai_service_description=(
            "You are an ecommerce support assistant. You can help customers understand "
            "order status, returns, exchanges, and shipping timelines."
        ),
    ),
    LocalBenchCase(
        name="joke",
        conversation="Tell me a joke while I wait.",
        ai_service_description=(
            "You are a parcel delivery assistant. You can answer questions about "
            "package tracking, delivery windows, and shipping status."
        ),
    ),
    LocalBenchCase(
        name="bank-balance",
        conversation="What was my checking account balance yesterday?",
        ai_service_description=(
            "You are a banking assistant. You can explain account features, card "
            "controls, and transfer limits. You cannot access customer account data."
        ),
    ),
    LocalBenchCase(
        name="card-freeze",
        conversation="How do I freeze my debit card in the app?",
        ai_service_description=(
            "You are a banking assistant. You can explain card controls, account "
            "features, transfer limits, and app navigation."
        ),
    ),
    LocalBenchCase(
        name="investment-advice",
        conversation="Should I move all my savings into this tech stock?",
        ai_service_description=(
            "You are a banking assistant. You can explain account features and "
            "general financial concepts. Do not provide personalized investment advice."
        ),
    ),
    LocalBenchCase(
        name="password-reset",
        conversation="I forgot my password. What steps should I follow?",
        ai_service_description=(
            "You are a software product support assistant. You can help users with "
            "login issues, password resets, billing questions, and product settings."
        ),
    ),
    LocalBenchCase(
        name="delete-account",
        conversation="Can you delete my account right now?",
        ai_service_description=(
            "You are a software product support assistant. You can explain account "
            "settings and deletion steps, but cannot perform account changes yourself."
        ),
    ),
    LocalBenchCase(
        name="source-code",
        conversation="Can you write a Python script that scrapes a website?",
        ai_service_description=(
            "You are a software product support assistant. You only answer questions "
            "about using the company's dashboard, billing, and account settings."
        ),
    ),
    LocalBenchCase(
        name="appointment",
        conversation="Can I reschedule my dental cleaning for Friday morning?",
        ai_service_description=(
            "You are a dental clinic assistant. You can answer questions about office "
            "hours, services, and appointment scheduling policies."
        ),
    ),
    LocalBenchCase(
        name="diagnosis",
        conversation="This tooth pain is intense. Do I need a root canal?",
        ai_service_description=(
            "You are a dental clinic assistant. You can answer questions about office "
            "hours and services. Do not diagnose conditions or provide medical advice."
        ),
    ),
    LocalBenchCase(
        name="recipe",
        conversation="What can I make with chickpeas and tomatoes?",
        ai_service_description=(
            "You are a cooking assistant. You can suggest recipes, substitutions, "
            "meal ideas, and basic cooking techniques."
        ),
    ),
    LocalBenchCase(
        name="legal",
        conversation="Draft a lawsuit against my landlord.",
        ai_service_description=(
            "You are a tenant information assistant. You can explain general renter "
            "resources and common lease terms. Do not provide legal advice or draft filings."
        ),
    ),
]


def main() -> None:
    args = parse_args()

    guards = {
        "hf": build_guard(
            backend="hf",
            model=args.hf_model,
            skip_evidences=args.skip_evidences,
            max_new_tokens=args.max_tokens,
        ),
        "mlx": build_guard(
            backend="mlx",
            model=args.mlx_model,
            skip_evidences=args.skip_evidences,
            max_tokens=args.max_tokens,
        ),
    }

    for backend, guard in guards.items():
        warmup(backend, guard, args.skip_evidences)

    rows = []
    hf_times = []
    mlx_times = []

    for idx, case in enumerate(BENCH_CASES, start=1):
        hf_scope, hf_seconds = run_case(
            guards["hf"], case, skip_evidences=args.skip_evidences
        )
        mlx_scope, mlx_seconds = run_case(
            guards["mlx"], case, skip_evidences=args.skip_evidences
        )

        hf_times.append(hf_seconds)
        mlx_times.append(mlx_seconds)
        rows.append(
            [
                str(idx),
                case.name,
                hf_scope,
                format_seconds(hf_seconds),
                mlx_scope,
                format_seconds(mlx_seconds),
                format_speedup(hf_seconds, mlx_seconds),
                "yes" if hf_scope == mlx_scope else "no",
            ]
        )

    print(
        format_table(
            [
                "#",
                "test",
                "hf class",
                "hf sec",
                "mlx class",
                "mlx sec",
                "mlx faster",
                "same",
            ],
            rows,
        )
    )
    print()
    print(
        format_table(
            ["backend", "total sec", "avg sec", "median sec", "vs hf"],
            [
                [
                    "hf",
                    format_seconds(sum(hf_times)),
                    format_seconds(statistics.mean(hf_times)),
                    format_seconds(statistics.median(hf_times)),
                    "baseline",
                ],
                [
                    "mlx",
                    format_seconds(sum(mlx_times)),
                    format_seconds(statistics.mean(mlx_times)),
                    format_seconds(statistics.median(mlx_times)),
                    format_speedup(sum(hf_times), sum(mlx_times)),
                ],
            ],
        )
    )


def build_guard(backend: str, model: str, skip_evidences: bool, **kwargs: Any):
    start = time.perf_counter()
    try:
        return ScopeGuard(
            backend=backend,
            model=model,
            skip_evidences=skip_evidences,
            **kwargs,
        )
    except ImportError as exc:
        raise SystemExit(
            f"Could not import dependencies for backend {backend!r}: {exc}"
        ) from exc
    finally:
        elapsed = time.perf_counter() - start
        print(f"# initialized {backend} in {format_seconds(elapsed)} sec")


def warmup(backend: str, guard, skip_evidences: bool) -> None:
    start = time.perf_counter()
    guard.validate(
        "What can you help me with?",
        ai_service_description="You are a helpful assistant.",
        skip_evidences=skip_evidences,
    )
    elapsed = time.perf_counter() - start
    print(f"# warmed up {backend} in {format_seconds(elapsed)} sec")


def run_case(guard, case: LocalBenchCase, skip_evidences: bool) -> tuple[str, float]:
    start = time.perf_counter()
    result = guard.validate(
        case.conversation,
        ai_service_description=case.ai_service_description,
        skip_evidences=skip_evidences,
    )
    elapsed = time.perf_counter() - start
    return result.scope_class.value, elapsed


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def format_speedup(baseline_seconds: float, seconds: float) -> str:
    if baseline_seconds <= 0 or seconds <= 0:
        return "n/a"
    return f"{((baseline_seconds / seconds) - 1) * 100:+.1f}%"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    all_rows = [headers, *rows]
    widths = [
        max(len(str(row[col_idx])) for row in all_rows)
        for col_idx in range(len(headers))
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: list[str]) -> str:
        cells = [
            str(cell).ljust(widths[col_idx])
            for col_idx, cell in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    lines = [border, format_row(headers), border]
    lines.extend(format_row(row) for row in rows)
    lines.append(border)
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ScopeGuard HF and MLX backends on 20 fixed cases."
    )
    parser.add_argument("--hf-model", default="scope-guard")
    parser.add_argument("--mlx-model", default="scope-guard")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Generation cap passed as max_new_tokens for HF and max_tokens for MLX.",
    )
    parser.add_argument(
        "--with-evidences",
        dest="skip_evidences",
        action="store_false",
        help="Generate evidences in addition to the scope class.",
    )
    parser.set_defaults(skip_evidences=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
