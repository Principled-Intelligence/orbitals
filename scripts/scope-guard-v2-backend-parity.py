"""Compare two ScopeGuard V2 backends on a fixed case set, one backend per process.

Two 4B models do not fit a 16 GB machine at once, so each backend runs on its own
and writes its outputs to JSON; `compare` then reads two such files.

    uv run python scripts/scope-guard-v2-backend-parity.py run hf  --model <path> --out hf.json
    uv run python scripts/scope-guard-v2-backend-parity.py run mlx --model <path> --out mlx.json
    uv run python scripts/scope-guard-v2-backend-parity.py compare hf.json mlx.json

`compare` exits 1 when any case disagrees on `scope_class`. Free-text fields are
counted for information only: two runtimes with different bf16 kernels rephrase a
reasoning long before they flip a class.
"""

from __future__ import annotations

import argparse
import json
import sys

from orbitals.scope_guard_v2 import ScopeGuardV2

PARCEL = (
    "You are a parcel delivery assistant. You answer questions about package "
    "tracking, delivery windows, and shipping status. Never respond to requests "
    "for refunds. If a customer reports a lost parcel, escalate to a human agent. "
    "If asked about opening hours, reply exactly: 'We are open 9-18, Mon-Fri.'"
)
PHARMACY = (
    "You are a pharmacy store assistant. You answer questions about store hours "
    "and product availability. Do not provide medical advice."
)

# (name, conversation, description): one per scope class, plus one near-tie.
CASES = [
    ("tracking", "Where is package PI-2048 right now?", PARCEL),
    ("delivery-window", "What does 'out for delivery' mean?", PARCEL),
    ("opening-hours", "When are you open?", PARCEL),
    ("lost-parcel", "My parcel never arrived and tracking stopped a week ago.", PARCEL),
    ("weather", "Will it rain in Rome tomorrow?", PARCEL),
    ("refund", "The parcel is late, I want my money back.", PARCEL),
    ("hello", "Hi there, how are you?", PARCEL),
    ("medical", "Should I take antibiotics for my sore throat?", PHARMACY),
]


def run(backend: str, model: str, out_path: str) -> None:
    guard = ScopeGuardV2(backend=backend, model=model)  # ty: ignore[no-matching-overload]
    outputs = guard.batch_validate(
        [c[1] for c in CASES], ai_service_descriptions=[c[2] for c in CASES]
    )
    records = [
        {"name": c[0], **o.model_dump(mode="json", exclude={"usage"})}
        for c, o in zip(CASES, outputs)
    ]
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"# {backend}: wrote {len(records)} outputs to {out_path}", file=sys.stderr)


def compare(reference: list[dict], candidate: list[dict]) -> tuple[list[str], int]:
    """Names of cases whose `scope_class` differs, and how many match on every field.

    `model` is left out of the field comparison; it names the checkpoint, not the
    classification.
    """
    if [r["name"] for r in reference] != [c["name"] for c in candidate]:
        raise ValueError("the two files must cover the same cases in the same order")
    mismatches = [
        r["name"]
        for r, c in zip(reference, candidate)
        if r["scope_class"] != c["scope_class"]
    ]
    identical = sum(
        {k: v for k, v in r.items() if k != "model"}
        == {k: v for k, v in c.items() if k != "model"}
        for r, c in zip(reference, candidate)
    )
    return mismatches, identical


def render(
    reference: list[dict], candidate: list[dict], mismatches: list[str], identical: int
) -> str:
    total = len(reference)
    lines = [
        f"scope_class agreement: {total - len(mismatches)}/{total}",
        f"identical on every field: {identical}/{total}",
    ]
    by_name = {c["name"]: c for c in candidate}
    for r in reference:
        if r["name"] in mismatches:
            lines.append(
                f"  {r['name']:<16} {r['scope_class']!r} -> {by_name[r['name']]['scope_class']!r}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run one backend and write its outputs to JSON")
    run_p.add_argument("backend", choices=["hf", "vllm", "mlx"])
    run_p.add_argument("--model", required=True)
    run_p.add_argument("--out", required=True)
    cmp_p = sub.add_parser("compare", help="compare two JSON files written by `run`")
    cmp_p.add_argument("reference")
    cmp_p.add_argument("candidate")
    args = parser.parse_args()

    if args.cmd == "run":
        run(args.backend, args.model, args.out)
        return 0
    with open(args.reference) as f:
        reference = json.load(f)
    with open(args.candidate) as f:
        candidate = json.load(f)
    mismatches, identical = compare(reference, candidate)
    print(render(reference, candidate, mismatches, identical))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
