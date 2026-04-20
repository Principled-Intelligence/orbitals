"""Tests for the shared batch-validation invariants in BaseScopeGuard.

The README documents two calling styles for `batch_validate`:
  - With a single `ai_service_description` applied to every conversation.
  - With a per-conversation `ai_service_descriptions` list.

Exactly one of the two must be provided, and their lengths must line up with
`conversations`. Using a mock backend lets us exercise this shared logic
without loading any model.
"""

from __future__ import annotations

from typing import Any

import pytest

from orbitals.scope_guard import ScopeClass, ScopeGuard, ScopeGuardOutput
from orbitals.scope_guard.guards.base import BaseScopeGuard


class _StubScopeGuard(ScopeGuard):
    """A minimal ScopeGuard subclass for exercising shared validation logic."""

    def __new__(cls, *args, **kwargs):
        return BaseScopeGuard.__new__(cls)  # bypass registry dispatch

    def __init__(self) -> None:
        self.backend = "stub"

    def _batch_validate(
        self,
        conversations,
        *,
        ai_service_description=None,
        ai_service_descriptions=None,
        skip_evidences=None,
        **kwargs: Any,
    ) -> list[ScopeGuardOutput]:
        return [
            ScopeGuardOutput(
                evidences=None,
                scope_class=ScopeClass.DIRECTLY_SUPPORTED,
                model="stub",
                usage=None,
            )
            for _ in conversations
        ]


def _make_guard() -> _StubScopeGuard:
    return _StubScopeGuard()


def test_empty_conversations_returns_empty_without_raising():
    guard = _make_guard()
    out = guard.batch_validate([], ai_service_description="desc")
    assert out == []


def test_providing_neither_description_raises():
    guard = _make_guard()
    with pytest.raises(ValueError, match="Either ai_service_description"):
        guard.batch_validate(["q"])


def test_providing_both_descriptions_raises():
    guard = _make_guard()
    with pytest.raises(ValueError, match="Only one between"):
        guard.batch_validate(
            ["q"],
            ai_service_description="desc",
            ai_service_descriptions=["desc"],
        )


def test_descriptions_length_mismatch_raises():
    guard = _make_guard()
    with pytest.raises(ValueError, match="number of conversations"):
        guard.batch_validate(
            ["q1", "q2", "q3"],
            ai_service_descriptions=["d1", "d2"],
        )


def test_single_description_broadcasts_to_all_conversations():
    guard = _make_guard()
    out = guard.batch_validate(
        ["q1", "q2", "q3"],
        ai_service_description="shared-desc",
    )
    assert len(out) == 3
    assert all(o.scope_class == ScopeClass.DIRECTLY_SUPPORTED for o in out)


def test_per_conversation_descriptions_accepted_when_lengths_match():
    guard = _make_guard()
    out = guard.batch_validate(
        ["q1", "q2"],
        ai_service_descriptions=["d1", "d2"],
    )
    assert len(out) == 2
