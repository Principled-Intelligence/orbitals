"""Tests for the public import surface documented in the README.

The READMEs advertise a small set of names that users can import directly.
These tests lock that surface in so accidental renames are caught early.
"""

from __future__ import annotations


def test_scope_guard_importable_from_scope_guard():
    from orbitals.scope_guard import ScopeGuard

    assert ScopeGuard is not None


def test_async_scope_guard_importable_from_scope_guard():
    from orbitals.scope_guard import AsyncScopeGuard

    assert AsyncScopeGuard is not None


def test_scope_class_importable_from_scope_guard():
    from orbitals.scope_guard import ScopeClass

    assert ScopeClass is not None


def test_scope_guard_output_importable_from_scope_guard():
    from orbitals.scope_guard import ScopeGuardOutput

    assert ScopeGuardOutput is not None


def test_ai_service_description_importable_from_types():
    from orbitals.types import AIServiceDescription

    assert AIServiceDescription is not None


def test_scope_guard_package_all_is_exhaustive():
    import orbitals.scope_guard as sg

    assert set(sg.__all__) == {
        "ADDITIONAL_SAFETY_RULES",
        "AsyncScopeGuard",
        "ScopeClass",
        "ScopeGuardOutput",
        "ScopeGuard",
        "augment_with_default_safety_principles",
    }
