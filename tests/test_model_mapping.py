"""Tests for the default-model alias resolution used throughout the README.

The README uses short aliases like `model="scope-guard-q"` and
`model="scope-guard-g"`. These are mapped to concrete Hugging Face
repository IDs by `ScopeGuard.maybe_map_model`.
"""

from __future__ import annotations

from orbitals.scope_guard import ScopeGuard
from orbitals.scope_guard.guards.base import MODEL_MAPPING


def test_default_alias_resolves_to_qwen_model():
    assert (
        ScopeGuard.maybe_map_model("scope-guard")
        == "principled-intelligence/scope-guard-4B-q-2601"
    )


def test_qwen_alias_resolves_to_qwen_model():
    assert (
        ScopeGuard.maybe_map_model("scope-guard-q")
        == "principled-intelligence/scope-guard-4B-q-2601"
    )


def test_gemma_alias_resolves_to_gemma_model():
    assert (
        ScopeGuard.maybe_map_model("scope-guard-g")
        == "principled-intelligence/scope-guard-4B-g-2601"
    )


def test_unknown_model_name_is_passed_through_unchanged():
    custom_name = "my-org/my-custom-model"
    assert ScopeGuard.maybe_map_model(custom_name) == custom_name


def test_model_mapping_contains_all_documented_aliases():
    assert "scope-guard" in MODEL_MAPPING
    assert "scope-guard-q" in MODEL_MAPPING
    assert "scope-guard-g" in MODEL_MAPPING
