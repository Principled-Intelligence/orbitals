"""Tests for ScopeClass values documented in the README.

From `README.scope-guard.md`:

    print(ScopeClass.DIRECTLY_SUPPORTED.value)    # "Directly Supported"
    print(ScopeClass.POTENTIALLY_SUPPORTED.value) # "Potentially Supported"
    print(ScopeClass.OUT_OF_SCOPE.value)          # "Out of Scope"
    print(ScopeClass.RESTRICTED.value)            # "Restricted"
    print(ScopeClass.CHIT_CHAT.value)             # "Chit Chat"
"""

from __future__ import annotations

from orbitals.scope_guard import ScopeClass


def test_directly_supported_value():
    assert ScopeClass.DIRECTLY_SUPPORTED.value == "Directly Supported"


def test_potentially_supported_value():
    assert ScopeClass.POTENTIALLY_SUPPORTED.value == "Potentially Supported"


def test_out_of_scope_value():
    assert ScopeClass.OUT_OF_SCOPE.value == "Out of Scope"


def test_restricted_value():
    assert ScopeClass.RESTRICTED.value == "Restricted"


def test_chit_chat_value():
    assert ScopeClass.CHIT_CHAT.value == "Chit Chat"


def test_scope_class_has_exactly_five_members():
    assert {m.value for m in ScopeClass} == {
        "Directly Supported",
        "Potentially Supported",
        "Out of Scope",
        "Restricted",
        "Chit Chat",
    }


def test_scope_class_is_string_enum():
    assert ScopeClass.RESTRICTED == "Restricted"


def test_scope_class_comparison_by_enum_member():
    cls = ScopeClass("Restricted")
    assert cls == ScopeClass.RESTRICTED


def test_scope_class_comparison_by_string_value():
    cls = ScopeClass.RESTRICTED
    assert cls.value == "Restricted"
