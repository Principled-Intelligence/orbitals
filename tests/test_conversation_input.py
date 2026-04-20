"""Tests for the flexible conversation input formats documented in the README.

The `validate` method accepts three conversation shapes:
  1. A plain user query string
  2. A single user message dict (OpenAI-style)
  3. A list of message dicts (multi-turn conversation)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orbitals.scope_guard.modeling import (
    ConversationUserMessage,
    ScopeGuardInputTypeAdapter,
)
from orbitals.types import ConversationMessage


def test_string_input_is_accepted():
    result = ScopeGuardInputTypeAdapter.validate_python(
        "When is my package scheduled to arrive?"
    )
    assert result == "When is my package scheduled to arrive?"
    assert isinstance(result, str)


def test_dict_input_with_user_role_is_accepted():
    result = ScopeGuardInputTypeAdapter.validate_python(
        {"role": "user", "content": "When is my package scheduled to arrive?"}
    )
    assert isinstance(result, ConversationUserMessage)
    assert result.role == "user"
    assert result.content == "When is my package scheduled to arrive?"


def test_dict_input_with_assistant_role_alone_is_rejected():
    # The README specifies that a single-message dict must be a user message.
    with pytest.raises(ValidationError):
        ScopeGuardInputTypeAdapter.validate_python(
            {"role": "assistant", "content": "Hi!"}
        )


def test_list_input_multi_turn_conversation_is_accepted():
    result = ScopeGuardInputTypeAdapter.validate_python(
        [
            {"role": "user", "content": "I ordered a package, tracking 1234"},
            {"role": "assistant", "content": "Great, it is in transit."},
            {"role": "user", "content": "If it does not arrive, can I refund?"},
        ]
    )
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(m, ConversationMessage) for m in result)
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[2].role == "user"


def test_list_input_with_unknown_role_is_rejected():
    with pytest.raises(ValidationError):
        ScopeGuardInputTypeAdapter.validate_python(
            [{"role": "system", "content": "You are a helpful assistant."}]
        )


def test_dict_input_missing_content_is_rejected():
    with pytest.raises(ValidationError):
        ScopeGuardInputTypeAdapter.validate_python({"role": "user"})


def test_dict_input_missing_role_is_rejected():
    with pytest.raises(ValidationError):
        ScopeGuardInputTypeAdapter.validate_python({"content": "hello"})


def test_empty_list_is_accepted():
    # A list of messages can validly be empty (no messages).
    result = ScopeGuardInputTypeAdapter.validate_python([])
    assert result == []
