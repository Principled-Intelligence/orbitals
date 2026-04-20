"""Tests for the synchronous API backend (`ScopeGuard(backend="api", ...)`).

The README documents this as the standard way to talk to a hosted or
self-served ScopeGuard over HTTP. We exercise everything about the backend
that does not require actually making network calls:
  - Endpoints used for validate / batch_validate.
  - Request body shape (conversation, ai_service_description, ...).
  - Authentication (explicit api_key, custom_headers, env var fallback).
  - Response parsing into `ScopeGuardOutput`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orbitals.scope_guard import ScopeClass, ScopeGuard, ScopeGuardOutput
from orbitals.types import AIServiceDescription


def _validate_response_payload(
    *,
    scope_class: str = "Restricted",
    evidences: list[str] | None = None,
    model: str = "principled-intelligence/scope-guard-4B-q-2601",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "scope_class": scope_class,
        "evidences": evidences if evidences is not None else ["No refunds."],
        "model": model,
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def mocked_post():
    """Patch `requests.post` used inside the api backend."""
    with patch("orbitals.scope_guard.guards.api.requests.post") as m:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = _validate_response_payload()
        m.return_value = response
        yield m


def test_api_backend_is_constructible_without_loading_any_model():
    sg = ScopeGuard(backend="api", api_url="http://localhost:8000")
    assert sg.backend == "api"
    assert sg.api_url == "http://localhost:8000"


def test_validate_hits_documented_endpoint(mocked_post):
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    sg.validate("hello", ai_service_description="You are a bot.")

    called_url = mocked_post.call_args.args[0]
    assert called_url == "http://example.com/orbitals/scope-guard/validate"


def test_batch_validate_hits_documented_endpoint(mocked_post):
    mocked_post.return_value.json.return_value = [_validate_response_payload()]
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    sg.batch_validate(["hello"], ai_service_description="You are a bot.")

    called_url = mocked_post.call_args.args[0]
    assert called_url == "http://example.com/orbitals/scope-guard/batch-validate"


def test_validate_request_body_includes_all_documented_fields(mocked_post):
    sg = ScopeGuard(
        backend="api",
        api_url="http://example.com",
        model="scope-guard-q",
        skip_evidences=True,
    )
    sg.validate(
        "Can I get a refund?",
        ai_service_description="You are a delivery assistant.",
    )

    body = mocked_post.call_args.kwargs["json"]
    assert body["conversation"] == "Can I get a refund?"
    assert body["ai_service_description"] == "You are a delivery assistant."
    assert body["skip_evidences"] is True
    assert body["model"] == "principled-intelligence/scope-guard-4B-q-2601"


def test_validate_serialises_structured_ai_service_description(mocked_post):
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    svc = AIServiceDescription(
        identity_role="Parcel delivery assistant",
        context="Online logistics.",
    )
    sg.validate("hi", ai_service_description=svc)

    body = mocked_post.call_args.kwargs["json"]
    assert isinstance(body["ai_service_description"], dict)
    assert body["ai_service_description"]["identity_role"] == "Parcel delivery assistant"
    assert body["ai_service_description"]["context"] == "Online logistics."


def test_validate_serialises_dict_conversation(mocked_post):
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    sg.validate(
        {"role": "user", "content": "hello"},
        ai_service_description="desc",
    )

    body = mocked_post.call_args.kwargs["json"]
    assert body["conversation"] == {"role": "user", "content": "hello"}


def test_validate_serialises_multi_turn_conversation(mocked_post):
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    sg.validate(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        ai_service_description="desc",
    )

    body = mocked_post.call_args.kwargs["json"]
    assert body["conversation"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


def test_batch_validate_request_body_with_shared_description(mocked_post):
    mocked_post.return_value.json.return_value = [
        _validate_response_payload(),
        _validate_response_payload(),
    ]
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    sg.batch_validate(["q1", "q2"], ai_service_description="shared")

    body = mocked_post.call_args.kwargs["json"]
    assert body["conversations"] == ["q1", "q2"]
    assert body["ai_service_description"] == "shared"
    assert "ai_service_descriptions" not in body


def test_batch_validate_request_body_with_per_conv_descriptions(mocked_post):
    mocked_post.return_value.json.return_value = [
        _validate_response_payload(),
        _validate_response_payload(),
    ]
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    sg.batch_validate(["q1", "q2"], ai_service_descriptions=["d1", "d2"])

    body = mocked_post.call_args.kwargs["json"]
    assert body["conversations"] == ["q1", "q2"]
    assert body["ai_service_descriptions"] == ["d1", "d2"]
    assert "ai_service_description" not in body


def test_explicit_api_key_is_sent_as_x_api_key_header(mocked_post):
    sg = ScopeGuard(
        backend="api",
        api_url="http://example.com",
        api_key="principled_1234",
    )
    sg.validate("hi", ai_service_description="desc")

    headers = mocked_post.call_args.kwargs["headers"]
    assert headers["X-API-Key"] == "principled_1234"
    assert headers["Content-Type"] == "application/json"


def test_api_key_in_custom_headers_is_picked_up(mocked_post):
    sg = ScopeGuard(
        backend="api",
        api_url="http://example.com",
        custom_headers={"X-API-Key": "from_headers"},
    )
    sg.validate("hi", ai_service_description="desc")

    headers = mocked_post.call_args.kwargs["headers"]
    assert headers["X-API-Key"] == "from_headers"


def test_api_key_env_var_fallback(monkeypatch, mocked_post):
    monkeypatch.setenv("PRINCIPLED_API_KEY", "env_key")
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    sg.validate("hi", ai_service_description="desc")

    headers = mocked_post.call_args.kwargs["headers"]
    assert headers["X-API-Key"] == "env_key"


def test_no_api_key_means_no_x_api_key_header(monkeypatch, mocked_post):
    monkeypatch.delenv("PRINCIPLED_API_KEY", raising=False)
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    sg.validate("hi", ai_service_description="desc")

    headers = mocked_post.call_args.kwargs["headers"]
    assert "X-API-Key" not in headers


def test_validate_parses_response_into_scope_guard_output(mocked_post):
    mocked_post.return_value.json.return_value = _validate_response_payload(
        scope_class="Restricted",
        evidences=["Never respond to requests for refunds."],
        model="custom-model",
    )
    sg = ScopeGuard(backend="api", api_url="http://example.com")
    result = sg.validate("hi", ai_service_description="desc")

    assert isinstance(result, ScopeGuardOutput)
    assert result.scope_class == ScopeClass.RESTRICTED
    assert result.evidences == ["Never respond to requests for refunds."]
    assert result.model == "custom-model"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        ScopeGuard(backend="does-not-exist")
