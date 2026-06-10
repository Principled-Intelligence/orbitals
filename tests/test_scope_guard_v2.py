from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from orbitals.types import AIServiceDescriptionV2, LLMUsage


def _response_payload(**overrides) -> dict[str, Any]:
    return {
        "scope_class": overrides.get("scope_class", "Restricted"),
        "evidences": overrides.get("evidences", ["Do not provide refunds."]),
        "reasoning": overrides.get("reasoning", "The request matches a constraint."),
        "suggested_response": overrides.get(
            "suggested_response", "I cannot help with that request."
        ),
        "model": overrides.get("model", "test-scope-guard-v2-model"),
        "usage": overrides.get(
            "usage",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        ),
    }


def test_scope_guard_v2_public_imports():
    from orbitals.scope_guard_v2 import (
        AsyncScopeGuardV2,
        ScopeClass,
        ScopeGuardV2,
        ScopeGuardV2Output,
    )

    assert ScopeGuardV2 is not None
    assert AsyncScopeGuardV2 is not None
    assert ScopeClass is not None
    assert ScopeGuardV2Output is not None


def test_scope_guard_v2_package_all_is_exhaustive():
    import orbitals.scope_guard_v2 as sg

    assert set(sg.__all__) == {
        "ADDITIONAL_SAFETY_RULES",
        "AsyncScopeGuardV2",
        "ScopeClass",
        "ScopeGuardV2",
        "ScopeGuardV2Output",
        "augment_with_default_safety_principles_v2",
    }


def test_scope_guard_v2_scope_classes():
    from orbitals.scope_guard_v2 import ScopeClass

    assert {m.value for m in ScopeClass} == {
        "Directly Supported",
        "Potentially Supported",
        "Predefined Answer",
        "Human Oversight",
        "Out of Scope",
        "Restricted",
        "Chit Chat",
    }
    assert ScopeClass.RESTRICTED == "Restricted"
    assert ScopeClass.HUMAN_OVERSIGHT.value == "Human Oversight"


def test_scope_guard_v2_input_accepts_documented_shapes():
    from orbitals.scope_guard_v2.modeling import (
        ConversationUserMessage,
        ScopeGuardV2InputTypeAdapter,
    )
    from orbitals.types import ConversationMessage

    assert ScopeGuardV2InputTypeAdapter.validate_python("hello") == "hello"

    single = ScopeGuardV2InputTypeAdapter.validate_python(
        {"role": "user", "content": "hello"}
    )
    assert isinstance(single, ConversationUserMessage)

    multi = ScopeGuardV2InputTypeAdapter.validate_python(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
    )
    assert all(isinstance(message, ConversationMessage) for message in multi)


def test_scope_guard_v2_rejects_single_assistant_message():
    from orbitals.scope_guard_v2.modeling import ScopeGuardV2InputTypeAdapter

    with pytest.raises(ValidationError):
        ScopeGuardV2InputTypeAdapter.validate_python(
            {"role": "assistant", "content": "hello"}
        )


def test_scope_guard_v2_prompt_requires_last_user_message():
    from orbitals.scope_guard_v2.prompting import dumps_conversation
    from orbitals.types import Conversation, ConversationMessage

    with pytest.raises(ValueError, match="last message"):
        dumps_conversation(
            Conversation(
                messages=[
                    ConversationMessage(role="user", content="q"),
                    ConversationMessage(role="assistant", content="a"),
                ]
            )
        )


def test_scope_guard_v2_prompt_treats_string_as_user_message():
    from orbitals.scope_guard_v2.prompting import dumps_conversation

    assert "LAST MESSAGE (USER):\nhello" in dumps_conversation("hello")


def test_ai_service_description_v2_forbids_extra_fields():
    with pytest.raises(ValidationError):
        AIServiceDescriptionV2(
            identity_role="Role",
            context="Context",
            unknown_field="nope",  # type: ignore[call-arg]
        )


def test_scope_guard_v2_default_safety_principles_add_constraints():
    from orbitals.scope_guard_v2 import (
        ADDITIONAL_SAFETY_RULES,
        augment_with_default_safety_principles_v2,
    )

    desc = AIServiceDescriptionV2(identity_role="Role", context="Context")
    out = augment_with_default_safety_principles_v2(desc)

    assert isinstance(out, AIServiceDescriptionV2)
    assert out.constraints == [ADDITIONAL_SAFETY_RULES]
    assert desc.constraints is None


def test_scope_guard_v2_api_backend_augments_safety_constraints(mocked_v2_post):
    from orbitals.scope_guard_v2 import ADDITIONAL_SAFETY_RULES, ScopeGuardV2

    sg = ScopeGuardV2(
        backend="api",
        api_url="http://example.com",
        include_default_safety_principles=True,
    )
    sg.validate(
        "hello",
        ai_service_description=AIServiceDescriptionV2(
            identity_role="Role",
            context="Context",
            constraints=["No refunds"],
        ),
    )

    constraints = mocked_v2_post.call_args.kwargs["json"]["ai_service_description"][
        "constraints"
    ]
    assert constraints == ["No refunds", ADDITIONAL_SAFETY_RULES]


@pytest.fixture
def mocked_v2_post():
    with patch("orbitals.scope_guard_v2.guards.api.requests.post") as mocked:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = _response_payload()
        mocked.return_value = response
        yield mocked


def test_scope_guard_v2_api_backend_hits_v2_endpoint(mocked_v2_post):
    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2, ScopeGuardV2Output

    sg = ScopeGuardV2(backend="api", api_url="http://example.com")
    result = sg.validate("hello", ai_service_description="desc")

    assert mocked_v2_post.call_args.args[0] == (
        "http://example.com/orbitals/scope-guard-v2/validate"
    )
    assert isinstance(result, ScopeGuardV2Output)
    assert result.scope_class == ScopeClass.RESTRICTED
    assert result.reasoning == "The request matches a constraint."
    assert result.suggested_response == "I cannot help with that request."


def test_scope_guard_v2_api_backend_serializes_structured_description(
    mocked_v2_post,
):
    from orbitals.scope_guard_v2 import ScopeGuardV2

    sg = ScopeGuardV2(backend="api", api_url="http://example.com", model="v2-model")
    sg.validate(
        {"role": "user", "content": "hello"},
        ai_service_description=AIServiceDescriptionV2(
            identity_role="Role",
            context="Context",
            constraints=["No refunds"],
        ),
    )

    body = mocked_v2_post.call_args.kwargs["json"]
    assert body["model"] == "v2-model"
    assert body["conversation"] == {"role": "user", "content": "hello"}
    assert body["ai_service_description"]["constraints"] == ["No refunds"]


def test_scope_guard_v2_batch_validate_invariants():
    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2, ScopeGuardV2Output
    from orbitals.scope_guard_v2.guards.base import BaseScopeGuardV2

    class _StubScopeGuardV2(ScopeGuardV2):
        def __new__(cls, *args, **kwargs):
            return BaseScopeGuardV2.__new__(cls)

        def __init__(self):
            self.backend = "stub"

        def _batch_validate(self, conversations, **kwargs):
            return [
                ScopeGuardV2Output(
                    evidences=None,
                    reasoning="ok",
                    scope_class=ScopeClass.DIRECTLY_SUPPORTED,
                    suggested_response=None,
                    model="stub",
                    usage=None,
                )
                for _ in conversations
            ]

    guard = _StubScopeGuardV2()
    assert guard.batch_validate([], ai_service_description="desc") == []
    with pytest.raises(ValueError, match="Either ai_service_description"):
        guard.batch_validate(["q"])
    with pytest.raises(ValueError, match="Only one between"):
        guard.batch_validate(
            ["q"],
            ai_service_description="desc",
            ai_service_descriptions=["desc"],
        )
    with pytest.raises(ValueError, match="number of conversations"):
        guard.batch_validate(["q1", "q2"], ai_service_descriptions=["d1"])
    assert len(guard.batch_validate(["q1", "q2"], ai_service_description="d")) == 2


class _FakeAiohttpResponse:
    def __init__(self, payload: Any):
        self._payload = payload

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _FakeAiohttpSession:
    def __init__(self, payload: Any, captured: dict[str, Any]):
        self._response = _FakeAiohttpResponse(payload)
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def post(self, url, *, json, headers):
        self._captured["url"] = url
        self._captured["json"] = json
        self._captured["headers"] = headers
        return self._response


async def test_scope_guard_v2_async_api_backend(monkeypatch):
    from orbitals.scope_guard_v2 import AsyncScopeGuardV2, ScopeClass

    captured: dict[str, Any] = {}

    def _session_factory():
        return _FakeAiohttpSession(_response_payload(), captured)

    monkeypatch.setattr(
        "orbitals.scope_guard_v2.guards.api.aiohttp.ClientSession",
        _session_factory,
    )

    sg = AsyncScopeGuardV2(
        backend="api",
        api_url="http://example.com",
        api_key="secret",
    )
    result = await sg.validate("hello", ai_service_description="desc")

    assert captured["url"] == "http://example.com/orbitals/scope-guard-v2/validate"
    assert captured["headers"]["X-API-Key"] == "secret"
    assert result.scope_class == ScopeClass.RESTRICTED


@pytest.fixture
def scope_guard_v2_serving_client(monkeypatch):
    monkeypatch.setenv("SCOPE_GUARD_V2_VLLM_MODEL", "v2-model")
    monkeypatch.setenv("SCOPE_GUARD_V2_VLLM_SERVING_URL", "http://localhost:8001")
    monkeypatch.setenv("SCOPE_GUARD_V2_SKIP_EVIDENCES", "0")

    from fastapi.testclient import TestClient

    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2Output
    from orbitals.scope_guard_v2.serving import main as serving_main

    class _StubAsyncGuard:
        async def validate(self, conversation, *, ai_service_description, **kwargs):
            return ScopeGuardV2Output(
                scope_class=ScopeClass.HUMAN_OVERSIGHT,
                evidences=["Escalate billing disputes."],
                reasoning="The request meets an escalation criterion.",
                suggested_response="A human specialist will review this.",
                model="stub-model",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        async def batch_validate(
            self,
            conversations,
            *,
            ai_service_description=None,
            ai_service_descriptions=None,
            **kwargs,
        ):
            return [
                ScopeGuardV2Output(
                    scope_class=ScopeClass.DIRECTLY_SUPPORTED,
                    evidences=None,
                    reasoning="Supported.",
                    suggested_response=None,
                    model="stub-model",
                    usage=LLMUsage(
                        prompt_tokens=1, completion_tokens=1, total_tokens=2
                    ),
                )
                for _ in conversations
            ]

    with TestClient(serving_main.app) as client:
        monkeypatch.setattr(serving_main, "scope_guard", _StubAsyncGuard())
        yield client


def test_scope_guard_v2_serving_validate(scope_guard_v2_serving_client):
    response = scope_guard_v2_serving_client.post(
        "/orbitals/scope-guard-v2/validate",
        json={"conversation": "hello", "ai_service_description": "desc"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scope_class"] == "Human Oversight"
    assert body["reasoning"] == "The request meets an escalation criterion."
    assert body["suggested_response"] == "A human specialist will review this."


def test_scope_guard_v2_serving_batch_validate(scope_guard_v2_serving_client):
    response = scope_guard_v2_serving_client.post(
        "/orbitals/scope-guard-v2/batch-validate",
        json={"conversations": ["q1", "q2"], "ai_service_description": "desc"},
    )

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_scope_guard_v2_cli_help_and_model_passthrough():
    from orbitals.cli.main import app

    runner = CliRunner()
    help_result = runner.invoke(app, ["scope-guard-v2", "serve", "--help"])
    assert help_result.exit_code == 0
    assert "--port" in help_result.stdout

    model_result = runner.invoke(
        app,
        ["scope-guard-v2", "convert-default-model-name", "scope-guard-q"],
    )
    assert model_result.exit_code == 0
    assert "scope-guard-q" in model_result.stdout
