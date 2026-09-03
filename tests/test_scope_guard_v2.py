from __future__ import annotations

import json
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


# --- output fields: selector helpers and the pinned prompt -----------------


def test_scope_guard_v2_system_prompt_is_byte_identical_to_training():
    """The client prompt must be the bytes the 2608 models trained on.

    A diverged system prompt degrades accuracy with no parse error to show for it,
    so drift has to be a red test. The hash is of the trainer's
    src/prompting.SYSTEM_PROMPT, trailing newline included (the chat template puts
    <|im_end|> right after the content, so the newline is part of what the model
    saw). Do NOT change this to 1f37419f...: that is the newline-stripped copy
    unsafe-eval and the Hub system_prompt.txt carry.
    """
    import hashlib

    from orbitals.scope_guard_v2.prompting import SYSTEM_PROMPT

    assert len(SYSTEM_PROMPT) == 7665
    assert SYSTEM_PROMPT.endswith("\n")
    assert (
        hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        == "f0f68e48096938b93c2c497386295ff7db2929b128e9dddea944ebb6578960b5"
    )


def test_scope_guard_v2_system_prompt_has_no_embedded_schema_or_skip_marker():
    from orbitals.scope_guard_v2.prompting import SELECTOR_HEADER, SYSTEM_PROMPT

    assert '"$defs"' not in SYSTEM_PROMPT
    assert "SKIP EVIDENCES" not in SYSTEM_PROMPT
    assert SELECTOR_HEADER in SYSTEM_PROMPT
    assert "exactly those keys" in SYSTEM_PROMPT


def test_normalize_selection_forces_scope_class_dedupes_and_reorders():
    from orbitals.scope_guard_v2.prompting import ALL_FIELDS, normalize_selection

    assert normalize_selection([]) == ("scope_class",)
    assert normalize_selection(["scope_class", "reasoning"]) == (
        "reasoning",
        "scope_class",
    )
    assert normalize_selection(["reasoning", "reasoning", "evidences"]) == (
        "evidences",
        "reasoning",
        "scope_class",
    )
    assert normalize_selection(reversed(ALL_FIELDS)) == ALL_FIELDS


def test_normalize_selection_rejects_unknown_fields():
    from orbitals.scope_guard_v2.prompting import normalize_selection

    with pytest.raises(ValueError, match="unknown output field"):
        normalize_selection(["scope_class", "confidence"])


def test_render_selector_block_matches_the_trainer_for_all_eight_selections():
    """Eight literal strings copied from the trainer's render_selector_block.

    The user turn must be byte-identical to training, and this block is the part
    of it the client composes itself.
    """
    from orbitals.scope_guard_v2.prompting import render_selector_block

    expected = {
        ("scope_class",): '**REQUESTED OUTPUT FIELDS**\n\n["scope_class"]',
        ("evidences", "scope_class"): '**REQUESTED OUTPUT FIELDS**\n\n["evidences", "scope_class"]',
        ("reasoning", "scope_class"): '**REQUESTED OUTPUT FIELDS**\n\n["reasoning", "scope_class"]',
        ("evidences", "reasoning", "scope_class"): '**REQUESTED OUTPUT FIELDS**\n\n["evidences", "reasoning", "scope_class"]',
        ("scope_class", "suggested_response"): '**REQUESTED OUTPUT FIELDS**\n\n["scope_class", "suggested_response"]',
        ("evidences", "scope_class", "suggested_response"): '**REQUESTED OUTPUT FIELDS**\n\n["evidences", "scope_class", "suggested_response"]',
        ("reasoning", "scope_class", "suggested_response"): '**REQUESTED OUTPUT FIELDS**\n\n["reasoning", "scope_class", "suggested_response"]',
        ("evidences", "reasoning", "scope_class", "suggested_response"): '**REQUESTED OUTPUT FIELDS**\n\n["evidences", "reasoning", "scope_class", "suggested_response"]',
    }
    for selection, block in expected.items():
        assert render_selector_block(selection) == block, selection
    # order-insensitive on input
    assert render_selector_block(["suggested_response", "scope_class", "evidences"]) == (
        expected[("evidences", "scope_class", "suggested_response")]
    )


def test_potentially_supported_description_is_the_promptfix_text():
    from orbitals.scope_guard_v2 import ScopeClass

    desc = ScopeClass.POTENTIALLY_SUPPORTED.description
    assert desc.startswith("The query is adjacent to the service's stated functionalities")
    assert "never as a way to avoid committing to a clearer class" in desc


# --- per-selection response model -----------------------------------------


def test_response_model_for_requires_exactly_the_selected_keys():
    from orbitals.scope_guard_v2.prompting import response_model_for

    model = response_model_for(["scope_class"])
    parsed = model.model_validate({"scope_class": "Restricted"})
    assert parsed.scope_class == "Restricted"

    with pytest.raises(ValidationError):  # unrequested key is forbidden
        model.model_validate({"scope_class": "Restricted", "reasoning": "because"})

    full = response_model_for(["evidences", "reasoning", "scope_class", "suggested_response"])
    with pytest.raises(ValidationError):  # requested key is required, even if nullable
        full.model_validate({"reasoning": "r", "scope_class": "Restricted", "suggested_response": None})
    ok = full.model_validate(
        {"evidences": None, "reasoning": "r", "scope_class": "Restricted", "suggested_response": None}
    )
    assert ok.evidences is None


def test_response_model_for_is_cached_per_normalized_selection():
    from orbitals.scope_guard_v2.prompting import response_model_for

    a = response_model_for(["reasoning", "scope_class"])
    b = response_model_for(["scope_class", "reasoning", "reasoning"])
    assert a is b


def test_response_model_for_schema_lists_only_selected_properties():
    from orbitals.scope_guard_v2.prompting import response_model_for

    schema = response_model_for(["reasoning", "scope_class"]).model_json_schema()
    assert list(schema["properties"]) == ["reasoning", "scope_class"]
    assert set(schema["required"]) == {"reasoning", "scope_class"}
    assert schema.get("additionalProperties") is False


def test_scope_guard_v2_output_reasoning_is_optional():
    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2Output

    out = ScopeGuardV2Output(scope_class=ScopeClass.CHIT_CHAT, model="m")
    assert out.reasoning is None
    assert out.evidences is None
    assert out.suggested_response is None


# --- user turn and selection resolution -------------------------------------


def _user_turn(**kwargs) -> str:
    from orbitals.scope_guard_v2.prompting import prepare_input_messages

    messages = prepare_input_messages("hello", "desc", **kwargs)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    return messages[1]["content"]


def test_prepare_input_messages_default_requests_all_four_fields():
    turn = _user_turn()
    assert turn.endswith(
        '**END OF THE CONVERSATION DUMP**\n\n\n**REQUESTED OUTPUT FIELDS**\n\n'
        '["evidences", "reasoning", "scope_class", "suggested_response"]'
    )
    assert "SKIP EVIDENCES" not in turn


def test_prepare_input_messages_skip_evidences_drops_only_evidences():
    turn = _user_turn(skip_evidences=True)
    assert turn.endswith('["reasoning", "scope_class", "suggested_response"]')
    assert _user_turn(skip_evidences=False) == _user_turn()


def test_prepare_input_messages_output_fields_selects_the_keys():
    turn = _user_turn(output_fields=["scope_class"])
    assert turn.endswith('**REQUESTED OUTPUT FIELDS**\n\n["scope_class"]')


def test_prepare_input_messages_conflicting_flags_warn_and_output_fields_wins():
    with pytest.warns(DeprecationWarning, match="output_fields"):
        turn = _user_turn(output_fields=["evidences", "scope_class"], skip_evidences=True)
    assert turn.endswith('["evidences", "scope_class"]')


def test_prepare_input_messages_agreeing_flags_do_not_warn(recwarn):
    # skip_evidences=True implies exactly these three, so the pair agrees
    turn = _user_turn(
        output_fields=["reasoning", "scope_class", "suggested_response"], skip_evidences=True
    )
    assert turn.endswith('["reasoning", "scope_class", "suggested_response"]')
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_resolve_selection_returns_none_when_nothing_was_passed():
    from orbitals.scope_guard_v2.prompting import resolve_selection

    assert resolve_selection(None, None) is None
    assert resolve_selection(None, True) == ("reasoning", "scope_class", "suggested_response")
    assert resolve_selection(None, False) == (
        "evidences", "reasoning", "scope_class", "suggested_response",
    )
    assert resolve_selection(["scope_class", "evidences"], None) == ("evidences", "scope_class")


def test_build_prompt_prefill_uses_the_selections_first_key():
    from orbitals.scope_guard_v2.prompting import build_prompt

    class _Tok:
        def apply_chat_template(self, messages, **kwargs):
            return "<prompt>"

    assert build_prompt(_Tok(), "hello", "desc", prefill=True).endswith('{"evidences":')
    assert build_prompt(_Tok(), "hello", "desc", skip_evidences=True, prefill=True).endswith(
        '{"reasoning":'
    )
    assert build_prompt(
        _Tok(), "hello", "desc", prefill=True, output_fields=["scope_class"]
    ).endswith('{"scope_class":')
    assert build_prompt(_Tok(), "hello", "desc") == "<prompt>"


# --- base guard resolution --------------------------------------------------


def _stub_guard(**ctor):
    """A ScopeGuardV2 with no backend, exposing what _batch_validate received."""
    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2, ScopeGuardV2Output
    from orbitals.scope_guard_v2.guards.base import BaseScopeGuardV2

    class _Stub(ScopeGuardV2):
        def __new__(cls, *args, **kwargs):
            return BaseScopeGuardV2.__new__(cls)

        def __init__(self, **kwargs):
            super().__init__("stub", **kwargs)
            self.seen: list[tuple[str, ...]] = []

        def _validate(self, conversation, **kwargs):
            # every real backend delegates single to batch; the stub does too
            return self._batch_validate([conversation], **kwargs)[0]

        def _batch_validate(self, conversations, *, skip_evidences=None, output_fields=None, **kwargs):
            selection = self._resolve_output_fields(output_fields, skip_evidences)
            self.seen.append(selection)
            return [
                ScopeGuardV2Output(scope_class=ScopeClass.CHIT_CHAT, model="stub")
                for _ in conversations
            ]

    return _Stub(**ctor)


def test_base_guard_resolution_order():
    all_four = ("evidences", "reasoning", "scope_class", "suggested_response")
    no_evid = ("reasoning", "scope_class", "suggested_response")

    g = _stub_guard()
    g.validate("q", ai_service_description="d")
    assert g.seen[-1] == all_four  # nothing anywhere -> all fields

    g = _stub_guard(skip_evidences=True)
    g.validate("q", ai_service_description="d")
    assert g.seen[-1] == no_evid  # constructor skip_evidences
    g.validate("q", ai_service_description="d", skip_evidences=False)
    assert g.seen[-1] == all_four  # per-call skip_evidences overrides constructor
    g.validate("q", ai_service_description="d", output_fields=["scope_class"])
    assert g.seen[-1] == ("scope_class",)  # per-call output_fields overrides everything

    g = _stub_guard(output_fields=["reasoning", "scope_class"])
    g.validate("q", ai_service_description="d")
    assert g.seen[-1] == ("reasoning", "scope_class")  # constructor output_fields
    g.validate("q", ai_service_description="d", skip_evidences=True)
    assert g.seen[-1] == no_evid  # per-call skip_evidences beats constructor output_fields
    g.batch_validate(["a", "b"], ai_service_description="d", output_fields=["scope_class"])
    assert g.seen[-1] == ("scope_class",)


def test_base_guard_constructor_conflict_warns_once_at_construction():
    with pytest.warns(DeprecationWarning, match="output_fields"):
        g = _stub_guard(output_fields=["evidences", "scope_class"], skip_evidences=True)
    g.validate("q", ai_service_description="d")
    assert g.seen[-1] == ("evidences", "scope_class")


# --- vllm backends -----------------------------------------------------------


def test_check_shipped_system_prompt_warns_only_on_real_drift(tmp_path, caplog):
    import logging

    from orbitals.scope_guard_v2.guards.vllm import check_shipped_system_prompt
    from orbitals.scope_guard_v2.prompting import SYSTEM_PROMPT

    # no file: silent
    check_shipped_system_prompt(str(tmp_path))
    # same bytes minus trailing newline (what the Hub ships): silent
    (tmp_path / "system_prompt.txt").write_text(SYSTEM_PROMPT.rstrip("\n"), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        check_shipped_system_prompt(str(tmp_path))
    assert not caplog.records
    # a different prompt generation: warns and names both hashes
    (tmp_path / "system_prompt.txt").write_text("You are a 2606 classifier.", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        check_shipped_system_prompt(str(tmp_path))
    assert any("system_prompt.txt" in r.message and "f0f68e48" in r.message for r in caplog.records)


async def test_async_vllm_api_backend_sends_selection_schema_and_parses_partial_output(monkeypatch):
    """The HTTP vLLM backend must ask for exactly the selected keys and accept a
    completion that contains only them."""
    from orbitals.scope_guard_v2 import AsyncScopeGuardV2, ScopeClass

    captured: dict[str, Any] = {}

    class _Tok:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            return "PROMPT"

        def encode(self, text):
            return [0] * 10

    monkeypatch.setattr(
        "orbitals.scope_guard_v2.guards.vllm._get_tokenizer", lambda name: _Tok()
    )

    payload = {
        "choices": [{"text": '{"scope_class": "Out of Scope"}'}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    }

    # vllm.py uses `async with session.post(...) as response`, unlike api.py which
    # awaits it, so the fake must return an async context manager, not a coroutine.
    class _PostCtx:
        async def __aenter__(self):
            return _FakeAiohttpResponse(payload)

        async def __aexit__(self, *exc_info):
            return None

    class _FakeVllmSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        def post(self, url, *, json, headers):
            captured["url"] = url
            captured["json"] = json
            return _PostCtx()

    def _session_factory():
        return _FakeVllmSession()

    monkeypatch.setattr(
        "orbitals.scope_guard_v2.guards.vllm.aiohttp.ClientSession", _session_factory
    )

    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x")
    result = await sg.validate(
        "hello", ai_service_description="desc", output_fields=["scope_class"]
    )

    body = captured["json"]
    assert body["prompt"] == "PROMPT"
    assert list(body["structured_outputs"]["json"]["properties"]) == ["scope_class"]
    assert captured["messages"][1]["content"].endswith('["scope_class"]')
    assert result.scope_class == ScopeClass.OUT_OF_SCOPE
    assert result.reasoning is None
    assert result.evidences is None


# --- hf backend -------------------------------------------------------------


def test_hf_backend_passes_selection_to_the_pipeline_and_accepts_partial_output(monkeypatch):
    """hf.py imports transformers lazily inside __init__, so a fake module in
    sys.modules is enough -- the real package is not a test dependency."""
    import sys
    import types

    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2

    seen: dict[str, Any] = {}

    class _FakePipeline:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def __call__(self, inputs, **kwargs):
            seen["call"] = kwargs
            single = isinstance(inputs, tuple)
            # a real model emits exactly the requested keys; so must the fake, or the
            # strict per-selection schema rejects it (which is the point of the schema)
            payload = {"reasoning": "r", "scope_class": "Chit Chat"}
            payload = {k: v for k, v in payload.items() if k in kwargs["output_fields"]}
            out = [{"generated_text": json.dumps(payload)}]
            return out if single else [out for _ in inputs]

    monkeypatch.setattr("orbitals.utils.maybe_configure_gpu_usage", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(pipeline=lambda **kwargs: _FakePipeline(**kwargs)),
    )

    sg = ScopeGuardV2(backend="hf", model="m", output_fields=["reasoning", "scope_class"])
    assert seen["init"]["output_fields"] == ("reasoning", "scope_class")

    result = sg.validate("hello", ai_service_description="desc")
    assert seen["call"]["output_fields"] == ("reasoning", "scope_class")
    assert result.scope_class == ScopeClass.CHIT_CHAT
    assert result.reasoning == "r"
    assert result.evidences is None

    results = sg.batch_validate(["a", "b"], ai_service_description="desc", output_fields=["scope_class"])
    assert seen["call"]["output_fields"] == ("scope_class",)
    assert len(results) == 2
