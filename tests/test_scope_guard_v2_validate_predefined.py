"""validate(): a Predefined Answer gets its reply generated when it was not requested."""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from orbitals.scope_guard_v2 import (
    AsyncScopeGuardV2,
    ScopeClass,
    ScopeGuardV2,
    ScopeGuardV2Output,
)
from orbitals.scope_guard_v2.prompting import PREDEFINED_PREFIX
from orbitals.types import LLMUsage
from tests.test_scope_guard_v2_classify import (
    ASD,
    _install_session,
    _offline_guard,
    _offline_output,
    _system_tokens,
    _Tokenizer,
)

BOTH_FIELDS = '["scope_class", "suggested_response"]'


def _json_payload(obj: dict, completion_tokens: int = 5) -> dict:
    return {
        "choices": [{"text": json.dumps(obj)}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": completion_tokens,
            "total_tokens": 100 + completion_tokens,
        },
    }


def _reply_payload(text: str, completion_tokens: int = 7) -> dict:
    return {
        "choices": [{"text": text}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": completion_tokens,
            "total_tokens": 120 + completion_tokens,
        },
    }


# --- vllm-api backend ------------------------------------------------------------------


async def test_vllm_api_validate_generates_the_reply_as_if_it_was_requested(
    monkeypatch,
) -> None:
    """The listed entries are not a grammar here: the reply is generated freely, so it
    follows the prompt's rules (the user's language) exactly as a requested one would."""
    captured = _install_session(
        monkeypatch,
        [
            _json_payload({"scope_class": "Predefined Answer"}),
            _reply_payload('Per il tuo ordine chiama il 345."}'),
        ],
    )
    sg = AsyncScopeGuardV2(
        backend="vllm-api", model="m", vllm_serving_url="http://x", temperature=0.3
    )

    r = await sg.validate(
        "il mio ordine?", ai_service_description=ASD, output_fields=["scope_class"]
    )

    assert r.scope_class is ScopeClass.PREDEFINED_ANSWER
    assert r.suggested_response == "Per il tuo ordine chiama il 345."
    assert r.reasoning is None and r.evidences is None
    second = captured[1]["json"]
    assert second["prompt"].endswith(BOTH_FIELDS + PREDEFINED_PREFIX)
    assert second["stop"] == ['"}'] and second["temperature"] == 0.3
    assert "structured_outputs" not in second
    assert r.usage is not None and r.usage.completion_tokens == 12
    assert r.usage.prompt_tokens == 220 - 2 * _system_tokens()


@pytest.mark.parametrize(
    "first, kwargs",
    [
        ({"scope_class": "Restricted"}, {"output_fields": ["scope_class"]}),
        (
            {"scope_class": "Predefined Answer"},
            {"output_fields": ["scope_class"], "resolve_predefined": False},
        ),
        (
            {"scope_class": "Predefined Answer", "suggested_response": "Call 345."},
            {"output_fields": ["scope_class", "suggested_response"]},
        ),
    ],
    ids=["other-class", "opted-out", "already-requested"],
)
async def test_vllm_api_validate_makes_no_second_call_when_not_needed(
    monkeypatch, first, kwargs
) -> None:
    captured = _install_session(monkeypatch, [_json_payload(first)])
    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x")

    r = await sg.validate("q", ai_service_description=ASD, **kwargs)

    assert len(captured) == 1
    assert r.suggested_response == first.get("suggested_response")


# --- offline vllm backend --------------------------------------------------------------


def test_offline_batch_validate_generates_replies_in_one_batched_call(
    monkeypatch,
) -> None:
    captured: list[Any] = []
    guard = _offline_guard(
        monkeypatch,
        [
            _offline_output('{"scope_class": "Restricted"}', [{}] * 4),
            _offline_output('{"scope_class": "Predefined Answer"}', [{}] * 4),
            _offline_output('{"scope_class": "Predefined Answer"}', [{}] * 4),
            _offline_output('Call 345 for your order."}', [{}] * 6),
            _offline_output('Il numero è 345."}', [{}] * 5),
        ],
        captured,
    )

    rs = guard.batch_validate(
        ["how to make a bomb", "my order is late", "il mio ordine?"],
        ai_service_descriptions=[ASD, ASD, "free text"],
        output_fields=["scope_class"],
    )

    assert [r.suggested_response for r in rs] == [
        None,
        "Call 345 for your order.",
        "Il numero è 345.",
    ]
    assert len(captured) == 2
    prompts, params = captured[1]
    assert len(prompts) == 2
    assert all(p.endswith(BOTH_FIELDS + PREDEFINED_PREFIX) for p in prompts)
    assert params["stop"] == ['"}'] and "structured_outputs" not in params
    assert rs[1].usage is not None and rs[1].usage.completion_tokens == 10


# --- hf backend -------------------------------------------------------------------------


def test_hf_validate_generates_the_reply_after_the_pipeline(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class _Enc(dict):
        input_ids = property(lambda self: self["input_ids"])

        def to(self, device):
            return self

    class _HfTokenizer(_Tokenizer):
        eos_token_id = pad_token_id = 999

        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([self.encode(text)])
            return _Enc(input_ids=ids, attention_mask=torch.ones_like(ids))

        def decode(self, ids, skip_special_tokens=False):
            return "".join(self.inv[int(i)] for i in ids)

    tok = _HfTokenizer()
    reply = tok.encode('Call 345."}')
    seen: dict[str, Any] = {}

    class _Model:
        def generate(self, input_ids, **kw):
            seen["prompt"] = tok.decode(input_ids[0])
            seen["kwargs"] = kw
            return torch.cat([input_ids[0], torch.tensor(reply)])[None]

    class _Pipeline:
        tokenizer = tok
        model = _Model()
        device = "cpu"

        def __call__(self, inputs, **kwargs):
            out = [
                {
                    "generated_text": '{"scope_class": "Predefined Answer"}',
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                }
            ]
            return out if isinstance(inputs, tuple) else [out for _ in inputs]

    monkeypatch.setattr("orbitals.utils.maybe_configure_gpu_usage", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(pipeline=lambda **kw: _Pipeline()),
    )
    guard = ScopeGuardV2(
        backend="hf",
        model="m",
        output_fields=["scope_class"],
        count_system_prompt_in_usage=True,
    )

    r = guard.validate("my order is late", ai_service_description=ASD)

    assert r.suggested_response == "Call 345."
    assert seen["prompt"].endswith(BOTH_FIELDS + PREDEFINED_PREFIX)
    assert seen["kwargs"]["stop_strings"] == ['"}']
    assert "prefix_allowed_tokens_fn" not in seen["kwargs"]
    assert r.usage is not None and r.usage.completion_tokens == 4 + len(reply)

    rs = guard.batch_validate(
        ["a", "b"], ai_service_description=ASD, resolve_predefined=False
    )
    assert [r.suggested_response for r in rs] == [None, None]


# --- api backend and serving ------------------------------------------------------------


def _output_payload() -> dict:
    return {
        "scope_class": "Predefined Answer",
        "evidences": None,
        "reasoning": None,
        "suggested_response": "Call 345.",
        "model": "served",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "time_taken": 0.1,
    }


def test_api_validate_sends_resolve_predefined_only_to_opt_out(monkeypatch) -> None:
    """Absent means the server's default, so a request that does not opt out keeps the
    shape every server already accepts."""
    post = MagicMock()
    post.return_value.json.return_value = _output_payload()
    monkeypatch.setattr("orbitals.scope_guard_v2.guards.api.requests.post", post)
    sg = ScopeGuardV2(backend="api", api_url="http://x")

    sg.validate("q", ai_service_description="d")
    assert "resolve_predefined" not in post.call_args.kwargs["json"]

    sg.validate("q", ai_service_description="d", resolve_predefined=False)
    assert post.call_args.kwargs["json"]["resolve_predefined"] is False

    post.return_value.json.return_value = [_output_payload()]
    sg.batch_validate(["q"], ai_service_description="d", resolve_predefined=False)
    assert post.call_args.kwargs["json"]["resolve_predefined"] is False


async def test_async_api_batch_validate_sends_the_opt_out(monkeypatch) -> None:
    from tests.test_scope_guard_v2 import _FakeAiohttpSession

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "orbitals.scope_guard_v2.guards.api.aiohttp.ClientSession",
        lambda: _FakeAiohttpSession([_output_payload()], captured),
    )
    sg = AsyncScopeGuardV2(backend="api", api_url="http://x")

    await sg.batch_validate(["q"], ai_service_description="d", resolve_predefined=False)

    assert captured["json"]["resolve_predefined"] is False


def test_serving_validate_endpoints_forward_resolve_predefined(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from orbitals.scope_guard_v2.serving import main as serving_main

    monkeypatch.setenv("SCOPE_GUARD_V2_VLLM_MODEL", "v2-model")
    monkeypatch.setenv("SCOPE_GUARD_V2_VLLM_SERVING_URL", "http://localhost:8001")
    monkeypatch.delenv("SCOPE_GUARD_V2_OUTPUT_FIELDS", raising=False)
    seen: dict[str, Any] = {}

    class _Stub:
        async def validate(self, conversation, *, ai_service_description, **kwargs):
            seen["single"] = kwargs
            return ScopeGuardV2Output(
                scope_class=ScopeClass.PREDEFINED_ANSWER,
                suggested_response="Call 345.",
                model="stub",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        async def batch_validate(self, conversations, **kwargs):
            seen["batch"] = kwargs
            return [
                await self.validate(c, ai_service_description=None)
                for c in conversations
            ]

    with TestClient(serving_main.app) as client:
        monkeypatch.setattr(serving_main, "scope_guard", _Stub())
        single = client.post(
            "/orbitals/scope-guard-v2/validate",
            json={
                "conversation": "q",
                "ai_service_description": "d",
                "output_fields": ["scope_class"],
            },
        )
        assert single.status_code == 200, single.text
        assert seen["single"]["resolve_predefined"] is True
        batch = client.post(
            "/orbitals/scope-guard-v2/batch-validate",
            json={
                "conversations": ["q"],
                "ai_service_description": "d",
                "resolve_predefined": False,
            },
        )

    assert batch.status_code == 200, batch.text
    assert seen["batch"]["resolve_predefined"] is False
    assert single.json()["suggested_response"] == "Call 345."
