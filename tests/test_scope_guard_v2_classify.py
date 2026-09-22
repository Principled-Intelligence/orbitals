"""classify(): class probabilities off the logits and predefined-response selection."""

from __future__ import annotations

import re
import sys
import types
from typing import Any

import pytest

from orbitals.scope_guard_v2 import (
    AsyncScopeGuardV2,
    ScopeClass,
    ScopeGuardV2,
    ScopeGuardV2Classification,
)
from orbitals.scope_guard_v2.prompting import (
    CLASS_PREFIX,
    PREDEFINED_PREFIX,
    SYSTEM_PROMPT,
    class_first_tokens,
    class_probabilities,
    predefined_candidates,
)
from orbitals.types import AIServiceDescriptionV2, PredefinedResponse


class _Tokenizer:
    """Deterministic tokenizer over words, whitespace runs and single punctuation marks,
    so a quote before a class name stays its own token as it does in the real BPE."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for w in re.findall(r"\w+|\s+|[^\w\s]", text):
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
                self.inv[self.vocab[w]] = w
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv[i] for i in ids)

    def apply_chat_template(self, messages, **kwargs) -> str:
        return "PROMPT " + messages[1]["content"][-40:]


TOKENS = class_first_tokens(_Tokenizer())
ASD = AIServiceDescriptionV2(
    identity_role="Pharmacy assistant",
    context="Customers",
    predefined_responses=[
        PredefinedResponse(trigger="diagnosis", response="I cannot provide medical diagnoses."),
        PredefinedResponse(trigger="orders", response="For help with your order call 345."),
    ],
)


# --- prompting helpers ----------------------------------------------------------------


def test_class_first_tokens_are_distinct_and_prefix_preserving() -> None:
    assert set(TOKENS) == {c.value for c in ScopeClass}
    assert len(set(TOKENS.values())) == 7
    assert TOKENS["Restricted"] == "Restricted" and TOKENS["Out of Scope"] == "Out"


def test_class_probabilities_softmax_temperature_and_floor() -> None:
    first = {c.value: c.value for c in ScopeClass}
    top = {"Restricted": -0.1, "Out of Scope": -2.5}  # five classes missing from top-k

    p = class_probabilities(top, first)
    assert abs(sum(p.values()) - 1) < 1e-9
    assert p["Restricted"] > p["Out of Scope"] > p["Chit Chat"]
    assert p["Chit Chat"] == p["Human Oversight"]  # both floored identically

    flat = class_probabilities(top, first, temperature=100.0)
    assert flat["Restricted"] < p["Restricted"]
    assert max(flat, key=flat.__getitem__) == "Restricted"
    with pytest.raises(ValueError):
        class_probabilities(top, first, temperature=0)



def test_predefined_candidates_shapes() -> None:
    assert predefined_candidates("free text") == []
    base = {"identity_role": "r", "context": "c"}
    assert predefined_candidates(AIServiceDescriptionV2(**base)) == []
    assert predefined_candidates(AIServiceDescriptionV2(**base, predefined_responses="a string")) == []
    asd = AIServiceDescriptionV2(**base, predefined_responses=[PredefinedResponse(trigger="t1", response="R1"), "R2", "  "])
    assert predefined_candidates(asd) == [("t1", "R1"), (None, "R2")]


# --- vllm-api backend ------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> Any:
        return self._payload


class _PostCtx:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def __aenter__(self):
        return _FakeResponse(self._payload)

    async def __aexit__(self, *exc_info):
        return None


def _install_session(monkeypatch, payloads: list[Any]) -> list[dict]:
    captured: list[dict] = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        def post(self, url, *, json, headers):
            captured.append({"url": url, "json": json})
            return _PostCtx(payloads[len(captured) - 1])

    monkeypatch.setattr("orbitals.scope_guard_v2.guards.vllm.aiohttp.ClientSession", lambda: _Session())
    monkeypatch.setattr("orbitals.scope_guard_v2.guards.vllm._get_tokenizer", lambda name: _Tokenizer())
    from orbitals.scope_guard_v2.guards import vllm as guard_module

    guard_module._class_tokens.cache_clear()
    return captured


def _class_payload(top: dict[str, float]) -> dict:
    return {
        "choices": [{"text": "x", "logprobs": {"top_logprobs": [top], "tokens": ["x"]}}],
        "usage": {"prompt_tokens": 110, "completion_tokens": 1, "total_tokens": 111},
    }


def _system_tokens() -> int:
    return len(_Tokenizer().encode(SYSTEM_PROMPT))


async def test_fast_classify_reads_probabilities_and_selects_the_entry(monkeypatch) -> None:
    captured = _install_session(
        monkeypatch,
        [
            _class_payload({TOKENS["Predefined Answer"]: -0.05, TOKENS["Directly Supported"]: -3.0}),
            {
                "choices": [{"text": "For help with your order call 345.", "logprobs": {"token_logprobs": [-0.1, -0.2]}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128},
            },
        ],
    )
    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x", decision_temperature=1.5)

    r = await sg.classify("my order is late", ai_service_description=ASD)

    assert isinstance(r, ScopeGuardV2Classification)
    assert r.scope_class is ScopeClass.PREDEFINED_ANSWER
    assert r.temperature == 1.5 and abs(sum(r.probabilities.values()) - 1) < 1e-9
    assert r.confidence == r.probabilities["Predefined Answer"] > 0.8
    assert r.predefined_response == "For help with your order call 345."
    first, second = captured
    assert first["json"]["max_tokens"] == 1 and first["json"]["logprobs"] == 20
    assert first["json"]["prompt"].endswith(CLASS_PREFIX) and "structured_outputs" not in first["json"]
    assert second["json"]["structured_outputs"] == {"choice": [p.response for p in ASD.predefined_responses]}  # type: ignore[union-attr]
    assert second["json"]["prompt"].endswith(PREDEFINED_PREFIX) and second["json"]["temperature"] == 0.0
    assert "logprobs" not in second["json"] and "stop" not in second["json"]
    assert r.usage is not None and r.usage.completion_tokens == 9
    assert r.usage.prompt_tokens == 230 - 2 * _system_tokens()



async def test_classify_returns_no_text_for_other_classes_or_when_not_resolving(monkeypatch) -> None:
    captured = _install_session(monkeypatch, [_class_payload({TOKENS["Restricted"]: -0.01}), _class_payload({TOKENS["Predefined Answer"]: -0.01})])
    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x")

    r1 = await sg.classify("q", ai_service_description=ASD)
    r2 = await sg.classify("q", ai_service_description=ASD, resolve_predefined=False)

    assert r1.scope_class is ScopeClass.RESTRICTED and r1.predefined_response is None
    assert r2.scope_class is ScopeClass.PREDEFINED_ANSWER and r2.predefined_response is None
    assert len(captured) == 2


async def test_classify_generates_the_reply_when_the_description_has_no_list(monkeypatch) -> None:
    captured = _install_session(
        monkeypatch,
        [
            _class_payload({TOKENS["Predefined Answer"]: -0.01}),
            {"choices": [{"text": 'Per assistenza chiama il \\"numero\\" 345."}'}], "usage": {"prompt_tokens": 130, "completion_tokens": 12, "total_tokens": 142}},
        ],
    )
    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x")

    r = await sg.classify("il mio ordine?", ai_service_description="A pharmacy assistant; for orders say to call 345.")

    assert r.scope_class is ScopeClass.PREDEFINED_ANSWER
    assert r.predefined_response == 'Per assistenza chiama il "numero" 345.'
    second = captured[1]["json"]
    assert second["prompt"].endswith(PREDEFINED_PREFIX) and second["stop"] == ['"}']
    assert "structured_outputs" not in second
    assert r.usage is not None and r.usage.completion_tokens == 13


async def test_classify_single_candidate_needs_no_second_call(monkeypatch) -> None:
    captured = _install_session(monkeypatch, [_class_payload({TOKENS["Predefined Answer"]: -0.01})])
    asd = AIServiceDescriptionV2(identity_role="r", context="c", predefined_responses=["Only answer."])
    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x")

    r = await sg.classify("q", ai_service_description=asd)

    assert r.predefined_response == "Only answer."
    assert len(captured) == 1


async def test_classify_rejects_non_candidate_text(monkeypatch) -> None:
    _install_session(
        monkeypatch,
        [_class_payload({TOKENS["Predefined Answer"]: -0.01}), {"choices": [{"text": "something else", "logprobs": {"token_logprobs": [-1.0]}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}],
    )
    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x")
    with pytest.raises(ValueError, match="non-candidate"):
        await sg.classify("q", ai_service_description=ASD)


# --- offline vllm backend --------------------------------------------------------------


def _offline_guard(monkeypatch, outputs: list[Any], captured: list[Any]):
    monkeypatch.setattr("orbitals.utils.maybe_configure_gpu_usage", lambda: None)
    monkeypatch.setattr("orbitals.scope_guard_v2.guards.vllm._get_tokenizer", lambda name: _Tokenizer())
    from orbitals.scope_guard_v2.guards import vllm as guard_module

    guard_module._class_tokens.cache_clear()

    def generate(prompts, params, use_tqdm=False):
        captured.append((prompts[0], params))
        return [outputs[len(captured) - 1]]

    fake_vllm = types.SimpleNamespace(
        LLM=lambda **kw: types.SimpleNamespace(generate=generate),
        SamplingParams=lambda **kw: dict(kw),
        sampling_params=types.SimpleNamespace(StructuredOutputsParams=lambda **kw: dict(kw)),
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    return ScopeGuardV2(backend="vllm", model="m")


def _lp(token: str, logprob: float):
    return types.SimpleNamespace(decoded_token=token, logprob=logprob)


def _offline_output(text: str, steps: list[dict], prompt_len: int = 50):
    return types.SimpleNamespace(
        prompt_token_ids=list(range(prompt_len)),
        outputs=[types.SimpleNamespace(text=text, token_ids=list(range(len(steps))), logprobs=steps)],
    )


def test_offline_fast_classify_and_selection(monkeypatch) -> None:
    captured: list[Any] = []
    guard = _offline_guard(
        monkeypatch,
        [
            _offline_output("x", [{1: _lp(TOKENS["Predefined Answer"], -0.1), 2: _lp(TOKENS["Restricted"], -2.0)}]),
            _offline_output("I cannot provide medical diagnoses.", [{3: _lp("I", -0.1)}, {4: _lp(" cannot", -0.2)}]),
        ],
        captured,
    )

    r = guard.classify("diagnose me", ai_service_description=ASD)

    assert r.scope_class is ScopeClass.PREDEFINED_ANSWER and r.temperature == 1.0
    assert r.predefined_response == "I cannot provide medical diagnoses."
    assert captured[0][0].endswith(CLASS_PREFIX) and captured[0][1]["max_tokens"] == 1
    assert captured[1][1]["structured_outputs"] == {"choice": [p.response for p in ASD.predefined_responses]}  # type: ignore[union-attr]
    assert r.usage is not None and r.usage.completion_tokens == 3



# --- other backends and serving ---------------------------------------------------------


def test_classify_is_unavailable_on_the_hosted_api_backend() -> None:
    sg = ScopeGuardV2(backend="api", api_key="k")
    with pytest.raises(NotImplementedError, match="api"):
        sg.classify("q", ai_service_description="d")


def test_serving_classify_endpoint_and_temperature_env(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from orbitals.scope_guard_v2.serving import main as serving_main
    from orbitals.types import LLMUsage

    monkeypatch.setenv("SCOPE_GUARD_V2_VLLM_MODEL", "v2-model")
    monkeypatch.setenv("SCOPE_GUARD_V2_VLLM_SERVING_URL", "http://localhost:8001")
    monkeypatch.setenv("SCOPE_GUARD_V2_DECISION_TEMPERATURE", "1.7")
    monkeypatch.setattr("orbitals.scope_guard_v2.prompting.check_shipped_system_prompt", lambda ref: None)
    seen: dict[str, Any] = {}

    class _Stub:
        async def classify(self, conversation, *, ai_service_description, **kwargs):
            seen.update(kwargs)
            return ScopeGuardV2Classification(
                scope_class=ScopeClass.PREDEFINED_ANSWER,
                probabilities={c.value: (0.94 if c is ScopeClass.PREDEFINED_ANSWER else 0.01) for c in ScopeClass},
                confidence=0.94,
                temperature=1.7,
                predefined_response="R",
                model="stub",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    with TestClient(serving_main.app) as client:
        assert serving_main.scope_guard.decision_temperature == 1.7  # type: ignore[attr-defined]
        monkeypatch.setattr(serving_main, "scope_guard", _Stub())
        r = client.post(
            "/orbitals/scope-guard-v2/classify",
            json={"conversation": "hi", "ai_service_description": "d", "resolve_predefined": False},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scope_class"] == "Predefined Answer" and body["predefined_response"] == "R"
    assert body["confidence"] == 0.94 and body["temperature"] == 1.7 and "time_taken" in body
    assert seen["resolve_predefined"] is False and "fast" not in seen and "temperature" not in seen


def test_serve_cli_rejects_a_non_positive_decision_temperature() -> None:
    from typer.testing import CliRunner

    from orbitals.scope_guard_v2.cli.serve import app

    result = CliRunner().invoke(app, ["some/model", "--decision-temperature", "0"])
    assert result.exit_code != 0
    assert "decision-temperature" in result.output


async def test_multi_line_entries_are_collapsed_for_the_grammar_and_returned_verbatim(monkeypatch) -> None:
    """vLLM cannot compile a choice grammar over texts with line breaks; the grammar sees
    one-line texts, the caller still gets the description's own entry."""
    address = "Address: Via di Casal Boccone, 188\nPhone: (+39) 0639931\nFax: (+39) 0639935"
    asd = AIServiceDescriptionV2(identity_role="r", context="c", predefined_responses=[address, "Write to info@almawave.it."])
    captured = _install_session(
        monkeypatch,
        [
            _class_payload({TOKENS["Predefined Answer"]: -0.01}),
            {"choices": [{"text": "Address: Via di Casal Boccone, 188 Phone: (+39) 0639931 Fax: (+39) 0639935"}], "usage": {"prompt_tokens": 1, "completion_tokens": 20, "total_tokens": 21}},
        ],
    )
    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x")

    r = await sg.classify("come vi contatto?", ai_service_description=asd)

    assert captured[1]["json"]["structured_outputs"]["choice"] == ["Address: Via di Casal Boccone, 188 Phone: (+39) 0639931 Fax: (+39) 0639935", "Write to info@almawave.it."]
    assert r.predefined_response == address
