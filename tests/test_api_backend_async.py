"""Tests for the async API backend (`AsyncScopeGuard(backend="api", ...)`).

Mirrors the sync tests in `test_api_backend_sync.py` but uses the async API
client, which is built on top of `aiohttp` instead of `requests`.
"""

from __future__ import annotations

from typing import Any

import pytest

from orbitals.scope_guard import AsyncScopeGuard, ScopeClass, ScopeGuardOutput


class _FakeAiohttpResponse:
    def __init__(self, payload: Any):
        self._payload = payload

    def raise_for_status(self):  # called synchronously by the client
        return None

    async def json(self):
        return self._payload


class _FakeAiohttpSession:
    """Stand-in for `aiohttp.ClientSession` recording the last POST call."""

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


@pytest.fixture
def captured():
    return {}


@pytest.fixture
def patch_session(captured, monkeypatch):
    """Install a fake aiohttp.ClientSession that records calls."""
    state: dict[str, Any] = {"payload": _default_payload()}

    def _session_factory():
        return _FakeAiohttpSession(state["payload"], captured)

    monkeypatch.setattr(
        "orbitals.scope_guard.guards.api.aiohttp.ClientSession",
        _session_factory,
    )
    return state


def _default_payload() -> dict[str, Any]:
    return {
        "scope_class": "Restricted",
        "evidences": ["No refunds."],
        "model": "principled-intelligence/scope-guard-4B-q-2601",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def test_async_api_backend_is_constructible_without_network():
    sg = AsyncScopeGuard(backend="api", api_url="http://localhost:8000")
    assert sg.backend == "api"
    assert sg.api_url == "http://localhost:8000"


async def test_async_validate_hits_documented_endpoint(captured, patch_session):
    sg = AsyncScopeGuard(backend="api", api_url="http://example.com")
    result = await sg.validate("hi", ai_service_description="desc")

    assert captured["url"] == "http://example.com/orbitals/scope-guard/validate"
    assert isinstance(result, ScopeGuardOutput)
    assert result.scope_class == ScopeClass.RESTRICTED


async def test_async_batch_validate_hits_documented_endpoint(
    captured, patch_session
):
    patch_session["payload"] = [_default_payload(), _default_payload()]
    sg = AsyncScopeGuard(backend="api", api_url="http://example.com")
    results = await sg.batch_validate(
        ["q1", "q2"], ai_service_description="desc"
    )

    assert (
        captured["url"]
        == "http://example.com/orbitals/scope-guard/batch-validate"
    )
    assert len(results) == 2
    assert all(isinstance(r, ScopeGuardOutput) for r in results)


async def test_async_validate_sends_headers_and_payload(captured, patch_session):
    sg = AsyncScopeGuard(
        backend="api",
        api_url="http://example.com",
        api_key="secret",
        skip_evidences=True,
    )
    await sg.validate({"role": "user", "content": "hi"}, ai_service_description="desc")

    assert captured["json"]["conversation"] == {"role": "user", "content": "hi"}
    assert captured["json"]["ai_service_description"] == "desc"
    assert captured["json"]["skip_evidences"] is True
    assert captured["headers"]["X-API-Key"] == "secret"
    assert captured["headers"]["Content-Type"] == "application/json"


async def test_async_batch_validate_body_with_per_conv_descriptions(
    captured, patch_session
):
    patch_session["payload"] = [_default_payload(), _default_payload()]
    sg = AsyncScopeGuard(backend="api", api_url="http://example.com")
    await sg.batch_validate(
        ["q1", "q2"], ai_service_descriptions=["d1", "d2"]
    )

    assert captured["json"]["conversations"] == ["q1", "q2"]
    assert captured["json"]["ai_service_descriptions"] == ["d1", "d2"]
    assert "ai_service_description" not in captured["json"]


async def test_async_empty_batch_returns_empty_without_touching_network(
    captured, patch_session
):
    sg = AsyncScopeGuard(backend="api", api_url="http://example.com")
    results = await sg.batch_validate([], ai_service_description="desc")
    assert results == []
    assert captured == {}  # no HTTP call was made
