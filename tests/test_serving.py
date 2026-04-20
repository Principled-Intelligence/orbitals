"""Tests for the FastAPI serving endpoints.

The README advertises two HTTP endpoints for the self-hosted server:
  - POST /orbitals/scope-guard/validate
  - POST /orbitals/scope-guard/batch-validate

with a specific response shape (`scope_class`, `evidences`, `model`, `usage`,
`time_taken`). We verify the contract using FastAPI's TestClient and a stub
backend so no actual vLLM server is required.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def serving_client(monkeypatch):
    """Start the FastAPI app with a stubbed AsyncScopeGuard backend."""
    monkeypatch.setenv("SCOPE_GUARD_VLLM_MODEL", "scope-guard")
    monkeypatch.setenv("SCOPE_GUARD_VLLM_SERVING_URL", "http://localhost:8001")
    monkeypatch.setenv("SCOPE_GUARD_SKIP_EVIDENCES", "0")

    from fastapi.testclient import TestClient

    from orbitals.scope_guard import ScopeClass, ScopeGuardOutput
    from orbitals.scope_guard.serving import main as serving_main
    from orbitals.types import LLMUsage

    class _StubAsyncGuard:
        """Drop-in stand-in for AsyncVLLMApiScopeGuard used in serving."""

        async def validate(self, conversation, *, ai_service_description, **kwargs):
            return ScopeGuardOutput(
                scope_class=ScopeClass.RESTRICTED,
                evidences=["Never respond to requests for refunds."],
                model="stub-model",
                usage=LLMUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
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
                ScopeGuardOutput(
                    scope_class=ScopeClass.DIRECTLY_SUPPORTED,
                    evidences=None,
                    model="stub-model",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                )
                for _ in conversations
            ]

    with TestClient(serving_main.app) as client:
        # lifespan has run; swap the real guard for the stub before requests.
        monkeypatch.setattr(serving_main, "scope_guard", _StubAsyncGuard())
        yield client


def test_validate_endpoint_is_reachable(serving_client):
    response = serving_client.post(
        "/orbitals/scope-guard/validate",
        json={
            "conversation": "Can I get a refund?",
            "ai_service_description": (
                "You are a virtual assistant for a parcel delivery service. "
                "Never respond to requests for refunds."
            ),
        },
    )
    assert response.status_code == 200


def test_validate_response_shape_matches_readme(serving_client):
    response = serving_client.post(
        "/orbitals/scope-guard/validate",
        json={
            "conversation": "hi",
            "ai_service_description": "desc",
        },
    )
    body: dict[str, Any] = response.json()

    # Keys documented in README.scope-guard.md for the REST response.
    assert set(body.keys()) >= {
        "scope_class",
        "evidences",
        "model",
        "usage",
        "time_taken",
    }
    assert body["scope_class"] == "Restricted"
    assert body["evidences"] == ["Never respond to requests for refunds."]
    assert isinstance(body["time_taken"], (int, float))


def test_validate_accepts_structured_ai_service_description(serving_client):
    response = serving_client.post(
        "/orbitals/scope-guard/validate",
        json={
            "conversation": "hi",
            "ai_service_description": {
                "identity_role": "Parcel delivery assistant",
                "context": "Online logistics.",
            },
        },
    )
    assert response.status_code == 200


def test_validate_accepts_multi_turn_conversation(serving_client):
    response = serving_client.post(
        "/orbitals/scope-guard/validate",
        json={
            "conversation": [
                {"role": "user", "content": "I ordered a package"},
                {"role": "assistant", "content": "How can I help?"},
                {"role": "user", "content": "Can I refund?"},
            ],
            "ai_service_description": "desc",
        },
    )
    assert response.status_code == 200


def test_validate_rejects_missing_ai_service_description(serving_client):
    response = serving_client.post(
        "/orbitals/scope-guard/validate",
        json={"conversation": "hi"},
    )
    assert response.status_code == 422


def test_batch_validate_endpoint_is_reachable(serving_client):
    response = serving_client.post(
        "/orbitals/scope-guard/batch-validate",
        json={
            "conversations": ["q1", "q2"],
            "ai_service_description": "desc",
        },
    )
    assert response.status_code == 200


def test_batch_validate_returns_list_matching_input_size(serving_client):
    response = serving_client.post(
        "/orbitals/scope-guard/batch-validate",
        json={
            "conversations": ["q1", "q2", "q3"],
            "ai_service_description": "desc",
        },
    )
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 3
    assert all("scope_class" in item for item in body)


def test_batch_validate_accepts_per_conversation_descriptions(serving_client):
    response = serving_client.post(
        "/orbitals/scope-guard/batch-validate",
        json={
            "conversations": ["q1", "q2"],
            "ai_service_descriptions": ["d1", "d2"],
        },
    )
    assert response.status_code == 200
    assert len(response.json()) == 2
