"""Tests for ClaimExtractor FastAPI serving endpoints."""

from __future__ import annotations

import itertools
from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def claim_extractor_serving_client(monkeypatch):
    monkeypatch.setenv("CLAIM_EXTRACTOR_VLLM_MODEL", "claim-extractor")
    monkeypatch.setenv("CLAIM_EXTRACTOR_VLLM_SERVING_URL", "http://localhost:8001")
    monkeypatch.setenv("CLAIM_EXTRACTOR_SKIP_EVIDENCES", "1")

    from fastapi.testclient import TestClient

    from orbitals.claim_extractor.modeling import ClaimExtractorOutput, Extractions
    from orbitals.claim_extractor.serving import main as serving_main
    from orbitals.types import LLMUsage

    class _StubAsyncExtractor:
        def __init__(self):
            self.extract_calls: list[Any] = []
            self.batch_extract_calls: list[Any] = []

        async def extract(self, conversation, *, ai_service_description=None, **kwargs):
            self.extract_calls.append(conversation)
            return ClaimExtractorOutput(
                extractions=Extractions(),
                model="stub-model",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            )

        async def batch_extract(
            self,
            conversations,
            *,
            ai_service_description=None,
            ai_service_descriptions=None,
            **kwargs,
        ):
            self.batch_extract_calls.append(list(conversations))
            return [
                ClaimExtractorOutput(
                    extractions=Extractions(),
                    model="stub-model",
                    usage=LLMUsage(
                        prompt_tokens=i + 1,
                        completion_tokens=i + 2,
                        total_tokens=(i + 1) + (i + 2),
                    ),
                )
                for i, _ in enumerate(conversations)
            ]

    with TestClient(serving_main.app) as client:
        stub = _StubAsyncExtractor()
        monkeypatch.setattr(serving_main, "claim_extractor", stub)
        setattr(client, "_claim_extractor_stub", stub)
        yield client


def test_extract_endpoint_response_matches_readme_shape(claim_extractor_serving_client):
    response = claim_extractor_serving_client.post(
        "/orbitals/claim-extractor/extract",
        json={
            "conversation": "Your package is in transit.",
            "ai_service_description": "You are a parcel delivery assistant.",
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert set(body.keys()) >= {"extractions", "model", "usage", "time_taken"}
    assert body["extractions"] == {"intents": [], "claims": []}
    assert body["model"] == "stub-model"
    assert body["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


def test_batch_extract_reports_distinct_per_item_timings(
    claim_extractor_serving_client,
    monkeypatch,
):
    from orbitals.claim_extractor.serving import main as serving_main

    times = itertools.chain([10.0, 11.0, 20.0, 23.0], itertools.repeat(99.0))
    monkeypatch.setattr(serving_main.time, "time", lambda: next(times))

    response = claim_extractor_serving_client.post(
        "/orbitals/claim-extractor/batch-extract",
        json={
            "conversations": ["q1", "q2"],
            "ai_service_description": "desc",
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert [item["time_taken"] for item in body] == [1.0, 3.0]


def test_extract_conversation_returns_one_extraction_per_prefix(
    claim_extractor_serving_client,
):
    response = claim_extractor_serving_client.post(
        "/orbitals/claim-extractor/extract-conversation",
        json={
            "conversation": [
                {"role": "user", "content": "I ordered a package."},
                {"role": "assistant", "content": "It arrives tomorrow."},
            ],
            "ai_service_description": "You are a parcel delivery assistant.",
        },
    )

    body = response.json()
    stub = getattr(claim_extractor_serving_client, "_claim_extractor_stub")

    assert response.status_code == 200
    assert isinstance(body["extractions"], list)
    assert len(body["extractions"]) == 2
    assert [
        [message.model_dump() for message in prefix]
        for prefix in stub.batch_extract_calls[-1]
    ] == [
        [{"role": "user", "content": "I ordered a package."}],
        [
            {"role": "user", "content": "I ordered a package."},
            {"role": "assistant", "content": "It arrives tomorrow."},
        ],
    ]
    assert body["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_cors_does_not_allow_credentials_with_wildcard_origin():
    from orbitals.claim_extractor.serving import main as serving_main

    middleware = next(
        item
        for item in serving_main.app.user_middleware
        if item.cls.__name__ == "CORSMiddleware"
    )

    assert middleware.kwargs["allow_origins"] == ["*"]
    assert middleware.kwargs["allow_credentials"] is False
