"""Tests for ClaimExtractor pure helpers and public contracts."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orbitals.claim_extractor import AsyncClaimExtractor, ClaimExtractor
from orbitals.claim_extractor.extractors.base import MODEL_MAPPING
from orbitals.claim_extractor.extractors.vllm import _USE_DEFAULT_SPECULATIVE_CONFIG
from orbitals.claim_extractor.modeling import ClaimExtractorOutput, Extractions
from orbitals.claim_extractor.prompting import (
    NoEvidenceExtractionsResponseModel,
    convert_to_conversation,
    prepare_messages,
    validate_extractions_response,
)
from orbitals.types import AIServiceDescription


def test_claim_extractor_default_alias_resolves_to_4B_q_model():
    assert (
        ClaimExtractor.maybe_map_model("claim-extractor")
        == "principled-intelligence/claim-extractor-4B-q-2605"
    )


def test_claim_extractor_q_alias_resolves_to_4B_q_model():
    assert (
        ClaimExtractor.maybe_map_model("claim-extractor-q")
        == "principled-intelligence/claim-extractor-4B-q-2605"
    )


def test_claim_extractor_4B_q_alias_resolves_to_4B_q_model():
    assert (
        ClaimExtractor.maybe_map_model("claim-extractor-4B-q")
        == "principled-intelligence/claim-extractor-4B-q-2605"
    )


def test_claim_extractor_2B_q_alias_resolves_to_2B_q_model():
    assert (
        ClaimExtractor.maybe_map_model("claim-extractor-2B-q")
        == "principled-intelligence/claim-extractor-2B-q-2605"
    )


def test_claim_extractor_unknown_model_name_is_passed_through_unchanged():
    custom_name = "my-org/my-custom-model"
    assert ClaimExtractor.maybe_map_model(custom_name) == custom_name


def test_claim_extractor_model_mapping_contains_documented_aliases():
    assert "claim-extractor" in MODEL_MAPPING
    assert "claim-extractor-q" in MODEL_MAPPING
    assert "claim-extractor-4B-q" in MODEL_MAPPING
    assert "claim-extractor-2B-q" in MODEL_MAPPING


def test_no_evidence_response_validates_and_converts_to_full_extractions():
    validated = validate_extractions_response(
        {
            "extractions": {
                "intents": [{"content": "The user wants package tracking."}],
                "claims": [
                    {
                        "subtype": "Factoid",
                        "content": "The package arrives on December 12, 2025.",
                    }
                ],
            }
        },
        skip_evidences=True,
    )

    assert isinstance(validated.extractions, Extractions)
    assert validated.extractions.intents[0].evidences == []
    assert validated.extractions.claims[0].evidences == []


def test_get_extractions_response_model_uses_no_evidence_schema():
    validated = NoEvidenceExtractionsResponseModel.model_validate(
        {"extractions": {"intents": [], "claims": []}}
    )

    converted = validated.to_extractions_response()

    assert isinstance(converted.extractions, Extractions)


def test_prepare_messages_serializes_structured_ai_service_description():
    description = AIServiceDescription(
        identity_role="Parcel delivery assistant",
        context="Online logistics.",
    )

    messages = prepare_messages(
        "Your package is in transit.",
        description,
        skip_evidences=True,
    )

    user_content = messages[1]["content"]
    assert '"identity_role":"Parcel delivery assistant"' in user_content
    assert "LAST MESSAGE (ASSISTANT)" in user_content


def test_convert_to_conversation_rejects_unknown_roles():
    with pytest.raises(ValueError):
        convert_to_conversation([{"role": "system", "content": "Nope."}])


def test_async_api_backend_aliases_are_not_advertised_as_constructible():
    with pytest.raises(ValueError):
        AsyncClaimExtractor(backend="async-api")

    with pytest.raises(ValueError):
        AsyncClaimExtractor(backend="vllm-async-api")


def test_vllm_default_speculative_config_sentinel_is_internal_only():
    ce = ClaimExtractor.__new__(
        ClaimExtractor,
        backend="vllm",
        speculative_config=_USE_DEFAULT_SPECULATIVE_CONFIG,
    )

    assert ce is not None


def _install_fake_pipeline_modules(monkeypatch: pytest.MonkeyPatch):
    class _FakePipeline:
        def __init__(self, model, tokenizer=None, **kwargs):
            self.model = model
            self.tokenizer = tokenizer
            self.device = "cpu"

    fake_torch = types.SimpleNamespace(inference_mode=lambda: _NullContext())
    fake_transformers = types.SimpleNamespace(Pipeline=_FakePipeline)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_claim_extractor_pipeline_forwards_false_constructor_default(monkeypatch):
    _install_fake_pipeline_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("hf_pipeline.claim_extractor"))
    pipeline = module.ClaimExtractionPipeline(
        model=object(),
        tokenizer=types.SimpleNamespace(pad_token="</s>"),
        skip_evidences=False,
    )

    preprocess_kwargs, _, _ = pipeline._sanitize_parameters()

    assert preprocess_kwargs == {"skip_evidences": False}


def test_claim_extractor_pipeline_allows_per_call_skip_evidences_override(monkeypatch):
    _install_fake_pipeline_modules(monkeypatch)
    module = importlib.reload(importlib.import_module("hf_pipeline.claim_extractor"))
    pipeline = module.ClaimExtractionPipeline(
        model=object(),
        tokenizer=types.SimpleNamespace(pad_token="</s>"),
        skip_evidences=True,
    )

    preprocess_kwargs, _, _ = pipeline._sanitize_parameters(skip_evidences=False)

    assert preprocess_kwargs == {"skip_evidences": False}


def _extract_response_payload(
    *,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "extractions": {"intents": [], "claims": []},
        "model": "principled-intelligence/claim-extractor-4B-q-2605",
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def mocked_claim_extractor_post():
    with patch("orbitals.claim_extractor.extractors.api.requests.post") as mocked:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = _extract_response_payload()
        mocked.return_value = response
        yield mocked


def test_claim_extractor_api_serializes_structured_ai_service_description(
    mocked_claim_extractor_post,
):
    ce = ClaimExtractor(backend="api", api_url="http://example.com")
    svc = AIServiceDescription(
        identity_role="Parcel delivery assistant",
        context="Online logistics.",
    )

    ce.extract("hi", ai_service_description=svc)

    body = mocked_claim_extractor_post.call_args.kwargs["json"]
    assert body["ai_service_description"] == {
        "identity_role": "Parcel delivery assistant",
        "context": "Online logistics.",
        "knowledge_scope": None,
        "functionalities": None,
        "principles": None,
        "website_url": None,
    }


def test_claim_extractor_api_does_not_mutate_caller_headers(
    mocked_claim_extractor_post,
):
    custom_headers = {"X-API-Key": "from_headers", "X-Trace": "abc"}

    ce = ClaimExtractor(
        backend="api",
        api_url="http://example.com",
        custom_headers=custom_headers,
    )
    ce.extract("hi")

    assert custom_headers == {"X-API-Key": "from_headers", "X-Trace": "abc"}
    headers = mocked_claim_extractor_post.call_args.kwargs["headers"]
    assert headers["X-API-Key"] == "from_headers"


def test_claim_extractor_api_parses_response_into_output(mocked_claim_extractor_post):
    ce = ClaimExtractor(backend="api", api_url="http://example.com")

    result = ce.extract("hi")

    assert isinstance(result, ClaimExtractorOutput)
    assert result.extractions == Extractions()
    assert result.model == "principled-intelligence/claim-extractor-4B-q-2605"
