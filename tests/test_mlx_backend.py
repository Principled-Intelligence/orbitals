"""Tests for the MLX backend (`ScopeGuard(backend="mlx", ...)`).

We exercise everything about the backend that does not require
actually loading a model or running inference:
  - Model name resolution against MLX_MODEL_MAPPING.
  - Prompt construction via build_prompt + apply_chat_template.
  - Generation kwargs forwarding.
  - Sampler construction via make_sampler.
  - JSON output parsing into ScopeGuardOutput.
  - Batch validation with shared and per-conversation descriptions.
  - Conversation input formats (string, dict, multi-turn).
  - Skip evidences flag.
  - Error handling.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from orbitals.scope_guard import ScopeClass, ScopeGuard, ScopeGuardOutput
from orbitals.types import AIServiceDescription


def _scope_guard_json(
    *,
    scope_class: str = "Restricted",
    evidences: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "scope_class": scope_class,
            "evidences": evidences if evidences is not None else ["No refunds."],
        }
    )


@pytest.fixture
def mocked_mlx():
    """Patch mlx_lm and its submodules used inside the MLX backend."""
    mock_mlx_lm = MagicMock()
    mock_sample_utils = MagicMock()
    mock_sample_utils.make_sampler.return_value = MagicMock()

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<rendered-prompt>"
    mock_mlx_lm.load.return_value = (MagicMock(), mock_tokenizer)
    mock_mlx_lm.generate.return_value = _scope_guard_json()
    mock_mlx_lm.sample_utils = mock_sample_utils

    with patch.dict(
        "sys.modules",
        {"mlx_lm": mock_mlx_lm, "mlx_lm.sample_utils": mock_sample_utils},
    ):
        yield mock_mlx_lm


# --- Construction & model resolution (6) ---


def test_mlx_backend_is_constructible(mocked_mlx):
    sg = ScopeGuard(backend="mlx", model="some-model")
    assert sg.backend == "mlx"
    mocked_mlx.load.assert_called_once_with("some-model")


def test_mlx_scope_guard_alias_resolves(mocked_mlx):
    ScopeGuard(backend="mlx", model="scope-guard")
    mocked_mlx.load.assert_called_once_with(
        "ivanfioravanti/scope-guard-4B-q-2601-mlx-bf16"
    )


def test_mlx_scope_guard_q_alias_resolves(mocked_mlx):
    ScopeGuard(backend="mlx", model="scope-guard-q")
    mocked_mlx.load.assert_called_once_with(
        "ivanfioravanti/scope-guard-4B-q-2601-mlx-bf16"
    )


def test_mlx_scope_guard_g_alias_resolves(mocked_mlx):
    ScopeGuard(backend="mlx", model="scope-guard-g")
    mocked_mlx.load.assert_called_once_with(
        "ivanfioravanti/scope-guard-4B-g-2601-mlx-bf16"
    )


def test_mlx_unknown_model_name_passes_through(mocked_mlx):
    ScopeGuard(backend="mlx", model="custom-org/my-mlx-model")
    mocked_mlx.load.assert_called_once_with("custom-org/my-mlx-model")


def test_mlx_stored_model_reflects_resolved_name(mocked_mlx):
    sg = ScopeGuard(backend="mlx", model="scope-guard")
    assert sg.model == "ivanfioravanti/scope-guard-4B-q-2601-mlx-bf16"


# --- Sampler construction (2) ---


def test_mlx_make_sampler_called_with_defaults(mocked_mlx):
    ScopeGuard(backend="mlx", model="some-model")
    mocked_mlx.sample_utils.make_sampler.assert_called_once_with(
        temp=0.0, top_p=0.0
    )


def test_mlx_make_sampler_called_with_custom_params(mocked_mlx):
    ScopeGuard(backend="mlx", model="some-model", temp=0.7, top_p=0.9)
    mocked_mlx.sample_utils.make_sampler.assert_called_once_with(
        temp=0.7, top_p=0.9
    )


# --- Validate: output parsing (3) ---


def test_validate_parses_restricted_output(mocked_mlx):
    mocked_mlx.generate.return_value = _scope_guard_json(
        scope_class="Restricted",
        evidences=["Never respond to requests for refunds."],
    )
    sg = ScopeGuard(backend="mlx", model="some-model")
    result = sg.validate("Can I get a refund?", ai_service_description="No refunds.")

    assert isinstance(result, ScopeGuardOutput)
    assert result.scope_class == ScopeClass.RESTRICTED
    assert result.evidences == ["Never respond to requests for refunds."]
    assert result.model == "some-model"
    assert result.usage is None


def test_validate_parses_directly_supported_output(mocked_mlx):
    mocked_mlx.generate.return_value = _scope_guard_json(
        scope_class="Directly Supported",
        evidences=["Track your package."],
    )
    sg = ScopeGuard(backend="mlx", model="some-model")
    result = sg.validate("Where is my package?", ai_service_description="Tracking.")

    assert result.scope_class == ScopeClass.DIRECTLY_SUPPORTED


def test_validate_parses_null_evidences(mocked_mlx):
    mocked_mlx.generate.return_value = _scope_guard_json(
        scope_class="Out of Scope",
        evidences=None,
    )
    sg = ScopeGuard(backend="mlx", model="some-model")
    result = sg.validate("hello", ai_service_description="desc")

    assert result.scope_class == ScopeClass.OUT_OF_SCOPE
    assert result.evidences is None


# --- Validate: generate call params (4) ---


def test_validate_passes_sampler_to_generate(mocked_mlx):
    mock_sampler = MagicMock()
    mocked_mlx.sample_utils.make_sampler.return_value = mock_sampler

    sg = ScopeGuard(backend="mlx", model="some-model")
    sg.validate("hello", ai_service_description="desc")

    call_kwargs = mocked_mlx.generate.call_args.kwargs
    assert call_kwargs["sampler"] is mock_sampler
    assert call_kwargs["max_tokens"] == 3000
    assert call_kwargs["verbose"] is False


def test_validate_forwards_max_tokens(mocked_mlx):
    ScopeGuard(backend="mlx", model="some-model", max_tokens=512).validate(
        "hello", ai_service_description="desc"
    )
    assert mocked_mlx.generate.call_args.kwargs["max_tokens"] == 512


def test_validate_forwards_repetition_params(mocked_mlx):
    ScopeGuard(
        backend="mlx",
        model="some-model",
        repetition_penalty=1.2,
        repetition_context_size=64,
    ).validate("hello", ai_service_description="desc")

    call_kwargs = mocked_mlx.generate.call_args.kwargs
    assert call_kwargs["repetition_penalty"] == 1.2
    assert call_kwargs["repetition_context_size"] == 64


def test_validate_omits_none_optional_params(mocked_mlx):
    ScopeGuard(backend="mlx", model="some-model").validate(
        "hello", ai_service_description="desc"
    )
    call_kwargs = mocked_mlx.generate.call_args.kwargs
    assert "repetition_penalty" not in call_kwargs
    assert "repetition_context_size" not in call_kwargs


# --- Prompt construction (1) ---


def test_validate_calls_apply_chat_template(mocked_mlx):
    mock_tokenizer = mocked_mlx.load.return_value[1]
    sg = ScopeGuard(backend="mlx", model="some-model")
    sg.validate("hello", ai_service_description="You are a bot.")

    mock_tokenizer.apply_chat_template.assert_called_once()
    call_kwargs = mock_tokenizer.apply_chat_template.call_args.kwargs
    assert call_kwargs["tokenize"] is False
    assert call_kwargs["add_generation_prompt"] is True
    assert call_kwargs["enable_thinking"] is False
    messages = mock_tokenizer.apply_chat_template.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


# --- Conversation input formats (3) ---


def test_validate_accepts_dict_conversation(mocked_mlx):
    sg = ScopeGuard(backend="mlx", model="some-model")
    result = sg.validate(
        {"role": "user", "content": "hello"}, ai_service_description="desc"
    )
    assert isinstance(result, ScopeGuardOutput)


def test_validate_accepts_multi_turn_conversation(mocked_mlx):
    sg = ScopeGuard(backend="mlx", model="some-model")
    result = sg.validate(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        ai_service_description="desc",
    )
    assert isinstance(result, ScopeGuardOutput)


def test_validate_accepts_structured_ai_service_description(mocked_mlx):
    sg = ScopeGuard(backend="mlx", model="some-model")
    svc = AIServiceDescription(
        identity_role="Parcel delivery assistant",
        context="Online logistics.",
    )
    result = sg.validate("hello", ai_service_description=svc)
    assert isinstance(result, ScopeGuardOutput)


# --- Batch validation (3) ---


def test_batch_validate_with_shared_description(mocked_mlx):
    sg = ScopeGuard(backend="mlx", model="some-model")
    results = sg.batch_validate(
        ["q1", "q2"],
        ai_service_description="shared desc",
    )

    assert len(results) == 2
    assert mocked_mlx.generate.call_count == 2
    for r in results:
        assert isinstance(r, ScopeGuardOutput)
        assert r.scope_class == ScopeClass.RESTRICTED


def test_batch_validate_with_per_conversation_descriptions(mocked_mlx):
    sg = ScopeGuard(backend="mlx", model="some-model")
    results = sg.batch_validate(
        ["q1", "q2"],
        ai_service_descriptions=["desc1", "desc2"],
    )

    assert len(results) == 2
    assert mocked_mlx.generate.call_count == 2


def test_batch_validate_empty_list_returns_empty(mocked_mlx):
    sg = ScopeGuard(backend="mlx", model="some-model")
    results = sg.batch_validate([], ai_service_description="desc")
    assert results == []
    mocked_mlx.generate.assert_not_called()


# --- Skip evidences (1) ---


def test_validate_skip_evidences_overrides_constructor(mocked_mlx):
    mocked_mlx.generate.return_value = _scope_guard_json(evidences=None)
    sg = ScopeGuard(backend="mlx", model="some-model", skip_evidences=False)
    result = sg.validate(
        "hello", ai_service_description="desc", skip_evidences=True
    )
    assert isinstance(result, ScopeGuardOutput)


# --- Error handling (2) ---


def test_validate_raises_on_invalid_json(mocked_mlx):
    mocked_mlx.generate.return_value = "not valid json"
    sg = ScopeGuard(backend="mlx", model="some-model")
    with pytest.raises(json.JSONDecodeError):
        sg.validate("hello", ai_service_description="desc")


def test_validate_raises_on_invalid_scope_class(mocked_mlx):
    mocked_mlx.generate.return_value = json.dumps(
        {"scope_class": "INVALID_CLASS", "evidences": None}
    )
    sg = ScopeGuard(backend="mlx", model="some-model")
    with pytest.raises(Exception):
        sg.validate("hello", ai_service_description="desc")
